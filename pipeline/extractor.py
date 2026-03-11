"""
Step 2: Per-Page Code Extractor

For EACH page independently (order-agnostic):
  a) Extract anchors (patientId, barcode, orderId, sampleType, dates, facility, doctor)
  b) Extract department + panel from section headers
  c) Classify page type (RESULT_TABLE / COMMENTARY / MARKETING / NARRATIVE / ADMIN)
  d) If RESULT_TABLE: parse HTML tables → structured test rows
  e) Handle non-standard result formats (inline text results, qualitative)
"""

import re
from html import unescape
from typing import List, Dict, Optional, Any, Tuple
from . import config


# ════════════════════════════════════════════════════════════
#  ANCHOR EXTRACTION
# ════════════════════════════════════════════════════════════

def _clean_value(val: str) -> str:
    """Strip HTML tags, entities, and whitespace from extracted values."""
    val = re.sub(r'<[^>]+>', '', val)
    val = unescape(val)
    return val.strip().rstrip('.')


def extract_anchors(lines: List[str]) -> Dict[str, str]:
    """
    Extract patient/report anchors from a page's lines.
    Uses a two-pass approach:
      Pass 1: Look for key-value pairs in <td> structures (Tata 1mg style)
      Pass 2: Look for key: value patterns in plain text
    """
    anchors = {}
    full_text = '\n'.join(lines)

    # ── Pass 1: HTML <td> key-value pairs ──
    # Pattern: <td>Label</td> <td>: Value</td>
    for i in range(len(lines) - 1):
        curr = lines[i].strip()
        nxt = lines[i + 1].strip()

        # Check if current line is a label td
        label_m = re.match(r'<td[^>]*>(.*?)</td>', curr)
        value_m = re.match(r'<td[^>]*>:\s*(.*?)(?:</td>)?$', nxt)

        if label_m and value_m:
            label = _clean_value(label_m.group(1)).lower()
            value = _clean_value(value_m.group(1))

            if not value:
                continue

            if 'patient id' in label or 'uhid' in label or 'mrn' in label:
                anchors['patientId'] = value
            elif 'barcode' in label:
                # May contain "D19907243 / 12827947"
                parts = re.split(r'\s*/\s*', value)
                anchors['barcodeId'] = parts[0].strip()
                if len(parts) > 1:
                    anchors['orderId'] = parts[1].strip()
            elif label in ('order id', 'order no', 'po no', 'bill no', 'lab no', 'sample no'):
                anchors['orderId'] = value
            elif 'name' == label or 'patient name' in label:
                anchors['patientName'] = value
            elif 'client name' in label:
                anchors['facility'] = value
            elif 'age' in label and 'gender' in label:
                # "27/Male" or "27 Years / Male"
                age_m = re.match(r'(\d+)\s*(?:Years?|Y|y)?\s*[/,]\s*(\w+)', value)
                if age_m:
                    anchors['age'] = age_m.group(1)
                    anchors['gender'] = age_m.group(2)
            elif 'gender' in label:
                anchors['gender'] = value
            elif 'age' in label:
                anchors['age'] = value
            elif 'collection date' in label or 'sample collected' in label:
                anchors['collectionDate'] = value
            elif 'report date' in label or 'reported on' in label:
                anchors['reportDate'] = value
            elif 'received' in label and 'date' in label:
                anchors['receivedDate'] = value
            elif 'sample type' in label or 'specimen' in label:
                anchors['sampleType'] = value
            elif 'referred by' in label or 'consultant' in label:
                anchors['doctor'] = value
            elif 'report status' in label:
                anchors['reportStatus'] = value

    # ── Pass 2: Plain text key: value patterns ──
    for line in lines:
        s = line.strip()

        # Name patterns
        if 'patientName' not in anchors:
            m = re.search(r'(?:Name|Patient)\s*:\s*((?:Mr\.|Mrs\.|Ms\.|Dr\.|Master\.)?\s*[A-Za-z\s.]+)', s)
            if m:
                anchors['patientName'] = _clean_value(m.group(1))

        # Patient ID patterns
        if 'patientId' not in anchors:
            m = re.search(r'(?:Patient\s*ID|UHID|MRN|CR\s*No|Reg\.?\s*No)\s*:\s*(\S+)', s)
            if m:
                anchors['patientId'] = _clean_value(m.group(1))

        # Lab/Order/Bill No
        if 'orderId' not in anchors:
            m = re.search(r'(?:Lab\s*No|Order\s*(?:ID|No)|Bill\s*No|Sample\s*No)\s*:\s*(\S+)', s)
            if m:
                anchors['orderId'] = _clean_value(m.group(1))

        # Facility from known patterns
        if 'facility' not in anchors:
            for kw in ['TATA 1MG', 'Dr Lal', 'Max Healthcare', 'Kapil', 'Sarvodaya',
                        'Lady Hardinge', 'NDMC', 'Apollo', 'Fortis', 'SRL', 'Metropolis',
                        'Thyrocare', 'Pathkind', 'Lala Ganga Ram']:
                if kw.upper() in s.upper():
                    anchors['facility'] = kw
                    break

    # ── Pass 3: Facility from header/title lines ──
    if 'facility' not in anchors:
        for line in lines[:10]:  # Usually in first few lines
            s = line.strip()
            if s.startswith('# ') and len(s) > 3:
                anchors['facility'] = s[2:].strip()
                break

    return anchors


# ════════════════════════════════════════════════════════════
#  DEPARTMENT + PANEL EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_department_and_panels(lines: List[str]) -> Tuple[str, List[Dict]]:
    """
    Extract department name and panel sections from a page.
    Returns (department, panels) where panels is a list of
    {'name': str, 'start_line': int}.

    Handles multiple formats:
      - ## HAEMATOLOGY (markdown h2)
      - <td colspan="5"><strong>Panel Name</strong></td>
      - <td>**Panel Name**</td>  (markdown bold in HTML)
      - ### Panel Name
    """
    department = ""
    panels = []

    for i, line in enumerate(lines):
        s = line.strip()

        # Department: ## HAEMATOLOGY, ## BIOCHEMISTRY, etc.
        if s.startswith('## '):
            heading = s[3:].strip()
            for dept in config.DEPARTMENTS:
                if dept in heading.upper():
                    department = dept
                    break

        # Panel from <strong> in colspan td
        strong_m = re.search(r'<strong>(.*?)</strong>', s)
        if strong_m and '<td' in s:
            panel_name = _clean_value(strong_m.group(1))
            if panel_name and len(panel_name) > 2:
                panels.append({'name': panel_name, 'start_line': i})

        # Panel from **bold** in td (Tata 1mg thyroid style)
        bold_m = re.search(r'\*\*(.*?)\*\*', s)
        if bold_m and '<td' in s and not strong_m:
            panel_name = _clean_value(bold_m.group(1))
            if panel_name and len(panel_name) > 2:
                panels.append({'name': panel_name, 'start_line': i})

        # Panel from ### heading (less common)
        if s.startswith('### ') and 'Comment' not in s and 'Note' not in s and 'Factor' not in s:
            heading = s[4:].strip()
            # Skip non-panel headings
            if heading.upper() not in ('COMMENT:', 'COMMENTS:', 'NOTE:', 'FACTORS THAT'):
                # Check if it looks like a medical panel name (not a generic heading)
                if any(c.isupper() for c in heading) and len(heading) > 3:
                    panels.append({'name': heading, 'start_line': i})

    return department, panels


# ════════════════════════════════════════════════════════════
#  PAGE CLASSIFICATION
# ════════════════════════════════════════════════════════════

def classify_page(lines: List[str], anchors: Dict) -> Dict[str, Any]:
    """
    Classify a page into a type. Returns:
    {
        'page_type': RESULT_TABLE | COMMENTARY | MARKETING | NARRATIVE | ADMIN | MIXED,
        'has_results': bool,
        'has_commentary': bool,
        'has_marketing': bool,
        'hiType_hint': str or None,
        'hiType_confidence': float,
    }
    """
    full_text = ' '.join(lines)
    upper_text = full_text.upper()

    has_table = '<table>' in full_text.lower() or '<table ' in full_text.lower()
    has_result_values = bool(re.search(r'<td>\d+\.?\d*</td>', full_text))
    has_commentary = any(kw in full_text for kw in ['Comment:', 'Reference:', 'Note:', 'guidelines'])
    has_marketing = any(kw.upper() in upper_text for kw in config.MARKETING_KEYWORDS)
    has_disclaimer = any(kw.upper() in upper_text for kw in config.DISCLAIMER_KEYWORDS)

    # Detect inline result lines (e.g., "Vitamin D (25-OH) 20.1 ng/ml")
    has_inline_results = bool(re.search(
        r'(?:Vitamin|Hemoglobin|Glucose|Cholesterol|Creatinine|TSH|T3|T4|HbA1c|ESR|CRP)'
        r'.*?\d+\.?\d*\s*(?:ng/ml|mg/dl|g/dL|µg/dL|IU/L|U/L|mEq/L|%|mm/hr|pg/ml)',
        full_text, re.IGNORECASE
    ))

    # HI Type hints based on content signals
    hiType_hint = None
    hiType_confidence = 0.0

    for hi_type, signals in config.HITYPE_SIGNALS.items():
        score = sum(1 for sig in signals if sig.upper() in upper_text)
        normalized = score / len(signals) if signals else 0
        if normalized > hiType_confidence:
            hiType_confidence = normalized
            hiType_hint = hi_type

    # If confidence is too low, don't guess — mark as UNKNOWN
    if hiType_confidence < 0.10:
        hiType_hint = None
        hiType_confidence = 0.0

    # Determine page type
    if has_marketing and not has_result_values:
        page_type = 'MARKETING'
    elif has_result_values or has_inline_results:
        if has_commentary:
            page_type = 'RESULT_WITH_COMMENTARY'
        else:
            page_type = 'RESULT_TABLE'
    elif has_commentary and not has_result_values:
        page_type = 'COMMENTARY'
    elif has_table and not has_result_values:
        # Table but no numeric results — could be reference/interpretation table
        page_type = 'REFERENCE_TABLE'
    elif has_disclaimer and not has_result_values:
        page_type = 'ADMIN'
    else:
        # Check for narrative medical content
        narrative_signals = ['diagnosis', 'complaint', 'history', 'examination',
                             'medication', 'advised', 'follow', 'prescription',
                             'Rx', 'Tab', 'Cap', 'mg']
        if any(sig.lower() in full_text.lower() for sig in narrative_signals):
            page_type = 'NARRATIVE'
        else:
            page_type = 'OTHER'

    return {
        'page_type': page_type,
        'has_results': has_result_values or has_inline_results,
        'has_commentary': has_commentary,
        'has_marketing': has_marketing,
        'hiType_hint': hiType_hint,
        'hiType_confidence': round(hiType_confidence, 2),
    }


# ════════════════════════════════════════════════════════════
#  HTML TABLE PARSER → STRUCTURED TEST ROWS
# ════════════════════════════════════════════════════════════

def _parse_range(display_range: str) -> Dict[str, Any]:
    """
    Parse a reference range string into structured form.
    Handles: "13.0-17.0", "< 200", ">= 60", multi-tier ranges.
    """
    result = {
        'display_range': display_range,
        'low': None,
        'high': None,
        'range_type': 'unknown',  # simple | upper_only | lower_only | multi_tier | none
    }

    if not display_range:
        result['range_type'] = 'none'
        return result

    # Simple range: "13.0-17.0" or "13.0 - 17.0" or "13.0–17.0"
    simple_m = re.match(r'^([\d.]+)\s*[-–]\s*([\d.]+)$', display_range.strip())
    if simple_m:
        result['low'] = float(simple_m.group(1))
        result['high'] = float(simple_m.group(2))
        result['range_type'] = 'simple'
        return result

    # Upper only: "< 200" or "<34" or "<=200"
    upper_m = re.match(r'^[<≤]\s*=?\s*([\d.]+)$', display_range.strip())
    if upper_m:
        result['high'] = float(upper_m.group(1))
        result['range_type'] = 'upper_only'
        return result

    # Lower only: "> 0.3" or ">=60"
    lower_m = re.match(r'^[>≥]\s*=?\s*([\d.]+)$', display_range.strip())
    if lower_m:
        result['low'] = float(lower_m.group(1))
        result['range_type'] = 'lower_only'
        return result

    # Multi-tier (lipid style): contains multiple thresholds with labels
    if any(kw in display_range.lower() for kw in ['desirable', 'borderline', 'high', 'low', 'normal']):
        result['range_type'] = 'multi_tier'
        # Try to extract the "desirable" or "normal" range
        desirable_m = re.search(r'(?:Desirable|Normal|Low\s*risk)[^<>]*[<:]\s*([\d.]+)', display_range)
        if desirable_m:
            result['high'] = float(desirable_m.group(1))
        desirable_low = re.search(r'(?:Desirable|Normal)[^<>]*([\d.]+)\s*[-–]\s*([\d.]+)', display_range)
        if desirable_low:
            result['low'] = float(desirable_low.group(1))
            result['high'] = float(desirable_low.group(2))
        return result

    # Ratio ranges: "12:1 - 20:1"
    ratio_m = re.match(r'([\d.]+):1\s*[-–]\s*([\d.]+):1', display_range.strip())
    if ratio_m:
        result['low'] = float(ratio_m.group(1))
        result['high'] = float(ratio_m.group(2))
        result['range_type'] = 'simple'
        return result

    return result


def _compute_abnormality(value_str: str, parsed_range: Dict) -> Optional[bool]:
    """Determine if a result is abnormal based on value vs range."""
    if parsed_range['range_type'] == 'none' or parsed_range['range_type'] == 'unknown':
        return None

    try:
        value = float(value_str)
    except (ValueError, TypeError):
        return None

    if parsed_range['range_type'] == 'simple':
        if parsed_range['low'] is not None and parsed_range['high'] is not None:
            return value < parsed_range['low'] or value > parsed_range['high']

    if parsed_range['range_type'] == 'upper_only':
        if parsed_range['high'] is not None:
            return value >= parsed_range['high']

    if parsed_range['range_type'] == 'lower_only':
        if parsed_range['low'] is not None:
            return value < parsed_range['low']

    if parsed_range['range_type'] == 'multi_tier':
        # Use extracted desirable thresholds if available
        if parsed_range['high'] is not None and parsed_range['low'] is not None:
            return value < parsed_range['low'] or value > parsed_range['high']
        elif parsed_range['high'] is not None:
            return value > parsed_range['high']

    return None


def extract_table_results(lines: List[str], department: str, panels: List[Dict]) -> List[Dict]:
    """
    Parse HTML table rows into structured test results.
    Each result: {test_name, value, unit, display_range, method, panel, department,
                  is_abnormal, parsed_range, scale_type}
    """
    results = []
    current_panel = panels[0]['name'] if panels else ""

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        # Track panel changes from colspan/strong rows
        if '<td' in s:
            strong_m = re.search(r'<strong>(.*?)</strong>', s)
            bold_m = re.search(r'\*\*(.*?)\*\*', s) if not strong_m else None
            colspan_m = re.search(r'colspan', s)

            if (strong_m or bold_m) and colspan_m:
                panel_name = _clean_value((strong_m or bold_m).group(1))
                if panel_name and len(panel_name) > 2:
                    current_panel = panel_name

            # Also catch panel in non-colspan bold td
            if (strong_m or bold_m) and not colspan_m:
                candidate = _clean_value((strong_m or bold_m).group(1))
                # Heuristic: if this td is the only content in a row, it's a panel header
                # Check surrounding lines for empty tds
                if i + 1 < len(lines) and re.match(r'<td[^>]*>\s*</td>', lines[i + 1].strip()):
                    current_panel = candidate

        # Detect table row start
        if s == '<tr>':
            row_cells = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != '</tr>' and len(row_cells) < 8:
                cell_m = re.match(r'<td[^>]*>(.*?)(?:</td>)?$', lines[j].strip())
                if cell_m:
                    cell_val = unescape(cell_m.group(1).strip())
                    # Strip nested HTML tags
                    cell_val = re.sub(r'<[^>]+>', ' ', cell_val).strip()
                    row_cells.append(cell_val)
                j += 1

            # 5-column result row: TestName, Value, Unit, Range, Method
            if len(row_cells) == 5:
                test_name, value, unit, ref_range, method = row_cells

                # Check if this is a result row (value is numeric or a recognized qualitative)
                is_numeric = bool(re.match(r'^[\d.]+$', value))
                is_qualitative = value.lower() in (
                    'positive', 'negative', 'reactive', 'non-reactive',
                    'detected', 'not detected', 'present', 'absent',
                    'normal', 'abnormal', 'trace', 'nil',
                    'yellow', 'straw', 'amber', 'clear', 'turbid', 'hazy',
                )

                if (is_numeric or is_qualitative) and test_name and test_name != 'Test Name':
                    parsed_range = _parse_range(ref_range)
                    is_abnormal = _compute_abnormality(value, parsed_range) if is_numeric else None

                    scale_type = 'Qn' if is_numeric else 'Ord'

                    results.append({
                        'test_name': test_name,
                        'value': float(value) if is_numeric else None,
                        'value_string': value,
                        'unit': unit,
                        'display_range': ref_range,
                        'method': method if method else None,
                        'panel': current_panel,
                        'department': department,
                        'is_abnormal': is_abnormal,
                        'parsed_range': parsed_range,
                        'scale_type': scale_type,
                    })

            # 4-column result row (some labs skip Method column)
            elif len(row_cells) == 4:
                test_name, value, unit, ref_range = row_cells
                is_numeric = bool(re.match(r'^[\d.]+$', value))

                if is_numeric and test_name and test_name != 'Test Name':
                    parsed_range = _parse_range(ref_range)
                    is_abnormal = _compute_abnormality(value, parsed_range)

                    results.append({
                        'test_name': test_name,
                        'value': float(value),
                        'value_string': value,
                        'unit': unit,
                        'display_range': ref_range,
                        'method': None,
                        'panel': current_panel,
                        'department': department,
                        'is_abnormal': is_abnormal,
                        'parsed_range': parsed_range,
                        'scale_type': 'Qn',
                    })

        i += 1

    return results


# ════════════════════════════════════════════════════════════
#  INLINE RESULT PARSER (for non-table results)
# ════════════════════════════════════════════════════════════

def extract_inline_results(lines: List[str], department: str) -> List[Dict]:
    """
    Extract results that appear as inline text rather than HTML tables.
    E.g.: "Vitamin D (25-OH) 20.1 ng/ml Deficiency: < 20, CLIA"
    """
    results = []
    full_text = '\n'.join(lines)

    # Pattern: TestName Value Unit [additional info]
    inline_patterns = [
        # "Vitamin D (25-OH) 20.1 ng/ml Deficiency: < 20, CLIA"
        (r'(Vitamin\s*D\s*\(25-OH\))\s+([\d.]+)\s+(ng/ml?)\s*(.*?)(?:,\s*(\w+))?$',
         {'panel': 'Vitamin D'}),
        # "Vitamin B12 561.0 pg/ml"
        (r'(Vitamin\s*B12)\s+([\d.]+)\s+(pg/ml?)\s*(.*?)$',
         {'panel': 'Vitamin B12'}),
        # "Serum Ferritin: 45.2 ng/mL (Ref: 20-200)" — colon-separated
        (r'([\w\s()-]+?):\s*([\d.]+)\s*((?:ng|mg|µg|pg|g|IU|U|mEq|mmol|µmol|nmol)/(?:ml|mL|dL|L))\s*(?:\((?:Ref:?)?\s*(.*?)\))?',
         {}),
        # "HBA1C: 7.2% (Normal <5.7 ...)" — percentage with colon
        (r'(HbA1c|HBA1C|eGFR|CRP|ESR)\s*:?\s*([\d.]+)\s*(%|mm/hr|mg/L|mL/min)\s*(.*?)$',
         {}),
        # "Hemoglobin  14.2 g/dL  (13-17)" — space-separated with parenthesized range
        (r'^([\w\s()-]+?)\s{2,}([\d.]+)\s+([\w/^.µ]+)\s+\(([\d.\s-]+)\)\s*$',
         {}),
        # "WBC  6500 /cumm  (4000-11000)" — space-separated
        (r'^([\w\s]+?)\s{2,}([\d.]+)\s+([\w/^.µ]+)\s+\(([\d.\s-]+)\)\s*$',
         {}),
        # "Platelet  2.1 lakhs  (1.5-4.0)"
        (r'^([\w\s]+?)\s{2,}([\d.]+)\s+(lakhs?|[\w/^.µ]+)\s+\(([\d.\s-]+)\)\s*$',
         {}),
        # Generic: "TestName Value Unit" (no range)
        (r'^([\w\s()-]+?)\s{2,}([\d.]+)\s+((?:ng|mg|µg|pg|g|IU|U|mEq|mmol|µmol|nmol)/(?:ml|mL|dL|L)|g/dL|%|mm/hr|fL|pg|lakhs?|/cumm|10\^[\d]/[µu]?L)',
         {}),
        # "Vitamin D3: 18.5 nmol/L (Ref: 50-125) LOW" — with trailing qualifier
        (r'([\w\s()-]+?):\s*([\d.]+)\s*([\w/]+)\s*(?:\((?:Ref:?)?\s*(.*?)\))?\s*(LOW|HIGH|ABNORMAL|CRITICAL)?$',
         {}),
    ]

    for line in lines:
        s = line.strip()
        for pattern, extra in inline_patterns:
            m = re.search(pattern, s, re.IGNORECASE)
            if m:
                test_name = m.group(1).strip()
                value = m.group(2)
                unit = m.group(3)
                range_info = m.group(4).strip() if m.lastindex >= 4 else ''
                method = m.group(5).strip() if m.lastindex >= 5 else None

                # Parse range from range_info
                range_m = re.search(r'(?:Deficiency|Low|Normal|High)?\s*:\s*[<>]?\s*[\d.]+', range_info)
                display_range = range_info if range_m else ''

                parsed_range = _parse_range(display_range)
                is_abnormal = _compute_abnormality(value, parsed_range)

                results.append({
                    'test_name': test_name,
                    'value': float(value),
                    'value_string': value,
                    'unit': unit,
                    'display_range': display_range,
                    'method': method,
                    'panel': extra.get('panel', ''),
                    'department': department,
                    'is_abnormal': is_abnormal,
                    'parsed_range': parsed_range,
                    'scale_type': 'Qn',
                })
                break  # Don't double-match

    return results


# ════════════════════════════════════════════════════════════
#  MAIN PER-PAGE EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_page(page: Dict) -> Dict:
    """
    Main entry point. Takes a segmented page dict and returns enriched data.

    Returns:
    {
        'page_id': str,
        'page_num': int,
        'anchors': {patientId, barcodeId, orderId, sampleType, ...},
        'department': str,
        'panels': [{name, start_line}],
        'classification': {page_type, has_results, hiType_hint, ...},
        'results': [{test_name, value, unit, ...}],
        'needs_llm': bool,
        'llm_reason': str or None,
        'raw_text': str,
    }
    """
    lines = page['lines']

    # Step a: Anchors
    anchors = extract_anchors(lines)

    # Step b: Department + Panels
    department, panels = extract_department_and_panels(lines)

    # Step c: Classification
    classification = classify_page(lines, anchors)

    # Step d: Extract results (if applicable)
    results = []
    needs_llm = False
    llm_reason = None

    if classification['page_type'] in ('RESULT_TABLE', 'RESULT_WITH_COMMENTARY'):
        # Try HTML table extraction
        table_results = extract_table_results(lines, department, panels)
        results.extend(table_results)

        # Also try inline results (catches Vitamin D, etc.)
        inline_results = extract_inline_results(lines, department)
        # Deduplicate: don't add inline if we already got it from table
        existing_names = {r['test_name'].lower() for r in results}
        for ir in inline_results:
            if ir['test_name'].lower() not in existing_names:
                results.extend(inline_results)
                break

    elif classification['page_type'] == 'NARRATIVE':
        needs_llm = True
        llm_reason = 'narrative_extraction'

    elif classification['page_type'] == 'REFERENCE_TABLE':
        # Reference/interpretation tables — no extraction, no LLM needed
        pass

    elif classification['page_type'] == 'MARKETING':
        # Discard
        pass

    elif classification['page_type'] == 'OTHER':
        # Might need LLM to determine what this is
        if classification['hiType_hint'] in ('Prescription', 'OP Consultation', 'Discharge Summary'):
            needs_llm = True
            llm_reason = 'narrative_extraction'

    # If we expected results but got none, flag for LLM
    if classification['has_results'] and not results:
        needs_llm = True
        llm_reason = 'failed_table_parse'

    return {
        'page_id': page['page_id'],
        'page_num': page['page_num'],
        'anchors': anchors,
        'department': department,
        'panels': panels,
        'classification': classification,
        'results': results,
        'needs_llm': needs_llm,
        'llm_reason': llm_reason,
        'raw_text': page['raw_text'],
        'char_count': page['char_count'],
    }

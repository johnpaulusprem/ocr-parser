"""
LLM-first pipeline: Split → LLM per page (parallel) → Merge → Return JSON

New approach:
  1. Segment text into pages
  2. Send EVERY page to Mistral 7B in parallel for structured extraction
  3. Merge all page results into a single unified output
  4. Return one JSON per input document
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from . import config
from .segmenter import segment


# ════════════════════════════════════════════════════════════
#  LLM CALL
# ════════════════════════════════════════════════════════════

def _call_llm(prompt: str, max_output_tokens: int = 2000) -> str:
    import requests
    try:
        response = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": 0.1,
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return '{}'


def _parse_llm_json(raw: str) -> Any:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ════════════════════════════════════════════════════════════
#  PER-PAGE EXTRACTION PROMPT
# ════════════════════════════════════════════════════════════

PAGE_EXTRACTION_PROMPT = """You are a medical document data extractor. Extract ALL structured data from this single page of a medical report.

Return ONLY valid JSON with this exact schema. No explanation, no text outside JSON.

{
  "page_type": "RESULT|NARRATIVE|MARKETING|DISCLAIMER|EMPTY",
  "patient": {
    "name": "",
    "patient_id": "",
    "age": "",
    "gender": "",
    "abha_no": ""
  },
  "report": {
    "facility": "",
    "doctor": "",
    "accession_no": "",
    "barcode_id": "",
    "order_id": "",
    "sample_type": "",
    "collection_date": "",
    "report_date": "",
    "report_status": ""
  },
  "department": "",
  "panel_name": "",
  "test_results": [
    {
      "test_name": "",
      "value": "",
      "unit": "",
      "reference_range": "",
      "method": ""
    }
  ],
  "narrative": {
    "diagnosis": [],
    "symptoms": [],
    "medications": [],
    "advice": [],
    "interpretation": "",
    "karyotype": "",
    "specimen": "",
    "method": ""
  }
}

Rules:
- Extract ONLY what is explicitly written. Do not infer or guess.
- For test_results: extract every test with its numeric/qualitative value, unit, reference range, method.
- Skip marketing content, disclaimers, page footers, lab addresses.
- If the page has no medical data (just ads/disclaimers), set page_type to "MARKETING" or "DISCLAIMER" and leave other fields empty.
- For patient info: extract name, ID, age, gender from any format (tables, plain text, key-value pairs).
- Do NOT extract "ISO 9001", "CIN:", page numbers, or registration numbers as test results.

PAGE TEXT:
{text}"""


# ════════════════════════════════════════════════════════════
#  TRIM & SPLIT TEXT TO FIT TOKEN BUDGET
# ════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Strip HTML tags and collapse multi-spaces, but preserve newlines for structure."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Collapse spaces within lines but keep newlines
    clean = re.sub(r'[^\S\n]+', ' ', clean)
    # Collapse 3+ consecutive blank lines into 2
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    return clean.strip()


def _estimate_tokens(text: str) -> int:
    return int(len(text) / config.CHARS_PER_TOKEN)


def _find_split_point(text: str, target: int) -> int:
    """Find the best semantic split point near `target` char position.
    Priority: blank line > section header > any newline. Never splits mid-line."""
    # Search window: look back up to 20% from target to find a good boundary
    window_start = max(0, target - int(target * 0.2))
    window = text[window_start:target]

    # Priority 1: Split on blank line (paragraph boundary)
    idx = window.rfind('\n\n')
    if idx != -1:
        return window_start + idx + 2  # after the blank line

    # Priority 2: Split on a line that looks like a section header
    #   e.g. "HAEMATOLOGY", "Department:", all-caps lines
    lines_in_window = window.split('\n')
    for i in range(len(lines_in_window) - 1, -1, -1):
        line = lines_in_window[i].strip()
        if (line.isupper() and len(line) > 3) or line.endswith(':'):
            # Split before this header line
            pos = window.rfind('\n' + lines_in_window[i])
            if pos != -1:
                return window_start + pos + 1  # at the start of header line

    # Priority 3: Split on any newline (never mid-line)
    idx = window.rfind('\n')
    if idx != -1:
        return window_start + idx + 1

    # Last resort: split at target (shouldn't happen with normal text)
    return target


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks that fit within max_chars.
    Uses semantic boundaries (blank lines, headers, line breaks) so that
    related data like 'Paracetamol 500mg 1-0-1' is never torn apart.
    Adds overlap context so the LLM has surrounding lines for context."""
    if len(text) <= max_chars:
        return [text]

    OVERLAP_LINES = 3  # repeat last N lines at start of next chunk for context
    chunks = []
    pos = 0
    overlap_text = ""

    while pos < len(text):
        remaining = len(text) - pos
        if remaining <= max_chars:
            chunk = overlap_text + text[pos:]
            # If overlap pushed us over budget, trim overlap
            if len(chunk) > max_chars:
                chunk = text[pos:]
            chunks.append(chunk)
            break

        # Find semantic split point
        split_at = _find_split_point(text, pos + max_chars - len(overlap_text))
        if split_at <= pos:
            split_at = pos + max_chars - len(overlap_text)

        chunk_text = text[pos:split_at]
        chunk = overlap_text + chunk_text
        chunks.append(chunk)

        # Build overlap from last N lines of this chunk
        lines = chunk_text.rstrip('\n').split('\n')
        overlap_lines = lines[-OVERLAP_LINES:] if len(lines) >= OVERLAP_LINES else lines
        overlap_text = '\n'.join(overlap_lines) + '\n'

        pos = split_at

    return chunks


# ════════════════════════════════════════════════════════════
#  PROCESS SINGLE PAGE (with chunk splitting if oversized)
# ════════════════════════════════════════════════════════════

def process_page(page: Dict) -> Dict:
    """Send a single page to LLM and return structured extraction.
    If the page text exceeds the token budget, splits into chunks
    and merges the chunk results."""
    page_num = page['page_num']
    raw_text = page['raw_text']

    # Skip very short pages (likely empty)
    if page['char_count'] < 50:
        return {
            'page_num': page_num,
            'page_type': 'EMPTY',
            'data': None,
            'error': None,
        }

    cleaned = _clean_text(raw_text)

    # Check if text fits in one LLM call
    prompt_template_chars = int(config.PROMPT_TEMPLATE_TOKENS * config.CHARS_PER_TOKEN)
    max_text_chars = config.MAX_INPUT_CHARS

    if len(cleaned) <= max_text_chars:
        # Single call — fits within budget
        prompt = PAGE_EXTRACTION_PROMPT.replace('{text}', cleaned)
        total_tokens = _estimate_tokens(prompt) + config.OUTPUT_BUDGET
        print(f"  Page {page_num}: {len(cleaned)} chars, ~{_estimate_tokens(cleaned)} input tokens, ~{total_tokens} total tokens")

        raw_response = _call_llm(prompt, max_output_tokens=config.OUTPUT_BUDGET)
        parsed = _parse_llm_json(raw_response)

        return {
            'page_num': page_num,
            'page_type': parsed.get('page_type', 'UNKNOWN') if parsed else 'ERROR',
            'data': parsed,
            'error': None if parsed else 'Failed to parse LLM response',
        }
    else:
        # Page too large — split into chunks and process each
        chunks = _split_into_chunks(cleaned, max_text_chars)
        print(f"  Page {page_num}: {len(cleaned)} chars OVERSIZED, splitting into {len(chunks)} chunks")

        chunk_results = []
        for i, chunk in enumerate(chunks):
            prompt = PAGE_EXTRACTION_PROMPT.replace('{text}', chunk)
            raw_response = _call_llm(prompt, max_output_tokens=config.OUTPUT_BUDGET)
            parsed = _parse_llm_json(raw_response)
            if parsed:
                chunk_results.append(parsed)

        if not chunk_results:
            return {
                'page_num': page_num,
                'page_type': 'ERROR',
                'data': None,
                'error': f'All {len(chunks)} chunks failed to parse',
            }

        # Merge chunk results: use first chunk as base, append test_results from others
        merged = chunk_results[0]
        for cr in chunk_results[1:]:
            # Merge test_results
            extra_results = cr.get('test_results', [])
            if extra_results:
                merged.setdefault('test_results', []).extend(extra_results)
            # Merge narrative fields
            for field in ['diagnosis', 'symptoms', 'medications', 'advice']:
                extras = cr.get('narrative', {}).get(field, [])
                if extras:
                    merged.setdefault('narrative', {}).setdefault(field, []).extend(extras)
            # Fill empty patient/report fields from later chunks
            for section in ['patient', 'report']:
                src = cr.get(section, {})
                dst = merged.setdefault(section, {})
                for k, v in src.items():
                    if v and not dst.get(k):
                        dst[k] = v

        return {
            'page_num': page_num,
            'page_type': merged.get('page_type', 'RESULT'),
            'data': merged,
            'error': None,
        }


# ════════════════════════════════════════════════════════════
#  PARALLEL PAGE PROCESSING
# ════════════════════════════════════════════════════════════

def process_all_pages_parallel(pages: List[Dict], max_workers: int = 4) -> List[Dict]:
    """Process all pages through LLM in parallel using ThreadPoolExecutor."""
    results = [None] * len(pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_page, page): i
            for i, page in enumerate(pages)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    'page_num': pages[idx]['page_num'],
                    'page_type': 'ERROR',
                    'data': None,
                    'error': str(e),
                }

    return results


# ════════════════════════════════════════════════════════════
#  MERGE PAGE RESULTS
# ════════════════════════════════════════════════════════════

def merge_page_results(page_results: List[Dict]) -> Dict:
    """Merge all page extractions into a single unified output."""

    # Collect best patient info (first non-empty wins)
    patient = {'name': '', 'patient_id': '', 'age': '', 'gender': '', 'abha_no': ''}
    report = {
        'facility': '', 'doctor': '', 'accession_no': '', 'barcode_id': '',
        'order_id': '', 'sample_type': '', 'collection_date': '',
        'report_date': '', 'report_status': '',
    }

    all_results = []
    all_diagnosis = []
    all_symptoms = []
    all_medications = []
    all_advice = []
    interpretation_parts = []
    departments = []
    panels = []
    karyotype = ''
    specimen = ''
    method = ''

    pages_processed = 0
    pages_with_data = 0
    pages_marketing = 0
    pages_error = 0

    for pr in page_results:
        pages_processed += 1
        data = pr.get('data')
        if not data:
            if pr.get('page_type') == 'ERROR':
                pages_error += 1
            continue

        page_type = data.get('page_type', '')
        if page_type in ('MARKETING', 'DISCLAIMER', 'EMPTY'):
            pages_marketing += 1
            continue

        pages_with_data += 1

        # Merge patient info
        p = data.get('patient', {})
        if p:
            for key in patient:
                val = p.get(key, '')
                if val and not patient[key]:
                    patient[key] = str(val).strip()

        # Merge report info
        r = data.get('report', {})
        if r:
            for key in report:
                val = r.get(key, '')
                if val and not report[key]:
                    report[key] = str(val).strip()

        # Collect department/panel
        dept = data.get('department', '')
        if dept and dept not in departments:
            departments.append(dept)
        panel = data.get('panel_name', '')
        if panel and panel not in panels:
            panels.append(panel)

        # Collect test results
        results = data.get('test_results', [])
        if results:
            for tr in results:
                if not tr.get('test_name'):
                    continue
                # Deduplicate by test_name
                existing_names = {r['test_name'].lower() for r in all_results}
                if tr['test_name'].lower() not in existing_names:
                    tr['page_num'] = pr['page_num']
                    tr['department'] = dept
                    tr['panel'] = panel
                    all_results.append(tr)

        # Collect narrative data
        narrative = data.get('narrative', {})
        if narrative:
            for d in narrative.get('diagnosis', []):
                if d and d not in all_diagnosis:
                    all_diagnosis.append(d)
            for s in narrative.get('symptoms', []):
                if s and s not in all_symptoms:
                    all_symptoms.append(s)
            for m in narrative.get('medications', []):
                if m and m not in all_medications:
                    all_medications.append(m)
            for a in narrative.get('advice', []):
                if a and a not in all_advice:
                    all_advice.append(a)
            interp = narrative.get('interpretation', '')
            if interp:
                interpretation_parts.append(interp)
            if narrative.get('karyotype') and not karyotype:
                karyotype = narrative['karyotype']
            if narrative.get('specimen') and not specimen:
                specimen = narrative['specimen']
            if narrative.get('method') and not method:
                method = narrative['method']

    # Determine hiType
    hi_type, hi_confidence = _determine_hi_type(all_results, all_diagnosis, all_medications, karyotype)

    # Build smartData
    smart_data = _build_smart_data(
        hi_type, all_results, all_diagnosis, all_symptoms,
        all_medications, all_advice, interpretation_parts,
        karyotype, specimen, method, report,
    )

    # Build PII
    age_years = None
    age_str = patient.get('age', '')
    if age_str:
        m = re.search(r'(\d+)', str(age_str))
        if m:
            age_years = int(m.group(1))

    pii = {
        'document': [{
            'PageNum': 1,
            'file_index': 0,
            'DocumentDate': report.get('report_date', ''),
            'Patient': {
                'Name': patient.get('name', ''),
                'Gender': patient.get('gender', ''),
                'patientId': patient.get('patient_id') or None,
                'Age': {
                    'Years': age_years,
                    'Months': None,
                    'Days': None,
                },
            },
            'Report': {
                'Doctor': report.get('doctor', ''),
                'Facility': report.get('facility', ''),
                'GeneratedDate': report.get('report_date', ''),
                'SampleReceivedDate': '',
                'SampleCollectionDate': report.get('collection_date', ''),
            },
        }],
    }

    return {
        'hiType': hi_type,
        'hiTypeConfidence': hi_confidence,
        'extractedText': '',
        'smartData': smart_data,
        'pii': pii,
        'debug': {
            'pages_processed': pages_processed,
            'pages_with_data': pages_with_data,
            'pages_marketing': pages_marketing,
            'pages_error': pages_error,
            'total_results': len(all_results),
            'total_diagnosis': len(all_diagnosis),
            'departments': departments,
            'panels': panels,
        },
    }


def _determine_hi_type(results, diagnosis, medications, karyotype):
    if results:
        return 'Diagnostic Report', 0.95
    if karyotype:
        return 'Diagnostic Report', 0.90
    if medications:
        return 'Prescription', 0.85
    if diagnosis:
        return 'Health Document Record', 0.70
    return 'Health Document Record', 0.30


def _build_smart_data(hi_type, results, diagnosis, symptoms, medications,
                      advice, interpretations, karyotype, specimen, method, report):
    data_items = []
    for r in results:
        # Parse value
        value = None
        value_string = str(r.get('value', ''))
        try:
            value = float(value_string)
            scale_type = 'Qn'
        except (ValueError, TypeError):
            scale_type = 'Ord'

        # Determine abnormality
        is_abnormal = _check_abnormal(value, r.get('reference_range', ''))

        data_items.append({
            'test_name': r.get('test_name', ''),
            'descriptive_test_name': r.get('test_name', ''),
            'test_context': {
                'alternate_names': None,
                'method': r.get('method') or None,
                'panel_name': r.get('panel', ''),
                'department': r.get('department', ''),
                'specimen': specimen or None,
                'scale_type': scale_type,
                'additional_info': None,
            },
            'file_index': 1,
            'page_number': r.get('page_num', 1),
            'result_date': 0,
            'confidence': 1.0 if value is not None else 0.8,
            'isAbnormal': is_abnormal,
            'data': {
                'value': value,
                'value_string': value_string,
                'unit': r.get('unit', ''),
                'unit_processed': r.get('unit', ''),
                'display_range': r.get('reference_range', ''),
            },
        })

    smart = {
        'labVitals': [],
        'data': data_items,
    }

    # Add narrative fields if present
    if diagnosis or symptoms or medications or advice or interpretations:
        smart['diagnosis'] = [{'name': d} for d in diagnosis] if diagnosis else []
        smart['symptoms'] = [{'name': s} for s in symptoms] if symptoms else []
        smart['medications'] = medications if medications else []
        smart['advice'] = [{'text': a} for a in advice] if advice else []
        if interpretations:
            smart['interpretation'] = ' '.join(interpretations)
        if karyotype:
            smart['karyotype'] = karyotype

    return smart


def _check_abnormal(value, ref_range):
    if value is None or not ref_range:
        return None
    ref = str(ref_range).strip()

    # Simple range: "13.0-17.0" or "13.0 - 17.0"
    m = re.match(r'^([\d.]+)\s*[-–]\s*([\d.]+)$', ref)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return value < low or value > high

    # Upper only: "<34" or "<=10"
    m = re.match(r'^[<≤]\s*=?\s*([\d.]+)', ref)
    if m:
        return value >= float(m.group(1))

    # Lower only: ">60" or ">=50"
    m = re.match(r'^[>≥]\s*=?\s*([\d.]+)', ref)
    if m:
        return value < float(m.group(1))

    return None


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def run_llm_pipeline(raw_text: str, verbose: bool = False) -> Dict:
    """
    LLM-first pipeline: Split → LLM per page (parallel) → Merge.
    Returns a single merged output JSON.
    """
    # Step 1: Segment
    pages = segment(raw_text)
    if verbose:
        print(f"Segmented into {len(pages)} pages")

    # Step 2: Process all pages through LLM in parallel
    if verbose:
        print(f"Sending {len(pages)} pages to LLM in parallel...")
    page_results = process_all_pages_parallel(pages, max_workers=4)

    if verbose:
        for pr in page_results:
            status = pr['page_type']
            err = pr.get('error', '')
            print(f"  Page {pr['page_num']}: {status}{' ERROR: ' + err if err else ''}")

    # Step 3: Merge all results
    merged = merge_page_results(page_results)

    if verbose:
        d = merged['debug']
        print(f"Merged: {d['pages_with_data']} data pages, "
              f"{d['pages_marketing']} marketing, "
              f"{d['total_results']} results, "
              f"{d['total_diagnosis']} diagnoses")

    return merged

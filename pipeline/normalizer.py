"""
Step 5: Normalizer + JSON Emitter

Converts doc_instances into target ABDM JSON schemas.
Handles:
  - Date normalization → unix timestamps
  - Unit normalization
  - Abnormality computation (code, not LLM)
  - Merging code-extracted + LLM-extracted data
  - Mapping to per-hiType output schemas
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any


# ════════════════════════════════════════════════════════════
#  DATE PARSING
# ════════════════════════════════════════════════════════════

DATE_FORMATS = [
    '%d/%b/%Y %I:%M%p',     # 12/May/2025 02:42PM
    '%d/%b/%Y %I:%M %p',    # 12/May/2025 02:42 PM
    '%d/%b/%Y',              # 12/May/2025
    '%d-%b-%Y',              # 12-May-2025
    '%d/%m/%Y',              # 12/05/2025
    '%d-%m-%Y',              # 12-05-2025
    '%Y-%m-%d',              # 2025-05-12
    '%d %b %Y',              # 12 May 2025
    '%d %B %Y',              # 12 May 2025
    '%b %d, %Y',             # May 12, 2025
    '%d/%m/%y',              # 12/05/25
]


def parse_date_to_unix(date_str: str) -> Optional[int]:
    """Parse a date string to unix timestamp. Returns None if unparseable."""
    if not date_str:
        return None

    date_str = date_str.strip()

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue

    # Try partial parsing — extract date portion
    m = re.search(r'(\d{1,2})[/.-](\w{3,9})[/.-](\d{2,4})', date_str)
    if m:
        partial = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        for fmt in ['%d/%b/%Y', '%d/%B/%Y', '%d/%m/%Y']:
            try:
                dt = datetime.strptime(partial, fmt)
                return int(dt.timestamp())
            except ValueError:
                continue

    return None


# ════════════════════════════════════════════════════════════
#  UNIT NORMALIZATION
# ════════════════════════════════════════════════════════════

UNIT_MAP = {
    '10^6/cu.mm': '10^6/µL',
    '10^3/ul': '10^3/µL',
    '10^3/µl': '10^3/µL',
    'ug/dl': 'µg/dL',
    'uiu/ml': 'µIU/mL',
    'mg/dl': 'mg/dL',
    'g/dl': 'g/dL',
    'pg/ml': 'pg/mL',
    'ng/ml': 'ng/mL',
    'meq/l': 'mEq/L',
    'u/l': 'U/L',
    'iu/l': 'IU/L',
    'mm/hr': 'mm/hr',
    'fl': 'fL',
    'pg': 'pg',
    '%': '%',
}


def normalize_unit(unit: str) -> str:
    """Normalize unit to standard form."""
    if not unit:
        return ''
    return UNIT_MAP.get(unit.lower().strip(), unit.strip())


# ════════════════════════════════════════════════════════════
#  SCALE TYPE DETERMINATION
# ════════════════════════════════════════════════════════════

QUALITATIVE_VALUES = {
    'positive', 'negative', 'reactive', 'non-reactive', 'nonreactive',
    'detected', 'not detected', 'present', 'absent',
    'normal', 'abnormal', 'trace', 'nil',
}

NOMINAL_VALUES = {
    'yellow', 'straw', 'amber', 'dark yellow', 'red', 'brown',
    'clear', 'turbid', 'hazy', 'slightly turbid', 'cloudy',
}


def determine_scale_type(value_string: str, test_name: str = '') -> str:
    """
    Determine LOINC scale type:
      Qn  = Quantitative (numeric)
      Ord = Ordinal (positive/negative/reactive)
      Nom = Nominal (color, appearance)
      Nar = Narrative (free text findings)
    """
    if not value_string:
        return 'Nar'

    vs = value_string.strip().lower()

    if vs in QUALITATIVE_VALUES:
        return 'Ord'
    if vs in NOMINAL_VALUES:
        return 'Nom'
    if re.match(r'^[\d.]+$', vs):
        return 'Qn'

    # MRI/imaging findings
    if any(kw in test_name.lower() for kw in ['mri', 'ct', 'x-ray', 'ultrasound', 'impression', 'findings']):
        return 'Nar'

    return 'Nar'


# ════════════════════════════════════════════════════════════
#  BUILD PII BLOCK
# ════════════════════════════════════════════════════════════

def build_pii(doc: Dict) -> Dict:
    """Build the pii.document[] block from merged anchors."""
    anchors = doc['merged_anchors']
    pages = doc['pages']

    # Find page numbers
    page_nums = sorted(set(p['page_num'] for p in pages))

    report_date = parse_date_to_unix(anchors.get('reportDate', ''))
    collection_date = parse_date_to_unix(anchors.get('collectionDate', ''))
    received_date = parse_date_to_unix(anchors.get('receivedDate', ''))

    # Parse age
    age_years = None
    age_months = None
    age_days = None
    age_str = anchors.get('age', '')
    if age_str:
        try:
            age_years = int(age_str)
        except ValueError:
            pass

    return {
        'document': [{
            'PageNum': page_nums[0] if page_nums else 1,
            'file_index': 0,
            'DocumentDate': report_date,
            'Patient': {
                'Name': anchors.get('patientName', ''),
                'Gender': anchors.get('gender', ''),
                'patientId': anchors.get('patientId'),
                'Age': {
                    'Years': age_years,
                    'Months': age_months,
                    'Days': age_days,
                },
            },
            'Report': {
                'Doctor': anchors.get('doctor', ''),
                'Facility': anchors.get('facility', ''),
                'GeneratedDate': anchors.get('reportDate', ''),
                'SampleReceivedDate': anchors.get('receivedDate', ''),
                'SampleCollectionDate': anchors.get('collectionDate', ''),
            },
        }],
    }


# ════════════════════════════════════════════════════════════
#  BUILD SMART DATA PER HI TYPE
# ════════════════════════════════════════════════════════════

def _build_diagnostic_report_data(doc: Dict) -> Dict:
    """Build smartData for Diagnostic Report."""
    all_results = []
    for page in doc['pages']:
        for r in page.get('results', []):
            result_date = parse_date_to_unix(
                doc['merged_anchors'].get('reportDate', '')
            )

            all_results.append({
                'test_name': r['test_name'],
                'descriptive_test_name': r['test_name'],
                'test_context': {
                    'alternate_names': None,
                    'method': r.get('method'),
                    'panel_name': r.get('panel', ''),
                    'department': r.get('department', ''),
                    'specimen': doc['merged_anchors'].get('sampleType'),
                    'scale_type': r.get('scale_type', determine_scale_type(
                        r.get('value_string', ''), r['test_name']
                    )),
                    'additional_info': None,
                },
                'file_index': 1,
                'page_number': page['page_num'],
                'result_date': result_date or 0,
                'confidence': 1.0 if r.get('value') is not None else 0.8,
                'isAbnormal': r.get('is_abnormal'),
                'data': {
                    'value': r.get('value'),
                    'value_string': r.get('value_string', ''),
                    'unit': normalize_unit(r.get('unit', '')),
                    'unit_processed': normalize_unit(r.get('unit', '')),
                    'display_range': r.get('display_range', ''),
                },
            })

    return {
        'labVitals': [],
        'data': all_results,
    }


def _build_narrative_smartdata(doc: Dict) -> Dict:
    """
    Build smartData for narrative HI types (Prescription, OP Consultation, Discharge Summary).
    Merges LLM-extracted data from narrative pages.
    """
    symptoms = []
    diagnosis = []
    medications = []
    advice = []
    followup = {'date': '', 'notes': None}
    lab_vitals = []

    for page in doc['pages']:
        llm_data = page.get('llm_extracted')
        if not llm_data or llm_data.get('_stub'):
            continue

        for s in llm_data.get('symptoms', []):
            if s.get('name') and s not in symptoms:
                symptoms.append(s)

        for d in llm_data.get('diagnosis', []):
            if d.get('name') and d not in diagnosis:
                diagnosis.append(d)

        for m in llm_data.get('medications', []):
            if m.get('name') and m not in medications:
                medications.append(m)

        for a in llm_data.get('advice', []):
            if a.get('text') and a not in advice:
                advice.append(a)

        if llm_data.get('followup', {}).get('date'):
            followup = llm_data['followup']

        for v in llm_data.get('vitals', []):
            if v.get('name'):
                lab_vitals.append({
                    'name': v['name'],
                    'value': v.get('value', ''),
                    'unit': v.get('unit', ''),
                    'date': parse_date_to_unix(doc['merged_anchors'].get('reportDate', '')) or 0,
                })

    result = {
        'symptoms': symptoms or [{'name': ''}],
        'diagnosis': diagnosis or [{'name': ''}],
        'medications': medications or [{
            'name': '', 'generic_name': None, 'timing': None,
            'frequency': {'custom': '', 'type': None},
            'duration': {'custom': ''},
            'dose': {'custom': ''},
        }],
        'advice': advice or [{'text': ''}],
        'followup': followup,
        'labVitals': lab_vitals or [{'name': '', 'value': '', 'unit': '', 'date': 0}],
        'data': [],
    }

    # Add medical history for OP Consultation / Discharge Summary
    if doc['hiType'] in ('OP Consultation', 'Discharge Summary'):
        result['medicalHistory'] = {
            'examinations': [{'name': ''}],
            'patientHistory': {
                'patientMedicalConditions': [{'name': ''}],
            } if doc['hiType'] == 'Discharge Summary' else None,
        }

    return result


# ════════════════════════════════════════════════════════════
#  MAIN NORMALIZER
# ════════════════════════════════════════════════════════════

def normalize_doc_instance(doc: Dict) -> Dict:
    """
    Convert a doc_instance into the target JSON output schema.

    Returns the final JSON object ready for output.
    """
    hiType = doc['hiType']
    confidence = doc['hiType_confidence']

    # Build PII
    pii = build_pii(doc)

    # Build smartData based on hiType
    if hiType == 'Diagnostic Report':
        smart_data = _build_diagnostic_report_data(doc)
    elif hiType in ('Prescription', 'OP Consultation', 'Discharge Summary'):
        smart_data = _build_narrative_smartdata(doc)
    elif hiType == 'Immunization Record':
        smart_data = _build_diagnostic_report_data(doc)  # Same structure, different panel
    elif hiType == 'Wellness Record':
        smart_data = _build_diagnostic_report_data(doc)
    elif hiType == 'UNKNOWN':
        return {
            'hiType': 'UNKNOWN',
            'hiTypeConfidence': 0.1,
            'extractedText': '',
            'smartData': None,
            'pii': None,
        }
    else:
        # Health Document Record (fallback)
        smart_data = _build_narrative_smartdata(doc)

    # Compute extracted text summary
    extracted_text = ''  # Could populate with raw text if needed

    return {
        'hiType': hiType,
        'hiTypeConfidence': round(confidence, 2),
        'extractedText': extracted_text,
        'smartData': smart_data,
        'pii': pii,
    }


def normalize_all(doc_instances: List[Dict]) -> List[Dict]:
    """Normalize all doc_instances into output JSONs."""
    return [normalize_doc_instance(doc) for doc in doc_instances]

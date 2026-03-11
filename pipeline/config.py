"""
Pipeline configuration.

All LLM budget and provider settings can be overridden via environment variables:
  LLM_MAX_CONTEXT_TOKENS  - Total token limit (input + output). Default: 16000
  LLM_OUTPUT_BUDGET       - Tokens reserved for model response. Default: 2000
  LLM_CHARS_PER_TOKEN     - Chars-per-token ratio. Default: 3.5
  LLM_MODEL               - Model name. Default: mistral:7b
  LLM_BASE_URL            - API endpoint. Default: http://localhost:11434/v1
  LLM_API_KEY             - API key. Default: ollama
"""

import os

# ── LLM Budget ──────────────────────────────────────────────
MAX_CONTEXT_TOKENS = int(os.environ.get("LLM_MAX_CONTEXT_TOKENS", 16000))
PROMPT_TEMPLATE_TOKENS = 400   # the extraction prompt template (without page text)
OUTPUT_BUDGET = int(os.environ.get("LLM_OUTPUT_BUDGET", 2000))
CHARS_PER_TOKEN = float(os.environ.get("LLM_CHARS_PER_TOKEN", 3.5))

INPUT_BUDGET = MAX_CONTEXT_TOKENS - PROMPT_TEMPLATE_TOKENS - OUTPUT_BUDGET
MAX_INPUT_CHARS = int(INPUT_BUDGET * CHARS_PER_TOKEN)

# Legacy (used by llm_handler.py)
MAX_TOKENS_PER_CALL = 6000

# ── LLM Provider ────────────────────────────────────────────
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral:7b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# ── Page Segmentation ───────────────────────────────────────
# Patterns that indicate a page boundary (compiled at import time)
PAGE_BOUNDARY_PATTERNS = [
    r'---\s*Page\s+\d+\s+of\s+\d+\s*---',       # --- Page 1 of 18 ---
    r'Page\s+\d+\s+of\s+\d+',                     # Page 1 of 16
]

# ── Marketing / Disclaimer Keywords ─────────────────────────
MARKETING_KEYWORDS = [
    'ENSURING ACCURACY', 'SHOP NOW', 'LIMITED OFFER', 'EXPLORE NOW',
    'OMEGA 3', 'Running Low', 'DePURA', 'SCHEDULE NOW', 'CONNECT NOW',
    'satisfied customers', 'lab tests booked', 'State of the Art',
    'Watch how we take care', 'Claim FREE doctor consultation',
    'Making Strides in Compassionate', 'Accurate Testing, Assured Quality',
]

DISCLAIMER_KEYWORDS = [
    'NABL certificate', 'Scan for digital copy', 'CLSI*',
    'T&C Apply', 'CIN:', 'ISO 9001',
]

# ── Department Detection ─────────────────────────────────────
DEPARTMENTS = [
    'HAEMATOLOGY', 'BIOCHEMISTRY', 'IMMUNOLOGY', 'CLINICAL PATHOLOGY',
    'SEROLOGY', 'MICROBIOLOGY', 'HISTOPATHOLOGY', 'CYTOLOGY',
    'ENDOCRINOLOGY', 'MOLECULAR BIOLOGY',
]

# ── HI Type Classification Signals ──────────────────────────
HITYPE_SIGNALS = {
    'Diagnostic Report': [
        'Test Name', 'Result', 'Bio. Ref. Interval', 'Reference Range',
        'Method', 'HAEMATOLOGY', 'BIOCHEMISTRY', 'IMMUNOLOGY',
        'CLINICAL PATHOLOGY', 'SEROLOGY', 'MRI', 'CT Scan', 'X-Ray',
        'Ultrasound', 'FINDINGS', 'IMPRESSION', 'Specimen',
    ],
    'Prescription': [
        'Rx', 'Tab', 'Cap', 'Syp', 'Inj', 'OD', 'BD', 'TDS',
        'SOS', 'Before food', 'After food', 'Empty stomach',
        'mg', 'ml', 'dose', 'frequency',
    ],
    'OP Consultation': [
        'Chief Complaint', 'History', 'Examination', 'Diagnosis',
        'Investigations', 'Advised', 'Follow up', 'OPD',
        'Consultation', 'C/O', 'O/E', 'Presenting complaints',
    ],
    'Discharge Summary': [
        'Discharge', 'Admission', 'Date of Admission',
        'Date of Discharge', 'Hospital Course', 'Condition at Discharge',
        'Discharge Medication', 'Final Diagnosis',
        'Indoor', 'IPD', 'Sent home', 'Condition on Discharge',
        'Treatment Given', 'IV antibiotics', 'admitted',
    ],
    'Immunization Record': [
        'Vaccine', 'Immunization', 'Dose', 'Batch', 'Site of injection',
        'Next due', 'Vaccination',
    ],
}

# ── Anchor Extraction Patterns ──────────────────────────────
# Keys: field name → list of (label_pattern, value_pattern) tuples
ANCHOR_PATTERNS = {
    'patientName': [
        (r'(?:Name|Patient\s*Name)\s*[:]\s*', r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Master\.)?\s*([A-Za-z\s.]+)'),
    ],
    'patientId': [
        (r'(?:Patient\s*ID|UHID|MRN|CR\s*No|Reg\.?\s*No|IPD\s*No|OPD\s*No|MaxID|HN)\s*[:]\s*', r'(\S+)'),
    ],
    'barcodeId': [
        (r'(?:Barcode\s*(?:ID)?|Accession\s*(?:No|ID)?)\s*[:]\s*', r'(\S+)'),
    ],
    'orderId': [
        (r'(?:Order\s*(?:ID|No)?|PO\s*No|Bill\s*No|Lab\s*No|Sample\s*No)\s*[:]\s*', r'(\S+)'),
    ],
    'sampleType': [
        (r'(?:Sample\s*Type|Specimen|Sample)\s*[:]\s*', r'([A-Za-z\s]+)'),
    ],
    'collectionDate': [
        (r'(?:Collection\s*Date|Sample\s*(?:Collected|Collection)\s*(?:Date)?|Received\s*(?:Date|On))\s*[:]\s*',
         r'(.+?)(?:\s*$)'),
    ],
    'reportDate': [
        (r'(?:Report\s*Date|Reported?\s*(?:Date|On)|Generated\s*(?:Date|On))\s*[:]\s*',
         r'(.+?)(?:\s*$)'),
    ],
    'facility': [
        (r'(?:Lab|Laboratory|Hospital|Centre|Center|Clinic|Institute)\s*[:]\s*', r'(.+?)(?:\s*$)'),
    ],
    'doctor': [
        (r'(?:Referred\s*By|Consultant|Doctor|Dr\.)\s*[:]\s*', r'(.+?)(?:\s*$)'),
    ],
}

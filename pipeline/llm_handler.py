"""
Step 4: LLM Handler — Minimal Mistral 3 14B Calls for Gaps

Only called when code extraction can't handle the content:
  Case A: Narrative pages (prescriptions, OP notes, handwritten)
  Case B: Ambiguous HI type classification
  Case C: Non-standard result formats that code parser missed

Budget: 6000 tokens per call (prompt + input + output).
"""

import json
import re
from typing import List, Dict, Optional, Any
from . import config


# ════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES (kept minimal to save tokens)
# ════════════════════════════════════════════════════════════

NARRATIVE_EXTRACTION_PROMPT = """Extract structured medical data from this clinical text.
Return ONLY valid JSON. No explanation.

Schema:
{"symptoms":[{"name":""}],"diagnosis":[{"name":""}],"medications":[{"name":"","generic_name":null,"timing":null,"frequency":{"custom":""},"duration":{"custom":""},"dose":{"custom":""}}],"advice":[{"text":""}],"followup":{"date":"","notes":null},"vitals":[{"name":"","value":"","unit":""}]}

Rules:
- Extract ONLY what is explicitly stated. Do not infer.
- If a field has no data, use empty array [] or null.
- For medications: include name exactly as written. Parse dose/frequency if present.
- For advice: include investigations ordered, lifestyle advice, follow-up instructions.

Text:
{text}"""

RESULT_EXTRACTION_PROMPT = """Extract lab test results from this text.
Return ONLY valid JSON array. No explanation.

Schema per result:
{"test_name":"","value":"","unit":"","display_range":"","method":null}

Rules:
- Extract test name, numeric or qualitative value, unit, reference range.
- Include ALL results. Do not skip any.
- If range is not present, use empty string.

Text:
{text}"""

CLASSIFY_PROMPT = """Classify this medical document page.
Return ONLY valid JSON. No explanation.

Options for hiType: Diagnostic Report, Prescription, OP Consultation, Discharge Summary, Immunization Record, Health Document Record, UNKNOWN

Schema:
{"hiType":"","confidence":0.0,"summary":"one line description"}

Text (first 500 chars):
{text}"""


# ════════════════════════════════════════════════════════════
#  TOKEN BUDGET CHECK
# ════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return int(len(text) / config.CHARS_PER_TOKEN)


def _trim_to_budget(text: str, max_tokens: int) -> str:
    """Trim text to fit within token budget."""
    max_chars = int(max_tokens * config.CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


# ════════════════════════════════════════════════════════════
#  LLM CALL INTERFACE (pluggable)
# ════════════════════════════════════════════════════════════

def _call_llm(prompt: str, max_output_tokens: int = 1200) -> str:
    """
    Call the LLM via Ollama's OpenAI-compatible API.
    """
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
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return '{}'


def _parse_llm_json(raw: str) -> Any:
    """Safely parse LLM output as JSON, handling common formatting issues."""
    # Strip markdown code fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON within the response
        m = re.search(r'[\[{].*[\]}]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════

def extract_narrative(page_text: str) -> Optional[Dict]:
    """
    Extract structured data from a narrative clinical page.
    Used for: prescriptions, OP consultations, discharge summaries, handwritten notes.

    Returns: {symptoms, diagnosis, medications, advice, followup, vitals} or None
    """
    # Strip HTML for cleaner input to LLM
    clean = re.sub(r'<[^>]+>', ' ', page_text)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Budget check
    prompt_overhead = _estimate_tokens(NARRATIVE_EXTRACTION_PROMPT.replace('{text}', ''))
    available = config.INPUT_BUDGET - prompt_overhead
    clean = _trim_to_budget(clean, available)

    prompt = NARRATIVE_EXTRACTION_PROMPT.replace('{text}', clean)

    total_tokens = _estimate_tokens(prompt) + config.OUTPUT_BUDGET
    if total_tokens > config.MAX_TOKENS_PER_CALL:
        # Further trim
        overage = total_tokens - config.MAX_TOKENS_PER_CALL
        clean = _trim_to_budget(clean, available - overage)
        prompt = NARRATIVE_EXTRACTION_PROMPT.replace('{text}', clean)

    raw = _call_llm(prompt)
    return _parse_llm_json(raw)


def extract_results_llm(page_text: str) -> Optional[List[Dict]]:
    """
    Extract test results from non-standard format text using LLM.
    Used when code parser fails to extract results from a page that has them.

    Returns: [{test_name, value, unit, display_range, method}] or None
    """
    clean = re.sub(r'<[^>]+>', ' ', page_text)
    clean = re.sub(r'\s+', ' ', clean).strip()

    prompt_overhead = _estimate_tokens(RESULT_EXTRACTION_PROMPT.replace('{text}', ''))
    available = config.INPUT_BUDGET - prompt_overhead
    clean = _trim_to_budget(clean, available)

    prompt = RESULT_EXTRACTION_PROMPT.replace('{text}', clean)

    raw = _call_llm(prompt)
    parsed = _parse_llm_json(raw)

    if isinstance(parsed, list):
        return parsed
    return None


def classify_page_llm(page_text: str) -> Optional[Dict]:
    """
    Classify an ambiguous page using LLM.

    Returns: {hiType, confidence, summary} or None
    """
    # Only send first 500 chars to save tokens
    clean = re.sub(r'<[^>]+>', ' ', page_text)
    clean = re.sub(r'\s+', ' ', clean).strip()[:500]

    prompt = CLASSIFY_PROMPT.replace('{text}', clean)

    raw = _call_llm(prompt)
    return _parse_llm_json(raw)


def process_llm_gaps(doc_instances: List[Dict]) -> List[Dict]:
    """
    Process all doc_instances that need LLM calls.
    Modifies doc_instances in-place, adding LLM-extracted data.

    Returns the same list with added fields:
      - page['llm_extracted']: data from LLM call
      - doc_instance['llm_calls_made']: count of LLM calls
    """
    for doc in doc_instances:
        llm_calls = 0

        for page in doc['pages']:
            if not page['needs_llm']:
                continue

            if page['llm_reason'] == 'narrative_extraction':
                extracted = extract_narrative(page['raw_text'])
                page['llm_extracted'] = extracted
                llm_calls += 1

            elif page['llm_reason'] == 'failed_table_parse':
                results = extract_results_llm(page['raw_text'])
                if results:
                    # Convert LLM results to same format as code-extracted
                    for r in results:
                        r.setdefault('panel', '')
                        r.setdefault('department', page.get('department', ''))
                        r.setdefault('is_abnormal', None)
                        r.setdefault('parsed_range', {'display_range': r.get('display_range', ''),
                                                       'range_type': 'unknown'})
                        r.setdefault('scale_type', 'Qn')
                        r.setdefault('value_string', str(r.get('value', '')))
                        try:
                            r['value'] = float(r['value']) if r.get('value') else None
                        except (ValueError, TypeError):
                            r['value'] = None
                            r['scale_type'] = 'Ord'

                    page['results'].extend(results)
                    page['llm_extracted'] = {'results': results}
                llm_calls += 1

        doc['llm_calls_made'] = llm_calls

    return doc_instances

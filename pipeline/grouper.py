"""
Step 3: Deterministic Grouper

Groups extracted pages into doc_instances (logical documents).
Handles:
  - Same patient, same order → one group (Tata 1mg 18-page case)
  - Same patient, different facilities → separate groups (Parveen case)
  - Multiple HI types in one PDF → separate doc_instances per type
  - Jumbled page order → order-agnostic grouping by anchors
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re


def _normalize_name(name: str) -> str:
    """Normalize patient name for fuzzy matching."""
    if not name:
        return ''
    # Remove titles
    name = re.sub(r'\b(Mr|Mrs|Ms|Dr|Master|Smt|Shri)\.?\s*', '', name, flags=re.IGNORECASE)
    # Uppercase, strip extra whitespace
    return ' '.join(name.upper().split())


def _build_group_key(anchors: Dict, classification: Dict) -> str:
    """
    Build a grouping key from anchors, using strongest available identifiers.

    Priority:
      1. patientId | orderId               (strongest — same lab order)
      2. patientId | barcodeId              (same specimen)
      3. patientId | facility | reportDate  (same visit)
      4. patientName(normalized) | facility (weakest — fuzzy)
    """
    pid = anchors.get('patientId', '')
    oid = anchors.get('orderId', '')
    bid = anchors.get('barcodeId', '')
    facility = anchors.get('facility', '')
    report_date = anchors.get('reportDate', '')
    collection_date = anchors.get('collectionDate', '')
    name = _normalize_name(anchors.get('patientName', ''))

    # Level 1: patientId + orderId
    if pid and oid:
        return f"{pid}|{oid}"

    # Level 2: patientId + barcodeId
    if pid and bid:
        return f"{pid}|{bid}"

    # Level 3: patientId + facility + date
    date = report_date or collection_date
    if pid and facility:
        if date:
            # Normalize date to just the day portion
            day_m = re.search(r'(\d{1,2})[/.-](\w{3,9})[/.-](\d{2,4})', date)
            if day_m:
                return f"{pid}|{facility}|{day_m.group(0)}"
        return f"{pid}|{facility}"

    # Level 4: name + facility + date
    if name and facility:
        if date:
            day_m = re.search(r'(\d{1,2})[/.-](\w{3,9})[/.-](\d{2,4})', date)
            if day_m:
                return f"{name}|{facility}|{day_m.group(0)}"
        return f"{name}|{facility}"

    # Level 5: name only (very weak)
    if name:
        return f"{name}|UNKNOWN_FACILITY"

    # Fallback: unique per page
    return f"UNGROUPED|{id(anchors)}"


def _determine_hitype(pages_in_group: List[Dict]) -> Tuple[str, float]:
    """
    Determine the HI type for a group of pages based on their content.

    Rules:
      - If ANY page has lab results → Diagnostic Report
      - If pages have medication/diagnosis narrative → check for admission keywords
        - Has admission → Discharge Summary
        - Has OPD/consultation → OP Consultation
        - Otherwise → Prescription
      - Fallback → Health Document Record
    """
    has_results = any(p['classification']['has_results'] for p in pages_in_group)
    total_results = sum(len(p['results']) for p in pages_in_group)

    # Collect all hiType hints
    hint_counts = defaultdict(float)
    for p in pages_in_group:
        hint = p['classification']['hiType_hint']
        conf = p['classification']['hiType_confidence']
        if hint:
            hint_counts[hint] += conf

    # Collect all page types
    page_types = [p['classification']['page_type'] for p in pages_in_group]
    all_text = ' '.join(p['raw_text'] for p in pages_in_group).lower()

    # Decision logic
    if has_results and total_results > 0:
        return 'Diagnostic Report', 0.95

    if 'Discharge Summary' in hint_counts:
        # Verify with discharge-specific keywords
        if any(kw in all_text for kw in ['discharge', 'admission', 'hospital course']):
            return 'Discharge Summary', 0.90

    if 'NARRATIVE' in page_types:
        # Distinguish OP Consultation vs Prescription
        has_opd = any(kw in all_text for kw in ['opd', 'consultation', 'o/e', 'examination', 'chief complaint'])
        has_rx = any(kw in all_text for kw in ['rx', 'tab', 'cap', 'syp', 'dose', 'frequency', 'bd', 'od', 'tds'])

        if has_opd and has_rx:
            return 'OP Consultation', 0.85  # OP notes with prescriptions
        elif has_opd:
            return 'OP Consultation', 0.85
        elif has_rx:
            return 'Prescription', 0.85

    # MRI/imaging reports
    if any(kw in all_text for kw in ['mri', 'ct scan', 'x-ray', 'ultrasound', 'findings', 'impression']):
        return 'Diagnostic Report', 0.80

    # If all pages are marketing/admin
    if all(pt in ('MARKETING', 'ADMIN') for pt in page_types):
        return 'UNKNOWN', 0.1

    # Commentary-only pages
    if all(pt in ('COMMENTARY', 'REFERENCE_TABLE', 'MARKETING', 'ADMIN') for pt in page_types):
        return 'UNKNOWN', 0.3

    # Fallback — only use hint if it's meaningful
    if hint_counts:
        best_hint = max(hint_counts, key=hint_counts.get)
        best_score = hint_counts[best_hint]
        # Only trust the hint if there's enough signal
        if best_score >= 0.15:
            return best_hint, 0.60
        # Low signal — needs LLM classification
        return 'Health Document Record', 0.30

    return 'Health Document Record', 0.30


def group_pages(extracted_pages: List[Dict]) -> List[Dict]:
    """
    Main entry point. Groups extracted pages into doc_instances.

    Returns list of:
    {
        'doc_id': 'doc_001',
        'group_key': str,
        'hiType': str,
        'hiType_confidence': float,
        'pages': [extracted_page_dicts],
        'result_pages': [pages with actual results],
        'commentary_pages': [commentary-only pages],
        'marketing_pages': [marketing pages — quarantined],
        'narrative_pages': [pages needing LLM extraction],
        'merged_anchors': {best anchors from all pages in group},
        'total_results': int,
        'needs_llm': bool,
    }
    """
    # Build groups by key
    groups = defaultdict(list)
    for page in extracted_pages:
        key = _build_group_key(page['anchors'], page['classification'])
        groups[key].append(page)

    # Convert to doc_instances
    doc_instances = []
    for idx, (key, pages) in enumerate(groups.items()):
        # Merge anchors: prefer non-empty values from any page
        merged_anchors = {}
        for page in pages:
            for k, v in page['anchors'].items():
                if v and (k not in merged_anchors or not merged_anchors[k]):
                    merged_anchors[k] = v

        # Classify pages within group
        result_pages = [p for p in pages if p['results']]
        commentary_pages = [p for p in pages
                           if p['classification']['page_type'] in ('COMMENTARY', 'REFERENCE_TABLE')
                           and not p['results']]
        marketing_pages = [p for p in pages
                          if p['classification']['page_type'] in ('MARKETING', 'ADMIN')]
        narrative_pages = [p for p in pages if p['needs_llm']]

        # Determine HI type
        hiType, confidence = _determine_hitype(pages)

        # Check if any pages need LLM
        needs_llm = any(p['needs_llm'] for p in pages)

        total_results = sum(len(p['results']) for p in pages)

        doc_instances.append({
            'doc_id': f"doc_{idx + 1:03d}",
            'group_key': key,
            'hiType': hiType,
            'hiType_confidence': confidence,
            'pages': pages,
            'result_pages': result_pages,
            'commentary_pages': commentary_pages,
            'marketing_pages': marketing_pages,
            'narrative_pages': narrative_pages,
            'merged_anchors': merged_anchors,
            'total_results': total_results,
            'needs_llm': needs_llm,
        })

    # Sort by doc_id
    doc_instances.sort(key=lambda d: d['doc_id'])

    return doc_instances

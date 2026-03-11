"""
Step 1: Deterministic Page Segmenter

Splits a raw OCR text stream into independent page chunks.
Handles:
  - Literal \\n (from OCR) vs real newlines
  - Page boundary markers (--- Page N of M ---)
  - Repeated header patterns (when no explicit page markers)
  - Jumbled page order (each page is self-contained)
"""

import re
from typing import List, Dict, Optional


def normalize_newlines(raw_text: str) -> List[str]:
    """Convert literal \\n to actual newlines, return list of lines."""
    # Check if the text uses literal \\n (common in OCR pipelines)
    if '\\n' in raw_text and '\n' not in raw_text:
        return raw_text.split('\\n')
    elif '\\n' in raw_text:
        # Mixed — normalize literal \\n first, then split on real newlines
        normalized = raw_text.replace('\\n', '\n')
        return normalized.split('\n')
    else:
        return raw_text.split('\n')


def find_page_boundaries(lines: List[str]) -> List[Dict]:
    """
    Detect page boundaries using multiple strategies.
    Returns list of {page_num, start_line, end_line, source_marker}.
    """
    boundaries = []

    # Strategy 1: Explicit page markers (--- Page N of M ---)
    for i, line in enumerate(lines):
        m = re.search(r'---\s*Page\s+(\d+)\s+of\s+(\d+)\s*---', line)
        if m:
            boundaries.append({
                'page_num': int(m.group(1)),
                'total_pages': int(m.group(2)),
                'start_line': i,
                'marker': 'explicit',
            })

    # Strategy 2: Inline "Page N of M" (without dashes)
    if not boundaries:
        for i, line in enumerate(lines):
            m = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', line)
            if m:
                boundaries.append({
                    'page_num': int(m.group(1)),
                    'total_pages': int(m.group(2)),
                    'start_line': i,
                    'marker': 'inline',
                })

    # Strategy 3: Repeated header blocks (detect by repeated Patient ID or facility name)
    if not boundaries:
        # Find lines that repeat — likely header starts
        header_candidates = []
        seen_lines = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 20 and stripped in seen_lines:
                # This line appeared before — likely a repeated header
                if seen_lines[stripped] not in [h['start_line'] for h in header_candidates]:
                    header_candidates.append({
                        'start_line': seen_lines[stripped],
                        'marker': 'repeated_header',
                    })
                header_candidates.append({
                    'start_line': i,
                    'marker': 'repeated_header',
                })
            seen_lines[stripped] = i

        # Use repeated headers as boundaries
        if header_candidates:
            # Deduplicate and sort
            seen_starts = set()
            for h in header_candidates:
                if h['start_line'] not in seen_starts:
                    boundaries.append({
                        'page_num': len(seen_starts) + 1,
                        'total_pages': None,
                        'start_line': h['start_line'],
                        'marker': 'repeated_header',
                    })
                    seen_starts.add(h['start_line'])

    # Strategy 4: No boundaries found — treat entire text as one page
    if not boundaries:
        boundaries.append({
            'page_num': 1,
            'total_pages': 1,
            'start_line': 0,
            'marker': 'single_page',
        })

    # Sort by start_line (important for jumbled markers)
    boundaries.sort(key=lambda b: b['start_line'])

    # Assign end_lines
    for i in range(len(boundaries)):
        if i + 1 < len(boundaries):
            boundaries[i]['end_line'] = boundaries[i + 1]['start_line']
        else:
            boundaries[i]['end_line'] = len(lines)

    return boundaries


def segment(raw_text: str) -> List[Dict]:
    """
    Main entry point. Splits OCR stream into independent page chunks.

    Returns list of:
    {
        'page_id': 'page_001',
        'page_num': 1,
        'total_pages': 18,
        'lines': [...],
        'raw_text': '...',
        'char_count': 5042,
        'marker': 'explicit',
    }
    """
    lines = normalize_newlines(raw_text)
    boundaries = find_page_boundaries(lines)

    pages = []
    for b in boundaries:
        page_lines = lines[b['start_line']:b['end_line']]
        raw = '\n'.join(page_lines)
        pages.append({
            'page_id': f"page_{b['page_num']:03d}",
            'page_num': b['page_num'],
            'total_pages': b.get('total_pages'),
            'lines': page_lines,
            'raw_text': raw,
            'char_count': len(raw),
            'marker': b['marker'],
        })

    return pages

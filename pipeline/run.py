"""
Main pipeline runner.

Usage:
    python -m pipeline.run <input_file> [--output-dir <dir>] [--verbose]
    python -m pipeline.run --all-samples [--output-dir <dir>]

Flow:
    Raw OCR text
      → Step 1: Segment into pages
      → Step 2: Per-page code extraction (anchors, tables, classification)
      → Step 3: Group pages into doc_instances
      → Step 4: LLM calls for narrative gaps (minimal)
      → Step 5: Normalize into ABDM JSON output
"""

import os
import sys
import json
import argparse
from typing import List, Dict

# Handle both module and direct execution
try:
    from .segmenter import segment
    from .extractor import extract_page
    from .grouper import group_pages
    from .llm_handler import process_llm_gaps
    from .normalizer import normalize_all
except ImportError:
    from segmenter import segment
    from extractor import extract_page
    from grouper import group_pages
    from llm_handler import process_llm_gaps
    from normalizer import normalize_all


def run_pipeline(raw_text: str, verbose: bool = False) -> Dict:
    """
    Run the full pipeline on raw OCR text.

    Returns:
    {
        'documents': [output JSONs per doc_instance],
        'debug': {
            'pages_segmented': int,
            'pages_with_results': int,
            'pages_marketing': int,
            'pages_needing_llm': int,
            'doc_instances': int,
            'total_results_extracted': int,
            'total_llm_calls': int,
        }
    }
    """
    # ── Step 1: Segment ──
    pages = segment(raw_text)
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: Segmented into {len(pages)} pages")
        for p in pages:
            print(f"  {p['page_id']}: {p['char_count']} chars ({p['marker']})")

    # ── Step 2: Per-page extraction ──
    extracted_pages = [extract_page(p) for p in pages]
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 2: Per-page extraction")
        for p in extracted_pages:
            anchors_summary = ', '.join(f"{k}={v}" for k, v in list(p['anchors'].items())[:3])
            print(f"  {p['page_id']}: {p['classification']['page_type']:30s} "
                  f"| {p['department']:20s} "
                  f"| {len(p['results']):2d} results "
                  f"| LLM={'YES' if p['needs_llm'] else 'no':3s} "
                  f"| {anchors_summary}")

    # ── Step 3: Grouping ──
    doc_instances = group_pages(extracted_pages)
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 3: Grouped into {len(doc_instances)} doc_instance(s)")
        for d in doc_instances:
            print(f"  {d['doc_id']}: hiType={d['hiType']:<25s} "
                  f"| {len(d['pages'])} pages "
                  f"| {d['total_results']} results "
                  f"| key={d['group_key'][:50]}")
            print(f"    result_pages={len(d['result_pages'])}, "
                  f"commentary={len(d['commentary_pages'])}, "
                  f"marketing={len(d['marketing_pages'])}, "
                  f"narrative(LLM)={len(d['narrative_pages'])}")

    # ── Step 4: LLM for gaps ──
    doc_instances = process_llm_gaps(doc_instances)
    total_llm_calls = sum(d.get('llm_calls_made', 0) for d in doc_instances)
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 4: LLM calls made: {total_llm_calls}")
        for d in doc_instances:
            if d.get('llm_calls_made', 0) > 0:
                print(f"  {d['doc_id']}: {d['llm_calls_made']} calls")

    # ── Step 5: Normalize ──
    outputs = normalize_all(doc_instances)
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 5: Generated {len(outputs)} output document(s)")
        for o in outputs:
            result_count = len(o.get('smartData', {}).get('data', [])) if o.get('smartData') else 0
            print(f"  hiType={o['hiType']:<25s} | confidence={o['hiTypeConfidence']} | results={result_count}")

    # ── Debug summary ──
    debug = {
        'pages_segmented': len(pages),
        'pages_with_results': sum(1 for p in extracted_pages if p['results']),
        'pages_marketing': sum(1 for p in extracted_pages
                               if p['classification']['page_type'] == 'MARKETING'),
        'pages_commentary_only': sum(1 for p in extracted_pages
                                      if p['classification']['page_type'] in ('COMMENTARY', 'REFERENCE_TABLE')
                                      and not p['results']),
        'pages_needing_llm': sum(1 for p in extracted_pages if p['needs_llm']),
        'doc_instances': len(doc_instances),
        'total_results_extracted': sum(
            len(o.get('smartData', {}).get('data', [])) if o.get('smartData') else 0
            for o in outputs
        ),
        'total_llm_calls': total_llm_calls,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        for k, v in debug.items():
            print(f"  {k}: {v}")

    return {
        'documents': outputs,
        'debug': debug,
    }


def process_file(filepath: str, output_dir: str, verbose: bool = False):
    """Process a single OCR text file."""
    print(f"\n{'#'*60}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'#'*60}")

    with open(filepath, 'r') as f:
        raw_text = f.read()

    result = run_pipeline(raw_text, verbose=verbose)

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    for i, doc in enumerate(result['documents']):
        hitype_clean = doc['hiType'].replace(' ', '_')
        out_name = f"{base_name}__{hitype_clean}_{i+1}.json"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        result_count = len(doc.get('smartData', {}).get('data', [])) if doc.get('smartData') else 0
        print(f"  → {out_name} (hiType={doc['hiType']}, results={result_count})")

    # Write debug info
    debug_path = os.path.join(output_dir, f"{base_name}__debug.json")
    with open(debug_path, 'w') as f:
        json.dump(result['debug'], f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description='OCR → JSON Pipeline')
    parser.add_argument('input_file', nargs='?', help='Path to OCR text file')
    parser.add_argument('--all-samples', action='store_true',
                        help='Process all sample files in ocrissue/')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <input_dir>/pipeline_output)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed pipeline trace')
    args = parser.parse_args()

    if args.all_samples:
        # Process all .txt files in the ocrissue directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_files = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.endswith('.txt')
        ]
        output_dir = args.output_dir or os.path.join(base_dir, 'pipeline_output')
    elif args.input_file:
        sample_files = [args.input_file]
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.input_file), 'pipeline_output'
        )
    else:
        parser.print_help()
        return

    for filepath in sorted(sample_files):
        try:
            process_file(filepath, output_dir, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

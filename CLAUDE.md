# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR-to-JSON pipeline for Indian healthcare documents, targeting ABDM (Ayushman Bharat Digital Mission) output schemas. The pipeline converts raw OCR text (from scanned medical reports) into structured JSON, using a **code-first extraction** approach with LLM calls (Mistral 3 14B via Ollama) reserved only for gaps the deterministic parser can't handle.

## Running the Pipeline

```bash
# Single file
python -m pipeline.run <input_file.txt> [--output-dir <dir>] [--verbose]

# All .txt samples in the parent directory
python -m pipeline.run --all-samples [--output-dir <dir>]
```

Output goes to `pipeline_output/` by default. Each input produces per-document JSON files (`<name>__<hiType>_<N>.json`) and a `__debug.json` summary.

## Architecture

The pipeline runs 5 sequential steps, each in its own module under `pipeline/`:

1. **`segmenter.py`** — Splits raw OCR text into pages. Handles literal `\n` from OCR, explicit `--- Page N of M ---` markers, repeated header detection, and single-page fallback.

2. **`extractor.py`** — Per-page extraction (order-agnostic). Extracts anchors (patient ID, barcode, dates, facility, doctor) via two-pass HTML `<td>` + plain-text regex. Detects department/panel from headers. Classifies pages as `RESULT_TABLE | COMMENTARY | MARKETING | NARRATIVE | ADMIN | OTHER`. Parses HTML table rows into structured test results (5-col and 4-col formats). Falls back to inline regex patterns for non-table results.

3. **`grouper.py`** — Groups pages into `doc_instances` by matching anchors (patientId+orderId > patientId+barcodeId > patientId+facility+date > name+facility). Determines HI type per group (Diagnostic Report, Prescription, OP Consultation, Discharge Summary, etc.).

4. **`llm_handler.py`** — LLM calls only for: narrative extraction (prescriptions, OP notes), failed table parses, or ambiguous classification. Currently **stubbed** — `_call_llm()` returns a stub JSON. Uncomment the Mistral/Ollama call block for production. Token budget: 6000 per call.

5. **`normalizer.py`** — Converts doc_instances into ABDM JSON output. Handles date parsing (Indian formats → unix timestamps), unit normalization, LOINC scale type assignment (`Qn`/`Ord`/`Nom`/`Nar`), and per-hiType schema building.

## Key Configuration

`pipeline/config.py` contains all tunable parameters: LLM budget/model settings, page boundary patterns, marketing keywords, department list, HI type classification signals, and anchor extraction regex patterns.

## Output Schemas

Reference schemas are in `output/` — one JSON per HI type (Diagnostic_Report, Prescription, OP_Consultation, Discharge_Summary, etc.). These define the target structure the normalizer emits.

## Important Design Decisions

- **Code-first, LLM-last**: Deterministic regex/HTML parsing handles ~80% of content. LLM is only called for narrative medical text or when the code parser fails on a page that should have results.
- **Order-agnostic**: Each page is processed independently — page order in the OCR stream doesn't matter.
- **Marketing quarantine**: Pages classified as MARKETING or ADMIN are grouped but their content is discarded from output.
- **No external dependencies beyond stdlib** for core pipeline (only `requests` needed if LLM is enabled).

# OCR-to-JSON Pipeline

Converts raw OCR text from Indian healthcare documents (scanned medical reports) into structured JSON compliant with **ABDM (Ayushman Bharat Digital Mission)** output schemas.

## How It Works

The pipeline uses a **code-first extraction** approach — deterministic regex/HTML parsing handles ~80% of content. An LLM (Mistral 7B via Ollama) is called only for narrative medical text or when the code parser fails.

Two pipeline modes are available:

| Mode | Entry Point | Description |
|------|-------------|-------------|
| **Code-first** (5-step) | `pipeline/run.py` | Segmenter → Extractor → Grouper → LLM for gaps → Normalizer |
| **LLM-first** (parallel) | `pipeline/llm_pipeline.py` | Segmenter → LLM per page (parallel) → Merge results |

The **Web API** (`app.py`) uses the LLM-first pipeline.

## Setup

**Requirements:** Python 3.12+

```bash
# Clone
git clone https://github.com/johnpaulusprem/ocr-parser.git
cd ocr-parser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# (Optional) Install dev tools
pip install -e ".[dev]"
```

### LLM Setup (Ollama)

The pipeline calls Mistral 7B through an OpenAI-compatible API. Install and run [Ollama](https://ollama.ai):

```bash
ollama pull mistral:7b
ollama serve  # starts the API on http://localhost:11434
```

## Running the Pipeline

### CLI (Code-first pipeline)

```bash
# Single file
python -m pipeline.run <input_file.txt> --verbose

# All .txt files in the parent directory
python -m pipeline.run --all-samples --output-dir ./my_output

# With custom output directory
python -m pipeline.run report.txt --output-dir ./results --verbose
```

Output goes to `pipeline_output/` by default. Each input produces:
- `<name>__<hiType>_<N>.json` — per-document structured JSON
- `<name>__debug.json` — pipeline debug summary

### Web API (LLM-first pipeline)

```bash
uvicorn app:app --reload
```

The API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | View current LLM configuration |
| `POST` | `/process` | Process an OCR document |

**POST /process** accepts:
- A `.json` file upload (with `extractedText` field) — from Lighton OCR
- A `.txt` file upload — plain OCR text
- A `text` form field — raw text string
- Optional overrides: `max_context_tokens`, `output_budget`

See [Sample Curls](#sample-curls) below for detailed examples.

### Docker

```bash
docker build -t ocr-pipeline .
docker run -p 8000:8000 ocr-pipeline
```

Note: The container needs access to your Ollama instance. Use `--network host` or set `LLM_BASE_URL` to point to your host.

## Sample Curls

### 1. Check current config

```bash
curl http://localhost:8000/config
```

Response:
```json
{
  "max_context_tokens": 16000,
  "prompt_template_tokens": 400,
  "output_budget": 2000,
  "input_budget": 13600,
  "chars_per_token": 3.5,
  "max_input_chars": 47600,
  "llm_model": "mistral:7b",
  "llm_base_url": "http://localhost:11434/v1"
}
```

### 2. Upload a JSON file (Lighton OCR format)

The JSON file must have an `extractedText` field containing the raw OCR text.

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@/path/to/ocr_output.json"
```

### 3. Upload a plain text file

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@/path/to/report.txt"
```

### 4. Send raw text directly

```bash
curl -X POST http://localhost:8000/process \
  -F 'text=Patient Name: Mr. John Paulus\nPatient ID: 817026\nAge: 28 Years\nGender: Male\nReferred By: Dr. Smith\nLab: XYZ Diagnostics\nReport Date: 05/03/2025\n\nHAEMATOLOGY\nTest Name          Result    Unit       Bio. Ref. Interval\nHaemoglobin        14.5      g/dL       13.0 - 17.0\nRBC Count          5.2       mill/cumm  4.5 - 5.5\nWBC Count          7800      /cumm      4000 - 11000\nPlatelet Count     250000    /cumm      150000 - 410000'
```

### 5. Send a JSON string with extractedText

```bash
curl -X POST http://localhost:8000/process \
  -F 'text={"extractedText": "Patient Name: Mrs. Helda Princy\nPatient ID: 392628\nAge: 30 Years\nGender: Female\n\nBIOCHEMISTRY\nTest Name          Result    Unit       Bio. Ref. Interval\nFasting Glucose    95        mg/dL      70 - 100\nHbA1c              5.4       %          4.0 - 5.6\nCreatinine         0.8       mg/dL      0.6 - 1.2"}'
```

### 6. Override token budget (for large documents)

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@large_report.json" \
  -F "max_context_tokens=32000" \
  -F "output_budget=4000"
```

### 7. Multipage lab report with multiple departments

```bash
curl -X POST http://localhost:8000/process \
  -F 'text=--- Page 1 of 3 ---\nPatient Name: Mr. Ravi Kumar\nPatient ID: 548800\nBarcode ID: 0183ZB002913\nAge: 45 Years\nGender: Male\nFacility: Apollo Diagnostics\nReport Date: 10/03/2025\n\nHAEMATOLOGY\nTest Name          Result    Unit       Bio. Ref. Interval\nHaemoglobin        11.2      g/dL       13.0 - 17.0\nPCV                34.5      %          40 - 50\nRBC Count          4.1       mill/cumm  4.5 - 5.5\n\n--- Page 2 of 3 ---\nBIOCHEMISTRY\nTest Name          Result    Unit       Bio. Ref. Interval\nUrea               45        mg/dL      17 - 43\nCreatinine         1.4       mg/dL      0.6 - 1.2\nUric Acid          7.8       mg/dL      3.4 - 7.0\n\n--- Page 3 of 3 ---\nIMMUNOLOGY\nTest Name          Result    Unit       Bio. Ref. Interval\nTSH                3.2       uIU/mL     0.27 - 4.2\nVitamin D          18.5      ng/mL      30 - 100\nVitamin B12        320       pg/mL      211 - 946'
```

### 8. Prescription document

```bash
curl -X POST http://localhost:8000/process \
  -F 'text=Patient Name: Mrs. Lakshmi Devi\nAge: 55 Years\nGender: Female\nDoctor: Dr. Anand Sharma\nDate: 08/03/2025\nOPD\n\nRx\n1. Tab Metformin 500mg  1-0-1  After food\n2. Tab Amlodipine 5mg   1-0-0  Before food\n3. Tab Atorvastatin 10mg 0-0-1  After food\n4. Cap Vitamin D3 60000 IU  Once weekly x 8 weeks\n\nAdvice:\n- Review after 3 months with HbA1c\n- Low salt, low sugar diet\n- Walk 30 mins daily'
```

### 9. Save response to file

```bash
curl -s -X POST http://localhost:8000/process \
  -F "file=@report.json" | python -m json.tool > output.json
```

### 10. Pretty-print response

```bash
curl -s -X POST http://localhost:8000/process \
  -F "file=@report.json" | python -m json.tool
```

## Configuration

All configuration lives in **`pipeline/config.py`**. Every LLM setting can be overridden via environment variables.

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `mistral:7b` | Model name (any Ollama model or OpenAI-compatible model) |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | API endpoint |
| `LLM_API_KEY` | `ollama` | API key (set this if using a hosted provider) |
| `LLM_MAX_CONTEXT_TOKENS` | `16000` | Total token limit (input + output) |
| `LLM_OUTPUT_BUDGET` | `2000` | Tokens reserved for model response |
| `LLM_CHARS_PER_TOKEN` | `3.5` | Characters-per-token estimate for budget calculation |

**Example — use a different model or provider:**

```bash
export LLM_MODEL="llama3:8b"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MAX_CONTEXT_TOKENS=32000
```

### What You Can Edit in `pipeline/config.py`

**Page segmentation** — `PAGE_BOUNDARY_PATTERNS`: regex patterns that detect page breaks in OCR text.

**Marketing filter** — `MARKETING_KEYWORDS` and `DISCLAIMER_KEYWORDS`: strings that mark pages as non-medical content (discarded from output).

**Department detection** — `DEPARTMENTS`: list of medical department names to detect from page headers (e.g., HAEMATOLOGY, BIOCHEMISTRY).

**Document type classification** — `HITYPE_SIGNALS`: keyword lists that determine whether a document is a Diagnostic Report, Prescription, OP Consultation, Discharge Summary, or Immunization Record.

**Patient/report field extraction** — `ANCHOR_PATTERNS`: regex patterns for extracting patient name, ID, barcode, dates, facility, doctor, etc. Edit these if your OCR documents use different label formats.

## Project Structure

```
├── app.py                  # FastAPI web server
├── pipeline/
│   ├── config.py           # All tunable parameters (start here)
│   ├── segmenter.py        # Step 1: Split OCR text into pages
│   ├── extractor.py        # Step 2: Per-page regex/HTML extraction
│   ├── grouper.py          # Step 3: Group pages into documents
│   ├── llm_handler.py      # Step 4: LLM calls for gaps (code-first mode)
│   ├── llm_pipeline.py     # Alternative LLM-first pipeline (used by API)
│   ├── normalizer.py       # Step 5: Convert to ABDM JSON output
│   └── run.py              # CLI entry point
├── Dockerfile
└── pyproject.toml
```

## Supported Document Types

- Diagnostic Report (lab results, imaging)
- Prescription
- OP Consultation
- Discharge Summary
- Immunization Record
- Health Document Record (fallback)

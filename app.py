"""
Web API for the OCR → ABDM JSON pipeline.

Usage:
    uvicorn app:app --reload
    # Then POST to http://localhost:8000/process
"""

import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pipeline import config
from pipeline.llm_pipeline import run_llm_pipeline

app = FastAPI(
    title="OCR Pipeline API",
    description="Converts raw OCR text from Indian healthcare documents into structured ABDM JSON.",
    version="0.2.0",
)


@app.get("/config")
async def get_config():
    """Return current LLM budget configuration."""
    return {
        "max_context_tokens": config.MAX_CONTEXT_TOKENS,
        "prompt_template_tokens": config.PROMPT_TEMPLATE_TOKENS,
        "output_budget": config.OUTPUT_BUDGET,
        "input_budget": config.INPUT_BUDGET,
        "chars_per_token": config.CHARS_PER_TOKEN,
        "max_input_chars": config.MAX_INPUT_CHARS,
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
    }


@app.post("/process")
async def process_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    max_context_tokens: Optional[int] = Form(None),
    output_budget: Optional[int] = Form(None),
):
    """
    Process OCR text through the LLM-first pipeline.

    Accepts:
    - A JSON file from Lighton OCR (with `extractedText` field) via `file`
    - A plain .txt file via `file`
    - Raw text via `text` form field
    - max_context_tokens: Override total token budget (default: 16000)
    - output_budget: Override output token budget (default: 2000)

    Returns a single merged JSON output.
    """
    # Apply per-request budget overrides
    if max_context_tokens is not None:
        config.MAX_CONTEXT_TOKENS = max_context_tokens
        config.INPUT_BUDGET = config.MAX_CONTEXT_TOKENS - config.PROMPT_TEMPLATE_TOKENS - config.OUTPUT_BUDGET
        config.MAX_INPUT_CHARS = int(config.INPUT_BUDGET * config.CHARS_PER_TOKEN)
    if output_budget is not None:
        config.OUTPUT_BUDGET = output_budget
        config.INPUT_BUDGET = config.MAX_CONTEXT_TOKENS - config.PROMPT_TEMPLATE_TOKENS - config.OUTPUT_BUDGET
        config.MAX_INPUT_CHARS = int(config.INPUT_BUDGET * config.CHARS_PER_TOKEN)
    if file:
        raw_bytes = await file.read()
        raw_content = raw_bytes.decode("utf-8")

        if file.filename and file.filename.endswith(".json"):
            try:
                data = json.loads(raw_content)
                raw_text = data.get("extractedText", "")
                if not raw_text:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "JSON file has no 'extractedText' field."},
                    )
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid JSON file."},
                )
        else:
            raw_text = raw_content
    elif text:
        try:
            data = json.loads(text)
            raw_text = data.get("extractedText", text)
        except (json.JSONDecodeError, AttributeError):
            raw_text = text
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide either a 'file' upload or 'text' field."},
        )

    result = run_llm_pipeline(raw_text, verbose=True)

    return result

# Project Overview and Architecture Notes

## 1) What This Project Is About

This project is an OCR testing platform focused on extracting text and field-value pairs from TIFF form images. It is designed to:

- run multiple OCR engines on the same input;
- compare extraction quality and confidence;
- reconstruct coherent lines and structured fields from OCR output;
- optionally combine multiple engine outputs using deterministic ensemble voting;
- export machine-readable JSON and optional HTML visualization output.

In short, it supports practical form digitization by reducing manual entry effort and giving measurable, comparable OCR outcomes.

## 2) Potential Attempts/Approaches Used in the Project

Based on the implementation and the final year project direction (AI + ICR for manual forms), the project aligns with these practical attempts:

1. **Single-engine OCR baseline**  
   Start with one engine (for example, Tesseract) to establish a minimum viable extraction flow.

2. **Multi-engine comparison**  
   Run multiple engines (Tesseract, PaddleOCR, EasyOCR, TrOCR) on the same document and compare strengths across print quality, handwriting noise, and layout variation.

3. **Structured field recovery**  
   Convert raw OCR lines into coherent form-like `label: value` fields to make output directly useful for downstream storage.

4. **Deterministic ensemble voting**  
   Combine multiple successful OCR outputs to improve field reliability where one engine alone is inconsistent.

5. **Human-review friendly reporting**  
   Generate JSON for system integration and HTML for quick manual verification of extracted values.

## 3) Typical Errors Encountered (and Why)

The project already anticipates several recurring OCR pipeline errors:

- **Missing OCR dependencies** (ImportError)  
  Happens when selected engines are not installed.

- **Tesseract binary not available**  
  Python package may be installed, but system Tesseract executable is missing.

- **TrOCR model availability issues** (OSError)  
  Occurs when model download is blocked/offline and no local `TROCR_MODEL` path is supplied.

- **Unsupported file format**  
  The CLI supports `.tif/.tiff`; other formats are rejected.

- **Invalid/empty TIFF content**  
  Input with unreadable or empty pages causes processing failures.

- **Insufficient ensemble inputs**  
  Ensemble voting requires at least two successful engine outputs.

These are expected risks in document digitization workflows and should be handled as part of operational setup.

## 4) Recommended Minimal Setup

A minimal, practical setup for reliable use:

1. **Environment**
   - Python 3.10+ (or current project-supported version)
   - Create and activate a virtual environment

2. **Core package**
   - Install `pillow`

3. **At least one OCR engine**
   - Tesseract path: install `pytesseract` **and** system tesseract binary
   - OR install one of: `paddleocr`, `easyocr`, or `transformers torch` (for TrOCR)

4. **Input discipline**
   - Use clean `.tif/.tiff` scans with consistent orientation and adequate resolution

5. **Validation**
   - Run unit tests with `python -m unittest -q`

For stronger production reliability, prefer at least two engines plus ensemble mode.

## 5) Architecture Recommendations (Using the Final Year Project Direction)

The final year project theme emphasizes AI/ICR for manual form processing, including noisy scans and structured extraction needs. A recommended architecture for this repository is:

### A. Layered OCR Processing Architecture

1. **Input & Preprocessing Layer**
   - Document intake, format validation, page normalization, and quality checks.

2. **OCR Engine Layer (Pluggable)**
   - Multiple independent OCR engines run in parallel or sequence.

3. **Normalization & Parsing Layer**
   - Convert engine-specific outputs into a common schema (lines, confidences, fields).

4. **Consensus Layer (Ensemble)**
   - Field-level voting and confidence aggregation for more stable extraction.

5. **Output & Review Layer**
   - JSON for systems, HTML for analysts, and optional human correction workflow.

### B. Deployment Recommendation by Maturity

- **Phase 1 (Current/Minimal):** local single service CLI with optional multi-engine support.
- **Phase 2 (Operational):** modular service split (ingestion, OCR workers, aggregation, output store).
- **Phase 3 (Scale):** asynchronous queue-based pipeline with monitored workers and audit logs.

### C. Why This Fits the Final Year Project

This architecture directly supports the documented problem domain:

- reduces manual data entry burden;
- supports mixed document quality and handwritten/printed variability;
- improves extraction robustness through multi-engine + consensus strategy;
- keeps the system extendable for newer AI form-understanding models.

## 6) Suggested Next Practical Improvements

- add preprocessing options (deskew, denoise, contrast enhancement) before OCR;
- define form-specific validation rules for critical fields;
- add confidence thresholds for auto-accept vs. manual review;
- maintain benchmark sets and compare engines periodically;
- add optional cloud OCR adapters when needed for difficult document types.

---

This file is intentionally separate from `README.md` and serves as a project-level documentation note for background, attempts, known errors, setup, and architecture direction.

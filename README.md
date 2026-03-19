# OCR Testing Platform (Free, Local)

This repository now provides a **minimal local OCR platform** to compare OCR engines on TIFF files and return extracted text with confidence scores.

## What it supports

- **Tesseract** (`pytesseract`)
- **PaddleOCR** (`paddleocr`)
- **EasyOCR** (`easyocr`)
- **TrOCR** (`transformers` + `torch`)  
  *(text extraction supported; confidence omitted in this minimal path)*

Each engine is optional. If one is not installed, the report includes a clear error for that engine while other selected engines continue to run.

## Minimal setup

```bash
python -m pip install pillow
```

Install only the engine(s) you want:

```bash
python -m pip install pytesseract
python -m pip install paddleocr
python -m pip install easyocr
python -m pip install transformers torch
```

> For Tesseract, you also need the system Tesseract binary installed on your machine.

## Usage

Run OCR on a TIFF/TIF file:

```bash
python main.py /path/to/file.tiff --engines tesseract paddleocr easyocr trocr
```

Save output JSON:

```bash
python main.py /path/to/file.tiff --engines tesseract easyocr --output report.json
```

The JSON output includes:

- file path
- page count
- per-engine status
- extracted lines (`page`, `text`, `confidence`)
- coherent reconstructed lines (`coherent_lines`)
- detected field-value pairs (`fields`)
- average confidence per engine (when available)

# OCR Testing Platform

This repository provides an OCR platform to compare engines on TIFF files and return extracted text with confidence scores.

## What it supports

### Local engines

- **Tesseract** (`pytesseract`) — confidence scale 0–100
- **PaddleOCR** (`paddleocr`) — confidence scale 0–1
- **EasyOCR** (`easyocr`) — confidence scale 0–1
- **TrOCR** (`transformers` + `torch`) — no confidence score in this path

### Cloud Document AI engines (higher accuracy, measurable metrics)

- **Amazon Textract** (`textract`) — confidence scale 0–1 (normalised from native 0–100)
- **Google Document AI** (`google_docai`) — confidence scale 0–1

Cloud engines are significantly more accurate than local engines and return a measurable confidence score for every extracted line.

Each engine is optional. If one is not installed or its credentials are missing, the report includes a clear error for that engine while other selected engines continue to run.

## Minimal setup

```bash
python -m pip install pillow
```

Install only the engine(s) you want:

```bash
# Local engines
python -m pip install pytesseract          # also requires the system tesseract binary
python -m pip install paddleocr
python -m pip install easyocr
python -m pip install transformers torch

# Cloud engines
python -m pip install boto3                # Amazon Textract
python -m pip install google-cloud-documentai  # Google Document AI
```

> For Tesseract, you also need the system Tesseract binary installed on your machine.

## Cloud engine credentials

### Amazon Textract

Standard AWS credential chain is used automatically (environment variables, `~/.aws/credentials`, IAM role, etc.):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1   # must support Textract
```

### Google Document AI

Application Default Credentials (ADC) or a service-account key file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Required processor settings:
export GOOGLE_DOCAI_PROJECT_ID=my-gcp-project
export GOOGLE_DOCAI_PROCESSOR_ID=my-processor-id
export GOOGLE_DOCAI_LOCATION=us      # default; use "eu" for EU processors
```

Create a Document OCR processor in the [Google Cloud Console](https://console.cloud.google.com/ai/document-ai) and use its processor ID above.

## Usage

Run OCR on a TIFF/TIF file:

```bash
python main.py /path/to/file.tiff --engines tesseract paddleocr easyocr trocr
```

Use cloud engines for higher accuracy:

```bash
python main.py /path/to/file.tiff --engines textract google_docai
```

Save output JSON:

```bash
python main.py /path/to/file.tiff --engines textract --output report.json
```

The JSON output includes:

- file path
- page count
- per-engine status
- extracted lines (`page`, `text`, `confidence`)
- average confidence per engine (when available)

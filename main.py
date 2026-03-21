#!/usr/bin/env python3
"""Minimal local OCR testing platform for TIFF files.

Supports optional OCR engines when installed:
- tesseract (pytesseract + system tesseract binary)
- paddleocr
- easyocr
- trocr (transformers + torch)
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


SUPPORTED_ENGINES = ("tesseract", "paddleocr", "easyocr", "trocr", "textract", "google_docai")


@dataclass
class OCRLine:
    page: int
    text: str
    confidence: float | None


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _average_confidence(lines: list[OCRLine]) -> float | None:
    confidences = [line.confidence for line in lines if line.confidence is not None]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


def _open_tiff_pages(path: str) -> list[Any]:
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        pages = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(img)]
    if not pages:
        raise ValueError("No pages were found in the TIFF file.")
    return pages


def _parse_tesseract_data(page_index: int, data: dict[str, list[str]]) -> list[OCRLine]:
    lines: list[OCRLine] = []
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    for idx, text in enumerate(texts):
        clean = _clean_text(text)
        if not clean:
            continue
        raw_conf = confidences[idx] if idx < len(confidences) else None
        try:
            conf = float(raw_conf) if raw_conf not in (None, "", "-1") else None
        except (TypeError, ValueError):
            conf = None
        lines.append(OCRLine(page=page_index, text=clean, confidence=conf))
    return lines


def _parse_paddle_result(page_index: int, result: Any) -> list[OCRLine]:
    lines: list[OCRLine] = []
    if not result:
        return lines

    # paddleocr commonly returns nested list(s) per page:
    # [[[box], (text, score)], ...] or [None]
    page_items = result[0] if isinstance(result, list) and result else []
    for item in page_items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        payload = item[1]
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            continue
        text = _clean_text(payload[0])
        if not text:
            continue
        try:
            conf = float(payload[1])
        except (TypeError, ValueError):
            conf = None
        lines.append(OCRLine(page=page_index, text=text, confidence=conf))
    return lines


def _parse_easyocr_result(page_index: int, result: Any) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = _clean_text(item[1])
        if not text:
            continue
        try:
            conf = float(item[2])
        except (TypeError, ValueError):
            conf = None
        lines.append(OCRLine(page=page_index, text=text, confidence=conf))
    return lines


def _run_tesseract(pages: list[Any]) -> dict[str, Any]:
    import pytesseract
    from pytesseract import Output

    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        data = pytesseract.image_to_data(page, output_type=Output.DICT)
        lines.extend(_parse_tesseract_data(i, data))
    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
    }


def _run_paddleocr(pages: list[Any]) -> dict[str, Any]:
    import numpy as np

    ocr = _get_paddleocr_reader()
    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        result = ocr.ocr(np.array(page), cls=True)
        lines.extend(_parse_paddle_result(i, result))
    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
    }


def _run_easyocr(pages: list[Any]) -> dict[str, Any]:
    import numpy as np

    reader = _get_easyocr_reader()
    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        result = reader.readtext(np.array(page))
        lines.extend(_parse_easyocr_result(i, result))
    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
    }


def _run_trocr(pages: list[Any]) -> dict[str, Any]:
    import torch

    processor, model = _get_trocr_processor_and_model()
    model.eval()

    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        pixel_values = processor(images=page, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = _clean_text(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        if text:
            # TrOCR does not expose token-level confidence in this basic path.
            lines.append(OCRLine(page=i, text=text, confidence=None))

    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
        "note": "TrOCR confidence is not available in this minimal implementation.",
    }


def _run_textract(pages: list[Any]) -> dict[str, Any]:
    """Run Amazon Textract on each page and return line-level results.

    Requires ``boto3`` and valid AWS credentials (environment variables
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION``
    or an IAM role/profile configured on the host).

    Textract ``Confidence`` values (0–100) are normalised to 0–1 so that
    they are directly comparable with EasyOCR and PaddleOCR outputs.
    """
    client = _get_textract_client()
    lines: list[OCRLine] = []

    for i, page in enumerate(pages):
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")

        response = client.detect_document_text(Document={"Bytes": buffer.getvalue()})

        for block in response.get("Blocks", []):
            if block.get("BlockType") != "LINE":
                continue
            text = _clean_text(block.get("Text", ""))
            if not text:
                continue
            raw_conf = block.get("Confidence")
            conf = round(float(raw_conf) / 100.0, 4) if raw_conf is not None else None
            lines.append(OCRLine(page=i, text=text, confidence=conf))

    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
    }


def _run_google_docai(pages: list[Any]) -> dict[str, Any]:
    """Run Google Document AI on each page and return line-level results.

    Requires ``google-cloud-documentai`` and valid credentials
    (``GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials).

    The following environment variables must be set:

    * ``GOOGLE_DOCAI_PROJECT_ID`` – GCP project ID
    * ``GOOGLE_DOCAI_PROCESSOR_ID`` – Document AI processor ID
    * ``GOOGLE_DOCAI_LOCATION`` – processor location, defaults to ``"us"``

    Confidence values are returned on the 0–1 scale as provided by the API.
    """
    project_id = os.environ.get("GOOGLE_DOCAI_PROJECT_ID", "")
    location = os.environ.get("GOOGLE_DOCAI_LOCATION", "us")
    processor_id = os.environ.get("GOOGLE_DOCAI_PROCESSOR_ID", "")

    if not project_id or not processor_id:
        raise EnvironmentError(
            "GOOGLE_DOCAI_PROJECT_ID, GOOGLE_DOCAI_PROCESSOR_ID, and "
            "GOOGLE_DOCAI_LOCATION (default: 'us') environment variables must be set."
        )

    from google.cloud import documentai_v1 as documentai

    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    processor_name = client.processor_path(project_id, location, processor_id)

    lines: list[OCRLine] = []

    for i, page in enumerate(pages):
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")

        raw_document = documentai.RawDocument(
            content=buffer.getvalue(),
            mime_type="image/png",
        )
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )
        result = client.process_document(request=request)
        document = result.document

        for doc_page in document.pages:
            for line in doc_page.lines:
                segments = line.layout.text_anchor.text_segments
                line_text = _clean_text(
                    "".join(
                        document.text[seg.start_index : seg.end_index]
                        for seg in segments
                    )
                )
                if not line_text:
                    continue
                raw_conf = line.layout.confidence
                conf = round(float(raw_conf), 4) if raw_conf else None
                lines.append(OCRLine(page=i, text=line_text, confidence=conf))

    return {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
    }


ENGINE_RUNNERS = {
    "tesseract": _run_tesseract,
    "paddleocr": _run_paddleocr,
    "easyocr": _run_easyocr,
    "trocr": _run_trocr,
    "textract": _run_textract,
    "google_docai": _run_google_docai,
}


@functools.lru_cache(maxsize=1)
def _get_paddleocr_reader() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en")


@functools.lru_cache(maxsize=1)
def _get_easyocr_reader() -> Any:
    from easyocr import Reader

    return Reader(["en"], gpu=False)


@functools.lru_cache(maxsize=1)
def _get_trocr_processor_and_model() -> tuple[Any, Any]:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model


def _get_textract_client() -> Any:
    import boto3

    return boto3.client("textract")


def run_ocr(path: str, engines: list[str]) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not path.lower().endswith((".tif", ".tiff")):
        raise ValueError("Only .tif/.tiff files are supported.")

    pages = _open_tiff_pages(path)
    results: dict[str, Any] = {}
    for engine in engines:
        runner = ENGINE_RUNNERS[engine]
        try:
            results[engine] = runner(pages)
        except ImportError as exc:
            install_hints = {
                "tesseract": "pip install pytesseract (and install the tesseract system binary)",
                "paddleocr": "pip install paddleocr",
                "easyocr": "pip install easyocr",
                "trocr": "pip install transformers torch",
                "textract": "pip install boto3",
                "google_docai": "pip install google-cloud-documentai",
            }
            results[engine] = {
                "status": "error",
                "engine": engine,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": f"Engine not available. Install with: {install_hints.get(engine, 'see documentation')}",
            }
        except EnvironmentError as exc:
            results[engine] = {
                "status": "error",
                "engine": engine,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "Check that all required environment variables are set (see README).",
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            results[engine] = {
                "status": "error",
                "engine": engine,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "file": path,
        "page_count": len(pages),
        "engines": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR on a TIFF file using Tesseract, PaddleOCR, EasyOCR, and/or TrOCR."
    )
    parser.add_argument("file", help="Absolute or relative path to input .tif/.tiff file")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["tesseract"],
        choices=SUPPORTED_ENGINES,
        help="OCR engine(s) to run",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file path. If omitted, results are printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        payload = run_ocr(args.file, args.engines)
        output = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                file.write(output + "\n")
            print(f"Saved OCR report to {args.output}")
        else:
            print(output)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        error_payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

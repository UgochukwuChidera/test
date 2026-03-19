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
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


SUPPORTED_ENGINES = ("tesseract", "paddleocr", "easyocr", "trocr")
_FIELD_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()'&\-.]{0,80})\s*:\s*(.*)$")


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

    block_nums = data.get("block_num", [])
    par_nums = data.get("par_num", [])
    line_nums = data.get("line_num", [])
    has_line_metadata = (
        len(block_nums) == len(texts)
        and len(par_nums) == len(texts)
        and len(line_nums) == len(texts)
    )

    line_keys: list[tuple[int, int, int]] = []
    if has_line_metadata:
        try:
            line_keys = [
                (int(block_nums[idx]), int(par_nums[idx]), int(line_nums[idx]))
                for idx in range(len(texts))
            ]
        except (TypeError, ValueError):
            has_line_metadata = False

    if not has_line_metadata:
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

    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx, text in enumerate(texts):
        clean = _clean_text(text)
        if not clean:
            continue

        raw_conf = confidences[idx] if idx < len(confidences) else None
        try:
            conf = float(raw_conf) if raw_conf not in (None, "", "-1") else None
        except (TypeError, ValueError):
            conf = None

        key = line_keys[idx]
        bucket = grouped.setdefault(
            key,
            {"parts": [], "confidences": []},
        )
        bucket["parts"].append(clean)
        if conf is not None:
            bucket["confidences"].append(conf)

    for key in sorted(grouped):
        payload = grouped[key]
        line_text = " ".join(payload["parts"]).strip()
        if not line_text:
            continue
        conf_values = payload["confidences"]
        line_conf = (
            round(sum(conf_values) / len(conf_values), 4) if conf_values else None
        )
        lines.append(OCRLine(page=page_index, text=line_text, confidence=line_conf))
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


def _build_coherent_output(lines: list[OCRLine]) -> tuple[list[str], dict[str, str]]:
    coherent_lines: list[str] = []
    fields: dict[str, str] = {}
    pending_field_label: str | None = None
    pending_field_value = ""

    def flush_pending_field() -> None:
        nonlocal pending_field_label, pending_field_value
        if pending_field_label is None:
            return
        normalized = " ".join(pending_field_value.split())
        fields[pending_field_label] = normalized
        coherent_lines.append(
            f"{pending_field_label}: {normalized}" if normalized else f"{pending_field_label}:"
        )
        pending_field_label = None
        pending_field_value = ""

    for line in lines:
        text = _clean_text(line.text)
        if not text:
            continue

        label_match = _FIELD_LABEL_RE.match(text)
        if label_match:
            flush_pending_field()
            pending_field_label = label_match.group(1).strip()
            pending_field_value = label_match.group(2).strip()
            continue

        if pending_field_label is not None:
            pending_field_value = (
                f"{pending_field_value} {text}".strip() if pending_field_value else text
            )
            continue

        coherent_lines.append(text)

    flush_pending_field()
    return coherent_lines, fields


def _build_engine_payload(lines: list[OCRLine], *, note: str | None = None) -> dict[str, Any]:
    coherent_lines, fields = _build_coherent_output(lines)
    payload: dict[str, Any] = {
        "status": "ok",
        "line_count": len(lines),
        "average_confidence": _average_confidence(lines),
        "lines": [asdict(line) for line in lines],
        "coherent_lines": coherent_lines,
        "fields": fields,
    }
    if note:
        payload["note"] = note
    return payload


def _run_tesseract(pages: list[Any]) -> dict[str, Any]:
    import pytesseract
    from pytesseract import Output

    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        data = pytesseract.image_to_data(page, output_type=Output.DICT)
        lines.extend(_parse_tesseract_data(i, data))
    return _build_engine_payload(lines)


def _run_paddleocr(pages: list[Any]) -> dict[str, Any]:
    import numpy as np

    ocr = _get_paddleocr_reader()
    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        result = ocr.ocr(np.array(page), cls=True)
        lines.extend(_parse_paddle_result(i, result))
    return _build_engine_payload(lines)


def _run_easyocr(pages: list[Any]) -> dict[str, Any]:
    import numpy as np

    reader = _get_easyocr_reader()
    lines: list[OCRLine] = []
    for i, page in enumerate(pages):
        result = reader.readtext(np.array(page))
        lines.extend(_parse_easyocr_result(i, result))
    return _build_engine_payload(lines)


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

    return _build_engine_payload(
        lines,
        note="TrOCR confidence is not available in this minimal implementation.",
    )


ENGINE_RUNNERS = {
    "tesseract": _run_tesseract,
    "paddleocr": _run_paddleocr,
    "easyocr": _run_easyocr,
    "trocr": _run_trocr,
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
            }
            results[engine] = {
                "status": "error",
                "engine": engine,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": f"Engine not available. Install with: {install_hints[engine]}",
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

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
import html
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


SUPPORTED_ENGINES = ("tesseract", "paddleocr", "easyocr", "trocr")
_MAX_FIELD_LABEL_LENGTH = 80
_MAX_FIELD_CONTINUATION_LINES = 1
_CONFIDENCE_PRECISION = 4
# Field labels in these forms are short phrases like "Mode of Study",
# "Registration ID", "Parent/Guardian Name", or "O'Level Result", followed by ":".
# Allowed punctuation is intentionally limited to separators seen in form labels:
# spaces, slash, parentheses, apostrophe, ampersand, hyphen, and period.
_FIELD_LABEL_RE = re.compile(
    rf"^\s*([A-Za-z][A-Za-z0-9 /()'&\-.]{{0,{_MAX_FIELD_LABEL_LENGTH}}})\s*:\s*(.*)$"
)


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
    return round(sum(confidences) / len(confidences), _CONFIDENCE_PRECISION)


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
            round(sum(conf_values) / len(conf_values), _CONFIDENCE_PRECISION)
            if conf_values
            else None
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
    pending_field_continuations = 0

    def flush_pending_field() -> None:
        nonlocal pending_field_label, pending_field_value, pending_field_continuations
        if pending_field_label is None:
            return
        normalized = " ".join(pending_field_value.split())
        fields[pending_field_label] = normalized
        coherent_lines.append(
            f"{pending_field_label}:" + (f" {normalized}" if normalized else "")
        )
        pending_field_label = None
        pending_field_value = ""
        pending_field_continuations = 0

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

        if (
            pending_field_label is not None
            and pending_field_continuations < _MAX_FIELD_CONTINUATION_LINES
        ):
            # Allow one continuation line to capture wrapped field values while
            # avoiding spillover from unrelated following sections.
            pending_field_value = (
                f"{pending_field_value} {text}".strip() if pending_field_value else text
            )
            pending_field_continuations += 1
            continue
        if pending_field_label is not None:
            flush_pending_field()

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


def _normalize_vote_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _build_ensemble_payload(results: dict[str, Any], engines: list[str]) -> dict[str, Any]:
    successful = [
        engine
        for engine in engines
        if isinstance(results.get(engine), dict) and results[engine].get("status") == "ok"
    ]
    if len(successful) < 2:
        return {
            "status": "error",
            "error": "Need at least two successful OCR engine outputs for ensemble voting.",
        }

    labels_in_order: list[str] = []
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in successful:
        payload = results[engine]
        avg_conf = payload.get("average_confidence")
        fields = payload.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        for label, raw_value in fields.items():
            label_text = _clean_text(label)
            value_text = _clean_text(raw_value)
            if not label_text:
                continue
            if label_text not in labels_in_order:
                labels_in_order.append(label_text)
            normalized = _normalize_vote_text(value_text)
            label_bucket = candidates.setdefault(label_text, {})
            candidate = label_bucket.setdefault(
                normalized,
                {
                    "value": value_text,
                    "engines": [],
                    "vote_count": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                },
            )
            candidate["engines"].append(engine)
            candidate["vote_count"] += 1
            if isinstance(avg_conf, (int, float)):
                candidate["confidence_sum"] += float(avg_conf)
                candidate["confidence_count"] += 1

    ensemble_fields: dict[str, str] = {}
    field_votes: dict[str, list[dict[str, Any]]] = {}
    supporting_confidences: list[float] = []

    for label in labels_in_order:
        label_candidates = list(candidates.get(label, {}).values())
        if not label_candidates:
            continue
        ranked = sorted(
            label_candidates,
            key=lambda item: (
                int(item["vote_count"]),
                float(item["confidence_sum"]),
                len(str(item["value"])),
                str(item["value"]).lower(),
            ),
            reverse=True,
        )
        winner = ranked[0]
        ensemble_fields[label] = winner["value"]
        if winner["confidence_count"]:
            supporting_confidences.append(winner["confidence_sum"] / winner["confidence_count"])
        field_votes[label] = [
            {
                "value": item["value"],
                "engines": item["engines"],
                "vote_count": item["vote_count"],
                "average_engine_confidence": (
                    round(item["confidence_sum"] / item["confidence_count"], _CONFIDENCE_PRECISION)
                    if item["confidence_count"]
                    else None
                ),
            }
            for item in ranked
        ]

    coherent_lines = [f"{label}: {value}".rstrip() for label, value in ensemble_fields.items()]
    average_confidence = (
        round(sum(supporting_confidences) / len(supporting_confidences), _CONFIDENCE_PRECISION)
        if supporting_confidences
        else None
    )
    return {
        "status": "ok",
        "line_count": len(coherent_lines),
        "average_confidence": average_confidence,
        "coherent_lines": coherent_lines,
        "fields": ensemble_fields,
        "field_votes": field_votes,
        "source_engines": successful,
        "note": "Deterministic field-level voting across successful engines.",
    }


def _render_html_report(payload: dict[str, Any]) -> str:
    engines = payload.get("engines", {})
    rows: list[str] = []
    for engine, result in engines.items():
        fields = result.get("fields") if isinstance(result, dict) else {}
        field_lines = "<br>".join(
            f"<strong>{html.escape(str(k))}</strong>: {html.escape(str(v))}"
            for k, v in (fields.items() if isinstance(fields, dict) else [])
        )
        if not field_lines:
            field_lines = "<em>No fields</em>"
        rows.append(
            "<tr>"
            f"<td>{html.escape(engine)}</td>"
            f"<td>{html.escape(str(result.get('status', 'unknown')) if isinstance(result, dict) else 'unknown')}</td>"
            f"<td>{html.escape(str(result.get('average_confidence')) if isinstance(result, dict) else '')}</td>"
            f"<td>{field_lines}</td>"
            "</tr>"
        )

    ensemble = engines.get("ensemble", {}) if isinstance(engines, dict) else {}
    ensemble_rows = ""
    if isinstance(ensemble, dict) and ensemble.get("status") == "ok":
        for label, value in (ensemble.get("fields") or {}).items():
            ensemble_rows += (
                "<tr>"
                f"<td>{html.escape(str(label))}</td>"
                f"<td>{html.escape(str(value))}</td>"
                "</tr>"
            )
    if not ensemble_rows:
        ensemble_rows = "<tr><td colspan='2'><em>No ensemble fields</em></td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Field Visualizer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    .muted {{ color: #475569; font-size: 0.95em; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>OCR Field Visualizer</h1>
    <p class="muted"><strong>File:</strong> {html.escape(str(payload.get("file", "")))}<br>
    <strong>Pages:</strong> {html.escape(str(payload.get("page_count", "")))}<br>
    <strong>Generated:</strong> {html.escape(str(payload.get("generated_at", "")))}</p>
  </div>
  <div class="card">
    <h2>Ensemble (Voted Fields)</h2>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>{ensemble_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Per-engine Field Mapping</h2>
    <table>
      <thead><tr><th>Engine</th><th>Status</th><th>Average Confidence</th><th>Fields</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""


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


def run_ocr(path: str, engines: list[str], *, include_ensemble: bool = False) -> dict[str, Any]:
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
    if include_ensemble:
        results["ensemble"] = _build_ensemble_payload(results, engines)

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
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable deterministic multi-engine field voting (requires at least 2 successful engines).",
    )
    parser.add_argument(
        "--html-output",
        default="",
        help="Optional output HTML report path for field/value visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        payload = run_ocr(args.file, args.engines, include_ensemble=args.ensemble)
        output = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                file.write(output + "\n")
            print(f"Saved OCR report to {args.output}")
        else:
            print(output)
        if args.html_output:
            html_report = _render_html_report(payload)
            with open(args.html_output, "w", encoding="utf-8") as file:
                file.write(html_report)
            print(f"Saved OCR HTML report to {args.html_output}")
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

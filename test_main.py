import unittest
from unittest.mock import patch

import main
from main import (
    OCRLine,
    _average_confidence,
    _build_coherent_output,
    _parse_easyocr_result,
    _parse_paddle_result,
    _parse_tesseract_data,
)


class OCRParsingTests(unittest.TestCase):
    @staticmethod
    def _raise_import_error(_pages: object) -> None:
        raise ImportError("missing")

    def test_average_confidence_ignores_none(self) -> None:
        lines = [
            OCRLine(page=0, text="a", confidence=0.8),
            OCRLine(page=0, text="b", confidence=None),
            OCRLine(page=0, text="c", confidence=0.6),
        ]
        self.assertEqual(_average_confidence(lines), 0.7)

    def test_parse_tesseract_data_extracts_text_and_confidence(self) -> None:
        parsed = _parse_tesseract_data(
            1,
            {
                "text": ["", "hello", "world"],
                "conf": ["-1", "84.2", "90"],
            },
        )
        self.assertEqual(
            [(line.page, line.text, line.confidence) for line in parsed],
            [(1, "hello", 84.2), (1, "world", 90.0)],
        )

    def test_parse_tesseract_data_groups_words_into_lines_when_metadata_present(self) -> None:
        parsed = _parse_tesseract_data(
            0,
            {
                "text": ["MEDICAL", "SCREENING", "", "AND", "DECLARATION"],
                "conf": ["93", "95", "-1", "95", "96"],
                "block_num": [1, 1, 1, 1, 1],
                "par_num": [1, 1, 1, 2, 2],
                "line_num": [1, 1, 1, 1, 1],
            },
        )
        self.assertEqual(
            [(line.page, line.text, line.confidence) for line in parsed],
            [(0, "MEDICAL SCREENING", 94.0), (0, "AND DECLARATION", 95.5)],
        )

    def test_parse_tesseract_data_falls_back_when_line_metadata_is_invalid(self) -> None:
        parsed = _parse_tesseract_data(
            0,
            {
                "text": ["MEDICAL", "SCREENING"],
                "conf": ["93", "95"],
                "block_num": ["x", "x"],
                "par_num": [1, 1],
                "line_num": [1, 1],
            },
        )
        self.assertEqual(
            [(line.page, line.text, line.confidence) for line in parsed],
            [(0, "MEDICAL", 93.0), (0, "SCREENING", 95.0)],
        )

    def test_build_coherent_output_merges_field_values_across_lines(self) -> None:
        coherent_lines, fields = _build_coherent_output(
            [
                OCRLine(page=0, text="Name: Jane", confidence=95.0),
                OCRLine(page=0, text="Doe", confidence=95.0),
                OCRLine(page=0, text="Registration ID: 1234", confidence=95.0),
                OCRLine(page=0, text="Gender: Female", confidence=95.0),
            ]
        )
        self.assertEqual(
            coherent_lines,
            ["Name: Jane Doe", "Registration ID: 1234", "Gender: Female"],
        )
        self.assertEqual(
            fields,
            {"Name": "Jane Doe", "Registration ID": "1234", "Gender": "Female"},
        )

    def test_build_coherent_output_limits_field_spillover(self) -> None:
        coherent_lines, fields = _build_coherent_output(
            [
                OCRLine(page=0, text="Mode of Study: Full-Time", confidence=95.0),
                OCRLine(page=0, text="( ) Part-Time", confidence=95.0),
                OCRLine(page=0, text="Course Registration", confidence=95.0),
            ]
        )
        self.assertEqual(
            coherent_lines,
            [
                "Mode of Study: Full-Time ( ) Part-Time",
                "Course Registration",
            ],
        )
        self.assertEqual(fields, {"Mode of Study": "Full-Time ( ) Part-Time"})

    def test_run_ocr_includes_coherent_lines_and_fields(self) -> None:
        fake_lines = [
            OCRLine(page=0, text="Name: Alice", confidence=98.0),
            OCRLine(page=0, text="Smith", confidence=98.0),
        ]
        with (
            patch("main.os.path.exists", return_value=True),
            patch("main._open_tiff_pages", return_value=["fake-page"]),
            patch.dict(
                main.ENGINE_RUNNERS,
                {"tesseract": lambda _pages: main._build_engine_payload(fake_lines)},
                clear=False,
            ),
        ):
            payload = main.run_ocr("/tmp/input.tiff", ["tesseract"])
        result = payload["engines"]["tesseract"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coherent_lines"], ["Name: Alice Smith"])
        self.assertEqual(result["fields"], {"Name": "Alice Smith"})

    def test_parse_paddle_result_extracts_text_and_confidence(self) -> None:
        parsed = _parse_paddle_result(
            0,
            [
                [
                    [[[0, 0], [1, 1]], ("alpha", 0.91)],
                    [[[0, 0], [1, 1]], ("beta", 0.72)],
                ]
            ],
        )
        self.assertEqual(
            [(line.text, line.confidence) for line in parsed],
            [("alpha", 0.91), ("beta", 0.72)],
        )

    def test_parse_easyocr_result_extracts_text_and_confidence(self) -> None:
        parsed = _parse_easyocr_result(
            2,
            [
                ([[0, 0], [1, 1]], "line 1", 0.95),
                ([[0, 0], [1, 1]], "line 2", 0.75),
            ],
        )
        self.assertEqual(
            [(line.page, line.text, line.confidence) for line in parsed],
            [(2, "line 1", 0.95), (2, "line 2", 0.75)],
        )

    def test_run_ocr_reports_install_hint_on_import_error(self) -> None:
        with (
            patch("main.os.path.exists", return_value=True),
            patch("main._open_tiff_pages", return_value=["fake-page"]),
            patch.dict(
                main.ENGINE_RUNNERS,
                {"tesseract": self._raise_import_error},
                clear=False,
            ),
        ):
            payload = main.run_ocr("/tmp/input.tiff", ["tesseract"])
        result = payload["engines"]["tesseract"]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["engine"], "tesseract")
        self.assertIn("Install with", result["hint"])
        self.assertIn("pip install pytesseract", result["hint"])
        self.assertIn("system binary", result["hint"])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import main
from main import (
    OCRLine,
    _average_confidence,
    _parse_easyocr_result,
    _parse_paddle_result,
    _parse_tesseract_data,
    _run_textract,
    _run_google_docai,
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


class TextractEngineTests(unittest.TestCase):
    """Unit tests for the Amazon Textract engine runner."""

    def _make_fake_page(self) -> MagicMock:
        page = MagicMock()
        # save() must accept keyword args; no return value needed
        page.save = MagicMock()
        return page

    def test_run_textract_extracts_lines_and_normalises_confidence(self) -> None:
        fake_response = {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Hello world", "Confidence": 99.5},
                {"BlockType": "WORD", "Text": "Hello", "Confidence": 99.0},
                {"BlockType": "LINE", "Text": "Second line", "Confidence": 80.0},
                {"BlockType": "LINE", "Text": "", "Confidence": 50.0},  # empty, skip
            ]
        }
        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = fake_response

        with (
            patch("main._get_textract_client", return_value=mock_client),
            patch("main.io.BytesIO"),
        ):
            result = _run_textract([self._make_fake_page()])

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["line_count"], 2)
        lines = result["lines"]
        self.assertEqual(lines[0]["text"], "Hello world")
        self.assertAlmostEqual(lines[0]["confidence"], round(99.5 / 100.0, 4))
        self.assertEqual(lines[1]["text"], "Second line")
        self.assertAlmostEqual(lines[1]["confidence"], round(80.0 / 100.0, 4))

    def test_run_textract_assigns_correct_page_indices(self) -> None:
        def fake_detect(Document):  # noqa: N803
            return {
                "Blocks": [{"BlockType": "LINE", "Text": "Page text", "Confidence": 90.0}]
            }

        mock_client = MagicMock()
        mock_client.detect_document_text.side_effect = fake_detect

        with (
            patch("main._get_textract_client", return_value=mock_client),
            patch("main.io.BytesIO"),
        ):
            result = _run_textract(
                [self._make_fake_page(), self._make_fake_page()]
            )

        self.assertEqual(result["line_count"], 2)
        self.assertEqual(result["lines"][0]["page"], 0)
        self.assertEqual(result["lines"][1]["page"], 1)

    def test_run_textract_missing_confidence_is_none(self) -> None:
        fake_response = {
            "Blocks": [{"BlockType": "LINE", "Text": "No conf"}]
        }
        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = fake_response

        with (
            patch("main._get_textract_client", return_value=mock_client),
            patch("main.io.BytesIO"),
        ):
            result = _run_textract([self._make_fake_page()])

        self.assertIsNone(result["lines"][0]["confidence"])

    def test_run_ocr_reports_install_hint_for_textract(self) -> None:
        def _raise(_pages: object) -> None:
            raise ImportError("No module named 'boto3'")

        with (
            patch("main.os.path.exists", return_value=True),
            patch("main._open_tiff_pages", return_value=["fake-page"]),
            patch.dict(main.ENGINE_RUNNERS, {"textract": _raise}, clear=False),
        ):
            payload = main.run_ocr("/tmp/input.tiff", ["textract"])

        result = payload["engines"]["textract"]
        self.assertEqual(result["status"], "error")
        self.assertIn("boto3", result["hint"])


class GoogleDocAIEngineTests(unittest.TestCase):
    """Unit tests for the Google Document AI engine runner."""

    def _make_fake_page(self) -> MagicMock:
        page = MagicMock()
        page.save = MagicMock()
        return page

    def _make_docai_document(self, lines_data: list[tuple[str, float]]) -> MagicMock:
        """Build a minimal mock of a ``google.cloud.documentai_v1.Document``."""
        full_text = "\n".join(text for text, _ in lines_data)
        doc = MagicMock()
        doc.text = full_text

        doc_lines = []
        offset = 0
        for text, conf in lines_data:
            seg = MagicMock()
            seg.start_index = offset
            seg.end_index = offset + len(text)
            offset += len(text) + 1  # +1 for \n

            layout = MagicMock()
            layout.text_anchor.text_segments = [seg]
            layout.confidence = conf

            line = MagicMock()
            line.layout = layout
            doc_lines.append(line)

        doc_page = MagicMock()
        doc_page.lines = doc_lines

        doc.pages = [doc_page]
        return doc

    def test_run_google_docai_extracts_lines_and_confidence(self) -> None:
        mock_doc = self._make_docai_document(
            [("Invoice total", 0.98), ("Date: 2024-01-01", 0.95)]
        )
        mock_result = MagicMock()
        mock_result.document = mock_doc

        mock_client = MagicMock()
        mock_client.process_document.return_value = mock_result
        mock_client.processor_path.return_value = "projects/p/locations/us/processors/x"

        mock_docai_module = MagicMock()
        mock_docai_module.DocumentProcessorServiceClient.return_value = mock_client
        mock_docai_module.RawDocument = MagicMock()
        mock_docai_module.ProcessRequest = MagicMock()

        env = {
            "GOOGLE_DOCAI_PROJECT_ID": "my-project",
            "GOOGLE_DOCAI_PROCESSOR_ID": "my-processor",
            "GOOGLE_DOCAI_LOCATION": "us",
        }

        # Stub the full google.cloud.documentai_v1 import chain.
        mock_google = MagicMock()
        mock_google.cloud.documentai_v1 = mock_docai_module

        with (
            patch.dict("os.environ", env),
            patch("main.io.BytesIO"),
            patch.dict(
                "sys.modules",
                {
                    "google": mock_google,
                    "google.cloud": mock_google.cloud,
                    "google.cloud.documentai_v1": mock_docai_module,
                },
            ),
        ):
            result = _run_google_docai([self._make_fake_page()])

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["line_count"], 2)
        self.assertEqual(result["lines"][0]["text"], "Invoice total")
        self.assertAlmostEqual(result["lines"][0]["confidence"], 0.98)
        self.assertEqual(result["lines"][1]["text"], "Date: 2024-01-01")

    def test_run_google_docai_raises_on_missing_env_vars(self) -> None:
        env = {"GOOGLE_DOCAI_PROJECT_ID": "", "GOOGLE_DOCAI_PROCESSOR_ID": ""}

        mock_docai_module = MagicMock()

        with (
            patch.dict("os.environ", env),
            patch.dict("sys.modules", {"google.cloud.documentai_v1": mock_docai_module}),
        ):
            with self.assertRaises(EnvironmentError) as ctx:
                _run_google_docai([self._make_fake_page()])

        self.assertIn("GOOGLE_DOCAI_PROJECT_ID", str(ctx.exception))

    def test_run_ocr_reports_env_error_for_google_docai(self) -> None:
        def _raise(_pages: object) -> None:
            raise EnvironmentError("GOOGLE_DOCAI_PROJECT_ID not set")

        with (
            patch("main.os.path.exists", return_value=True),
            patch("main._open_tiff_pages", return_value=["fake-page"]),
            patch.dict(main.ENGINE_RUNNERS, {"google_docai": _raise}, clear=False),
        ):
            payload = main.run_ocr("/tmp/input.tiff", ["google_docai"])

        result = payload["engines"]["google_docai"]
        self.assertEqual(result["status"], "error")
        self.assertIn("environment variables", result["hint"])

    def test_run_ocr_reports_install_hint_for_google_docai(self) -> None:
        def _raise(_pages: object) -> None:
            raise ImportError("No module named 'google.cloud.documentai_v1'")

        with (
            patch("main.os.path.exists", return_value=True),
            patch("main._open_tiff_pages", return_value=["fake-page"]),
            patch.dict(main.ENGINE_RUNNERS, {"google_docai": _raise}, clear=False),
        ):
            payload = main.run_ocr("/tmp/input.tiff", ["google_docai"])

        result = payload["engines"]["google_docai"]
        self.assertEqual(result["status"], "error")
        self.assertIn("google-cloud-documentai", result["hint"])


if __name__ == "__main__":
    unittest.main()

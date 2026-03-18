import unittest

from main import (
    OCRLine,
    _average_confidence,
    _parse_easyocr_result,
    _parse_paddle_result,
    _parse_tesseract_data,
)


class OCRParsingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

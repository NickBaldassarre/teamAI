from __future__ import annotations

import unittest

from teamai.json_utils import JsonExtractionError, extract_json_object


class JsonUtilsTest(unittest.TestCase):
    def test_extracts_plain_json(self) -> None:
        parsed = extract_json_object('{"done": true, "summary": "ok"}')
        self.assertTrue(parsed["done"])
        self.assertEqual(parsed["summary"], "ok")

    def test_extracts_fenced_json(self) -> None:
        parsed = extract_json_object(
            "Here you go\n```json\n{\"summary\": \"next\", \"should_stop\": false}\n```"
        )
        self.assertEqual(parsed["summary"], "next")

    def test_extracts_json_embedded_in_prose(self) -> None:
        parsed = extract_json_object(
            'I will use this object next: {"summary": "inspect", "should_stop": false} End.'
        )
        self.assertEqual(parsed["summary"], "inspect")

    def test_raises_when_missing_json(self) -> None:
        with self.assertRaises(JsonExtractionError):
            extract_json_object("not json at all")


if __name__ == "__main__":
    unittest.main()

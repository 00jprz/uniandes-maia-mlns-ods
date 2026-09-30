import json
import unittest
from pathlib import Path

from src.ModelController import ModelController


class ModelControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = ModelController()
        metadata_path = Path("resources/models/ods_text_lsa_classifier.metadata.json")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def test_reproduces_notebook_verification_predictions(self):
        verification = self.metadata["verification"]
        predictions = self.controller.predict_many(verification["new_texts"])
        self.assertEqual(predictions, verification["predictions"])

    def test_returns_original_integer_labels(self):
        prediction = self.controller.predict(
            "La energía solar permite ampliar el acceso a electricidad limpia."
        )
        self.assertIsInstance(prediction, int)
        self.assertIn(prediction, self.controller.get_categories())
        self.assertEqual(self.controller.get_categories(), list(range(1, 17)))

    def test_rejects_empty_or_invalid_inputs(self):
        invalid_inputs = ["", "   ", None, 123, [], ["texto válido", None]]
        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.controller.predict_many(value)


if __name__ == "__main__":
    unittest.main()

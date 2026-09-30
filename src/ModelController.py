import hashlib
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

import Definitions
from src.DataPreprocessing import DataPreprocessing


class ModelController:
    """Carga el pipeline ODS y expone inferencia para uno o varios textos."""

    MODEL_FILENAME = "ods_text_lsa_classifier.joblib"
    METADATA_FILENAME = "ods_text_lsa_classifier.metadata.json"
    REQUIRED_STEPS = ["tfidf", "dimred", "model"]

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or Path(Definitions.ROOT_DIR) / "resources" / "models")
        self.model_path = self.model_dir / self.MODEL_FILENAME
        self.metadata_path = self.model_dir / self.METADATA_FILENAME
        self.d_processing = DataPreprocessing()

        self.metadata = self._load_metadata()
        self._validate_artifact_hash()
        self.model = self._load_model()
        self._validate_model_contract()

    def _load_metadata(self):
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"No se encontraron los metadatos del modelo: {self.metadata_path}")

        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"No fue posible leer los metadatos: {self.metadata_path}") from exc

    def _validate_artifact_hash(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo ODS: {self.model_path}")

        expected_hash = self.metadata.get("artifact", {}).get("sha256")
        if not expected_hash:
            raise RuntimeError("Los metadatos no contienen el hash SHA-256 del modelo.")

        digest = hashlib.sha256()
        with self.model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != expected_hash:
            raise RuntimeError("El hash del modelo no coincide con los metadatos de exportación.")

    def _load_model(self):
        try:
            return joblib.load(self.model_path)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "No fue posible importar el preprocesamiento requerido por el modelo. "
                "Comprueba ods_text_preprocessing.py y las dependencias instaladas."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"No fue posible cargar el modelo: {self.model_path}") from exc

    def _validate_model_contract(self):
        if not isinstance(self.model, Pipeline):
            raise TypeError("El artefacto ODS debe contener un sklearn.pipeline.Pipeline.")

        actual_steps = [name for name, _ in self.model.steps]
        if actual_steps != self.REQUIRED_STEPS:
            raise RuntimeError(
                f"Pasos inesperados en el pipeline: {actual_steps}; "
                f"se esperaban {self.REQUIRED_STEPS}."
            )

        expected_labels = self.metadata.get("labels", {}).get("original_values")
        actual_labels = [self._to_builtin(value) for value in self.model.named_steps["model"].classes_]
        if actual_labels != expected_labels:
            raise RuntimeError(
                f"Las etiquetas del modelo {actual_labels} no coinciden con los metadatos "
                f"{expected_labels}."
            )

    @staticmethod
    def _to_builtin(value):
        return value.item() if hasattr(value, "item") else value

    def get_categories(self):
        return list(self.metadata["labels"]["original_values"])

    def predict_many(self, texts):
        prepared_texts = self.d_processing.transform(texts)
        predictions = self.model.predict(prepared_texts)
        return [self._to_builtin(value) for value in predictions]

    def predict(self, text):
        return self.predict_many(text)[0]

    def get_word_weights(self, text):
        """Pesos TF-IDF de las palabras del texto presentes en el vocabulario."""
        prepared_texts = self.d_processing.transform(text)
        vectorizer = self.model.named_steps["tfidf"]
        vector = vectorizer.transform(prepared_texts).getrow(0)
        terms = vectorizer.get_feature_names_out()
        return {
            str(terms[index]): float(weight)
            for index, weight in zip(vector.indices, vector.data)
            if weight > 0
        }

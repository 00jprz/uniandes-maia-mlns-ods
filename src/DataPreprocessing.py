from collections.abc import Sequence


class DataPreprocessing:
    """Valida el contrato de entrada sin repetir la limpieza del pipeline."""

    def transform(self, data):
        """Devuelve una lista de textos crudos, no vacíos, lista para inferencia."""
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)):
            texts = list(data)
        else:
            raise TypeError("La entrada debe ser un texto o una secuencia de textos.")

        if not texts:
            raise ValueError("Se requiere al menos un texto para realizar la predicción.")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"El elemento en la posición {index} debe ser texto.")
            if not text.strip():
                raise ValueError(f"El texto en la posición {index} está vacío.")

        return texts

"""Preprocesamiento reutilizable para el clasificador de textos ODS.

La lista de stopwords se inyecta desde el pipeline exportado para conservar
exactamente la utilizada durante el entrenamiento del notebook.
"""

from nltk import RegexpTokenizer


_TOKENIZER = RegexpTokenizer(r"\w+")


def preprocess_text(text, *, stopwords_es):
    """Replica la limpieza del notebook para un texto de entrada."""
    tokens = _TOKENIZER.tokenize(str(text).lower())
    tokens = [word for word in tokens if word not in stopwords_es]
    tokens = [word for word in tokens if len(word) > 2]
    return " ".join(tokens)

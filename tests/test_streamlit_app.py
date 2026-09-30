import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTest(unittest.TestCase):
    APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    def test_predicts_from_raw_text(self):
        app = AppTest.from_file(self.APP_PATH, default_timeout=60).run()
        self.assertFalse(app.exception)

        app.text_area[0].input(
            "La educación inclusiva y de calidad amplía las oportunidades de aprendizaje."
        )
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertEqual(
            app.success[0].value,
            "El número del Objetivo de Desarrollo Sostenible (ODS) predicho es: 4",
        )
        self.assertEqual(app.subheader[0].value, "Nube de palabras · ODS 4")
        self.assertEqual(len(app.get("image")), 1)

        app.text_area[0].input("El acceso al agua potable mejora la salud de las comunidades.")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.subheader[0].value, "Nube de palabras · ODS 6")
        self.assertEqual(len(app.get("image")), 1)

    def test_text_without_known_words_does_not_crash(self):
        app = AppTest.from_file(self.APP_PATH, default_timeout=60).run()
        app.text_area[0].input("de la y qzxqzxqzx")
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertIn("No hay palabras reconocidas", app.info[0].value)
        self.assertEqual(len(app.get("image")), 0)

    def test_reports_empty_text(self):
        app = AppTest.from_file(self.APP_PATH, default_timeout=60).run()
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.error[0].value, "El texto en la posición 0 está vacío.")
        self.assertEqual(len(app.get("image")), 0)


if __name__ == "__main__":
    unittest.main()

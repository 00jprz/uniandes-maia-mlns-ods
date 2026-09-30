import streamlit as st
from wordcloud import WordCloud

from src.ModelController import ModelController


st.set_page_config(
    layout="centered",
    page_title="Clasificador de textos dentro de los 17 objetivos de desarrollo sostenible (ODS)",
    page_icon="🌍",
)


@st.cache_resource
def load_controller():
    return ModelController()


st.title("Clasificador de textos dentro de los 17 objetivos de desarrollo sostenible (ODS)")
st.write(
    "Ingresa un texto en español. El modelo devolverá la etiqueta del numero del objetivos de desarrollo sostenible (ODS)"
    "aprendida durante el entrenamiento."
)

try:
    ctrl = load_controller()
except Exception as exc:
    st.error(f"No fue posible cargar el modelo: {exc}")
    st.stop()

with st.form(key="prediction_form"):
    input_text = st.text_area(
        "Texto para clasificar",
        height=180,
        placeholder="Ejemplo: El acceso al agua potable mejora la salud de las comunidades.",
    )
    submit_button = st.form_submit_button(label="Predecir ODS")

if submit_button:
    try:
        prediction = ctrl.predict(input_text)
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.success(f"El número del Objetivo de Desarrollo Sostenible (ODS) predicho es: {prediction}")
        st.subheader(f"Nube de palabras · ODS {prediction}")
        word_weights = ctrl.get_word_weights(input_text)
        if word_weights:
            cloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                colormap="viridis",
                max_words=80,
                prefer_horizontal=0.9,
                random_state=42,
                relative_scaling=0.5,
            ).generate_from_frequencies(word_weights)
            st.image(
                cloud.to_array(),
                caption=f"Palabras del texto clasificado como ODS {prediction}",
                width="stretch",
            )
            st.caption(
                "El tamaño refleja el peso TF-IDF de cada palabra en el texto ingresado. "
                "La nube resume los términos reconocidos por el modelo; "
                "no indica su contribución a la predicción ni la confianza del resultado."
            )
        else:
            st.info(
                "No hay palabras reconocidas por el modelo para generar la nube. "
                "Prueba con un texto más descriptivo sobre desarrollo sostenible."
            )
st.caption(f"""
    <div style="
        background-color: #E3F2FD;
        color: #243746;
        text-align: center;
        padding: 16px;
        border-radius: 10px;
        line-height: 1.8;
    ">
        <em>JAVIER PEREZ OSORIO</em><br>
        <em>Versión de: Septiembre 19, 2026</em><br>
        <em>Universidad de los Andes -</em>
        Microproyecto de Inteligencia Artificial
    </div>
    """,
    unsafe_allow_html=True)

import streamlit as st
from transformers import pipeline
import scipy.io.wavfile
import tempfile

st.set_page_config(page_title="Texto para Áudio", page_icon="🎙️")

st.title("Texto → Áudio")
st.markdown("---")

# carregar modelo uma única vez
@st.cache_resource
def load_model():
    return pipeline("text-to-speech", model="suno/bark-small")

pipe = load_model()

# entrada de texto
texto = st.text_area(
    "Digite o texto:",
    "Olá, este é um teste de áudio Laurindo"
)

if st.button("Gerar Áudio"):

    if texto.strip() == "":
        st.warning("Digite algum texto!")
    else:
        with st.spinner("Gerando áudio..."):
            
            audio = pipe(texto)

            # salvar em arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                scipy.io.wavfile.write(
                    tmp.name,
                    rate=audio["sampling_rate"],
                    data=audio["audio"]
                )
                temp_path = tmp.name

        st.success("Áudio gerado!")

        # player no Streamlit
        st.audio(temp_path)

        # botão download
        with open(temp_path, "rb") as f:
            st.download_button(
                label="Baixar áudio",
                data=f,
                file_name="audio.wav",
                mime="audio/wav"
            )
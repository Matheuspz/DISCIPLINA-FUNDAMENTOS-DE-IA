import streamlit as st
from PyPDF2 import PdfReader
import tempfile
import os
import re
from gtts import gTTS
from datetime import datetime

# ==================== GROQ + LANGCHAIN ====================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="EduTec Pro", layout="wide")
st.title("📄 EduTec Pro!")
st.subheader("Resuma e Ousa seus PDFs aqui!")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🔑 Groq API Key")
    groq_api_key = st.text_input("Cole sua Groq API Key:", type="password")

    model_choice = st.selectbox(
        "Escolha o Modelo",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index=0
    )

# ====================== UPLOAD ======================
uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extraindo texto..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        raw_text = "".join([page.extract_text() or "" for page in reader.pages])
        os.unlink(tmp_path)

        text = re.sub(r'\n+', ' ', raw_text)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()

    st.success(f"✅ Texto extraído! ({len(text):,} caracteres)")

    col1, col2 = st.columns(2)
    groq_btn = col1.button("🤖 Gerar Resumo com Groq", type="primary", use_container_width=True)
    audio_btn = col2.button("🔊 Gerar Áudio", use_container_width=True)

    # ====================== RESUMO COM GROQ ======================
    if groq_btn:
        if not groq_api_key:
            st.error("❌ Insira sua Groq API Key na barra lateral.")
        else:
            with st.spinner("🤖 Gerando resumo com Groq..."):
                try:
                    llm = ChatGroq(
                        api_key=groq_api_key,
                        model_name=model_choice,
                        temperature=0.3,
                        max_tokens=2048
                    )

                    prompt_template = ChatPromptTemplate.from_template("""
Você é um especialista em resumir documentos em português brasileiro.
Crie um resumo claro, profissional e bem estruturado contendo:
- Título principal
- Introdução curta
- Principais pontos em bullet points
- Conclusão

Mantenha números e dados importantes.

Texto:
{text}

Resumo:
""")

                    chain = prompt_template | llm
                    response = chain.invoke({"text": text[:18000]})
                    summary = response.content

                    st.subheader("📝 Resumo Gerado pela Groq")
                    st.markdown(summary)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

                    # Download TXT
                    st.download_button(
                        "⬇️ Baixar Resumo (TXT)",
                        summary,
                        f"resumo_groq_{timestamp}.txt",
                        "text/plain"
                    )

                    # ====================== PDF MELHORADO ======================
                    try:
                        from fpdf import FPDF


                        class PDF(FPDF):
                            def header(self):
                                self.set_font('Arial', 'B', 14)
                                self.cell(0, 10, 'Resumo Gerado por Groq', 0, 1, 'C')
                                self.ln(5)


                        pdf = PDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)

                        # Encode seguro para português
                        safe_text = summary.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 8, safe_text)

                        pdf_name = f"resumo_groq_{timestamp}.pdf"
                        pdf.output(pdf_name)

                        with open(pdf_name, "rb") as f:
                            pdf_bytes = f.read()

                        st.download_button(
                            "⬇️ Baixar Resumo como PDF",
                            pdf_bytes,
                            pdf_name,
                            "application/pdf"
                        )
                        os.remove(pdf_name)
                    except Exception as pdf_error:
                        st.warning(f"PDF não gerado: {pdf_error}")

                except Exception as e:
                    st.error(f"Erro com Groq: {e}")

    # ====================== ÁUDIO ======================
    if audio_btn:
        with st.spinner("Gerando áudio..."):
            try:
                text_audio = text[:3500]
                tts = gTTS(text=text_audio, lang='pt', slow=False)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    audio_path = tmp_audio.name

                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                st.subheader("🔊 Áudio Gerado")
                st.audio(audio_bytes, format="audio/mp3")

                st.download_button(
                    "⬇️ Baixar Áudio (MP3)",
                    audio_bytes,
                    "pdf_audio.mp3",
                    "audio/mp3"
                )
                os.unlink(audio_path)
            except Exception as e:
                st.error(f"Erro no áudio: {e}")

    with st.expander("Ver Texto Extraído"):
        st.text_area("", text, height=400)

else:
    st.info("Faça upload de um PDF para começar.")
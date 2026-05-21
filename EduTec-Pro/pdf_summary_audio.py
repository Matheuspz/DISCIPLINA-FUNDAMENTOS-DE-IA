import os
import tempfile

import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from fpdf import FPDF
from gtts import gTTS

st.set_page_config(page_title="Resumo e Áudio de PDF", page_icon="📚")
st.title("Resumo de PDF + Áudio (MP3)")
st.markdown("""Faça upload de um arquivo PDF. O app extrai o texto, gera um resumo, cria um PDF resumido e um arquivo MP3 com a leitura do resumo.""")

@st.cache_resource
def load_summarizer():
    # Modelo leve para sumarização
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()


def extract_text_from_pdf(file_path):
    text_parts = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        # try to cut at newline or space
        if end < length:
            sep = text.rfind("\n", start, end)
            if sep == -1:
                sep = text.rfind(" ", start, end)
            if sep > start:
                end = sep
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def summarize_text(text):
    chunks = chunk_text(text, max_chars=1000)
    if not chunks:
        return ""
    summaries = []
    for chunk in chunks:
        try:
            s = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(s[0]["summary_text"]) 
        except Exception:
            # if summarizer fails on chunk, keep chunk as-is truncated
            summaries.append(chunk[:500])
    combined = "\n\n".join(summaries)
    # final pass to compress combined summaries if it's still long
    if len(combined) > 800:
        try:
            final = summarizer(combined, max_length=180, min_length=60, do_sample=False)
            return final[0]["summary_text"]
        except Exception:
            return combined
    else:
        return combined


def create_pdf_from_text(text, out_path, title="Resumo"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, title + "\n\n")
    pdf.multi_cell(0, 6, text)
    pdf.output(out_path)


uploaded = st.file_uploader("Envie um arquivo PDF", type=["pdf"]) 

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded.read())
        tmp_pdf_path = tmp_pdf.name

    st.info("Extraindo texto do PDF...")
    text = extract_text_from_pdf(tmp_pdf_path)

    if not text.strip():
        st.warning("Não foi possível extrair texto deste PDF.")
    else:
        st.info("Gerando resumo (pode demorar alguns segundos)...")
        summary = summarize_text(text)
        st.success("Resumo gerado")

        st.subheader("Resumo")
        st.write(summary)

        # Create summarized PDF
        base_name = os.path.splitext(uploaded.name)[0]
        summary_pdf_name = f"{base_name}_resumo.pdf"
        summary_mp3_name = f"{base_name}_resumo.mp3"

        tmp_dir = tempfile.gettempdir()
        summary_pdf_path = os.path.join(tmp_dir, summary_pdf_name)
        summary_mp3_path = os.path.join(tmp_dir, summary_mp3_name)

        create_pdf_from_text(summary, summary_pdf_path, title=f"Resumo de {uploaded.name}")

        # Generate MP3 using gTTS
        try:
            tts = gTTS(summary, lang="pt")
            tts.save(summary_mp3_path)
        except Exception as e:
            st.error(f"Erro ao gerar áudio: {e}")
            summary_mp3_path = None

        # Provide downloads and player
        with open(summary_pdf_path, "rb") as fpdf:
            st.download_button("Baixar PDF resumido", data=fpdf, file_name=summary_pdf_name, mime="application/pdf")

        if summary_mp3_path and os.path.exists(summary_mp3_path):
            st.audio(summary_mp3_path)
            with open(summary_mp3_path, "rb") as fmp3:
                st.download_button("Baixar MP3 do resumo", data=fmp3, file_name=summary_mp3_name, mime="audio/mpeg")

        # cleanup temp original
        try:
            os.remove(tmp_pdf_path)
        except Exception:
            pass

else:
    st.info("Envie um PDF para começar.")

# Footer
st.markdown("---")
st.write("Feito com ❤️ — EduTec-Pro")

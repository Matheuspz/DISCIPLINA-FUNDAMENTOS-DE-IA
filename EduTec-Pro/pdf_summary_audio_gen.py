import os
import tempfile
import io

import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, logging as hf_logging
hf_logging.set_verbosity_error()
import scipy.io.wavfile as wavfile

# Optional: pydub for WAV->MP3 conversion (requires ffmpeg installed)
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False

st.set_page_config(page_title="Resumo Generativo de PDF + Áudio", page_icon="🤖📚")
st.title("Resumo Generativo de PDF + Áudio (MP3)")
st.markdown("""Upload de um PDF, sumarização usando modelos generativos e síntese de voz por modelo neural.

Notas:
- Modelos são carregados localmente via Hugging Face Transformers. Podem exigir download e GPU.
- Para gerar MP3 a partir de WAV, instalar ffmpeg (pydub usa ffmpeg).
""")

@st.cache_resource
def load_models():
    import torch
    # device index for pipeline: 0 for GPU, -1 for CPU
    device = 0 if torch.cuda.is_available() else -1

    # Use text-generation since 'summarization' task isn't available in this transformers build
    # Flan-T5 small is a good tradeoff for local inference
    summarizer = pipeline("text-generation", model="google/flan-t5-small", device=device)

    # Text-to-speech model (as used in Aula6)
    tts = pipeline("text-to-speech", model="suno/bark-small", device=device)

    return summarizer, tts

# Load on demand
with st.spinner("Carregando modelos (pode demorar)..."):
    try:
        summarizer, tts = load_models()
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        st.stop()


def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            sep = text.rfind("\n", start, end)
            if sep == -1:
                sep = text.rfind(" ", start, end)
            if sep > start:
                end = sep
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def generative_summarize(text):
    # Try generative summarization via the loaded pipeline (instruction prompt).
    # If generation fails or simply echoes the prompt, fallback to an extractive method.
    import re
    from collections import Counter

    def extractive_summary(src_text, max_sentences=3):
        # Simple frequency-based extractive summarizer (works offline, lightweight)
        stopwords = set([
            # small combined Portuguese and English stopwords
            'de','a','o','que','e','do','da','em','um','para','é','com','não','uma','os','no','se','na','por','mais','as','dos','como','mas','foi','ao','ele','das','tem','à','seu','sua','ou','ser','quando','muito','há','nos','já','está','eu','também','só','pelo','pela','até','isso','entre','era','depois','sem','mesmo','aos','ter','seus','quem','nas','me','essa','num','nem','suas','meu','às','minha','têm','numa','pelos','elas','havia','isto','eles','estavam','você','vos','o','la',
            'the','and','is','in','it','of','to','a','that','with','as','for','its','on','by','an','be','this','are'
        ])
        sentences = re.split(r'(?<=[.!?])\s+', src_text.strip())
        if len(sentences) <= max_sentences:
            return src_text.strip()
        words = re.findall(r"\w+", src_text.lower())
        words = [w for w in words if w not in stopwords]
        if not words:
            return ' '.join(sentences[:max_sentences])
        freq = Counter(words)
        maxf = max(freq.values())
        for k in freq:
            freq[k] = freq[k] / maxf
        sent_scores = []
        for i, s in enumerate(sentences):
            s_words = re.findall(r"\w+", s.lower())
            score = sum(freq.get(w, 0) for w in s_words)
            sent_scores.append((i, score, s))
        top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sorted = sorted(top, key=lambda x: x[0])
        return ' '.join([t[2].strip() for t in top_sorted])

    chunks = chunk_text(text, max_chars=1000)
    partials = []
    for chunk in chunks:
        prompt = f"Resuma o texto a seguir em português de forma clara e concisa:\n\n{chunk}\n\nResumo:"
        try:
            out = summarizer(prompt, max_new_tokens=150, do_sample=False, temperature=0.0)
            gen = out[0].get("generated_text") or out[0].get("text") or ""
            partials.append(gen.strip())
        except Exception:
            partials.append('')

    combined = "\n\n".join([p for p in partials if p])

    # Detect failure or prompt-echoing: too short or contains the original prompt markers
    fail_conditions = (
        not combined.strip(),
        len(combined.strip()) < 20,
        'resuma o texto' in combined.lower(),
        'text for summary' in combined.lower(),
    )

    if any(fail_conditions):
        # fallback to extractive summary and inform user via Streamlit
        try:
            st.warning('Geração falhou ou retornou prompt. Usando resumo extractivo como fallback.')
        except Exception:
            pass
        return extractive_summary(text, max_sentences=3)

    # final compression pass (try to summarize the generated partials into a concise text)
    if len(combined) > 800:
        prompt2 = f"Resuma o texto a seguir reforçando clareza e concisão:\n\n{combined}\n\nResumo:"
        try:
            out2 = summarizer(prompt2, max_new_tokens=200, do_sample=False, temperature=0.0)
            combined2 = out2[0].get("generated_text") or out2[0].get("text") or combined
            if combined2 and len(combined2.strip()) > 20:
                return combined2.strip()
        except Exception:
            pass

    return combined.strip()


def synthesize_tts_to_wav_bytes(text):
    # Use TTS pipeline (suno/bark-small) as in Aula6
    audio = tts(text)
    rate = audio.get("sampling_rate")
    data = audio.get("audio")
    # write to BytesIO as WAV
    bio = io.BytesIO()
    wavfile.write(bio, rate, data)
    bio.seek(0)
    return bio


uploaded = st.file_uploader("Envie um PDF", type=["pdf"]) 

if not uploaded:
    st.info("Envie um PDF para começar.")
    st.stop()

pdf_bytes = uploaded.read()
st.info("Extraindo texto do PDF...")
text = extract_text_from_pdf_bytes(pdf_bytes)
if not text or not text.strip():
    st.warning("Não foi possível extrair texto do PDF.")
    st.stop()

st.info("Gerando resumo com modelo generativo (pode demorar)...")
summary = generative_summarize(text)
st.success("Resumo gerado")

st.subheader("Resumo")
st.write(summary)

# Create summarized PDF
base = os.path.splitext(uploaded.name)[0]
summary_pdf_name = f"{base}_resumo_generativo.pdf"
summary_mp3_name = f"{base}_resumo_generativo.mp3"

# create simple PDF using reportlab-free approach to avoid heavy deps
try:
    from fpdf import FPDF
    def create_pdf(text, out_path, title=None):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        if title:
            pdf.multi_cell(0, 6, title + "\n\n")
        pdf.multi_cell(0, 6, text)
        pdf.output(out_path)
except Exception:
    create_pdf = None

# write summary PDF to temp and provide download
tmpdir = tempfile.gettempdir()
summary_pdf_path = os.path.join(tmpdir, summary_pdf_name)
if create_pdf:
    try:
        create_pdf(summary, summary_pdf_path, title=f"Resumo generativo de {uploaded.name}")
        with open(summary_pdf_path, "rb") as f:
            st.download_button("Baixar PDF resumido", data=f, file_name=summary_pdf_name, mime="application/pdf")
    except Exception as e:
        st.warning(f"Não foi possível criar PDF: {e}")
else:
    st.warning("FPDF não instalado — não será possível gerar PDF resumido. Instale 'fpdf'.")

# Synthesize TTS -> WAV in memory
st.info("Gerando áudio do resumo (modelo TTS generativo)...")
try:
    wav_bio = synthesize_tts_to_wav_bytes(summary)
    # Offer WAV player/download
    st.audio(wav_bio)
    # prepare downloads
    wav_bio.seek(0)
    if _HAS_PYDUB:
        try:
            # Convert to MP3 using pydub (requires ffmpeg)
            audio_seg = AudioSegment.from_file(wav_bio, format="wav")
            mp3_bio = io.BytesIO()
            audio_seg.export(mp3_bio, format="mp3")
            mp3_bio.seek(0)
            st.audio(mp3_bio)
            st.download_button("Baixar MP3 do resumo", data=mp3_bio, file_name=summary_mp3_name, mime="audio/mpeg")
        except Exception as e:
            st.warning(f"Conversão para MP3 falhou (ffmpeg pode estar ausente): {e}. Fornecendo WAV em vez de MP3.")
            wav_bio.seek(0)
            st.download_button("Baixar WAV do resumo", data=wav_bio, file_name=f"{base}_resumo.wav", mime="audio/wav")
    else:
        wav_bio.seek(0)
        st.warning("pydub não está disponível — fornecendo WAV. Para MP3 instale pydub e ffmpeg.")
        st.download_button("Baixar WAV do resumo", data=wav_bio, file_name=f"{base}_resumo.wav", mime="audio/wav")
except Exception as e:
    st.error(f"Erro durante síntese TTS: {e}")

st.markdown("---")
st.write("Feito com modelos generativos — Baseado em Aula6")

import streamlit as st
import torch
import asyncio
import edge_tts
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Multimodal Regional Translator", layout="wide")
st.title("Multimodal Context-Aware Regional Translator")
st.caption("Translate regional audio or text into contextual English with text and voice output.")

# Cache heavy models so they only load once
@st.cache_resource
def load_models():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return asr, tokenizer, model

with st.spinner("Initializing models..."):
    asr_pipe, trans_tokenizer, trans_model = load_models()

# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def synthesize_speech(text, output_file="output.mp3"):
    comm = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await comm.save(output_file)
    return output_file

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    audio_file = st.file_uploader("Upload Regional Audio (.wav / .mp3)", type=["wav", "mp3"])
    text_input = st.text_input("Or enter regional text directly:")
    dialect = st.selectbox("Select Dialect Variety", ["General", "North-Karnataka", "Mysore", "Slang / Colloquial"])
    submit_btn = st.button("Translate & Synthesize", type="primary")

with col2:
    st.subheader("Output")
    recognized_placeholder = st.empty()
    translated_placeholder = st.empty()
    audio_placeholder = st.empty()

if submit_btn:
    source_text = ""
    if audio_file is not None:
        with open("temp_input.wav", "wb") as f:
            f.write(audio_file.read())
        transcription = asr_pipe("temp_input.wav")
        source_text = transcription["text"].strip()
    elif text_input.strip():
        source_text = text_input.strip()

    if not source_text:
        st.warning("Please provide either an audio file or enter text.")
    else:
        # Context building
        context_str = " | ".join(st.session_state.chat_history[-2:]) if st.session_state.chat_history else "Start of dialogue"
        prompt = f"Translate to English preserving context and dialect meaning. Dialect: {dialect} | Context: {context_str} | Input: {source_text}"

        # Translation
        inputs = trans_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_tokens = trans_model.generate(**inputs, max_length=128)
        english_text = trans_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Audio synthesis
        audio_path = "output.mp3"
        asyncio.run(synthesize_speech(english_text, audio_path))

        # Update state
        st.session_state.chat_history.append(f"Input: {source_text} -> English: {english_text}")

        # Render outputs
        recognized_placeholder.text_area("Recognized Input:", value=source_text, height=70)
        translated_placeholder.success(f"**English Translation:** {english_text}")
        audio_placeholder.audio(audio_path, format="audio/mp3")

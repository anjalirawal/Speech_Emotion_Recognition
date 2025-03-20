import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import os
import tempfile
import torch
import torchaudio
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from speechbrain.inference.interfaces import foreign_class
import pydub
from pydub import AudioSegment

# Model DIR
# MODEL_DIR = "/Users/anjalirawal/Desktop/Desktop - Anjali’s MacBook Air/Northeastern/ALY6980/User Interface/emotion-recognition-wav2vec2-IEMOCAP"


#Loading the SpeechBrain model 
@st.cache_resource
def load_local_model():
    model = foreign_class(
        source="./local_model",  # Load from local directory instead of Hugging Face
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    return model

model = load_local_model()

# Function to process audio
def process_audio(audio_file):
    signal, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        signal = transform(signal)
    return signal

# ✅ Function to predict emotion
def predict_emotion(audio_file):
    with st.spinner("Analyzing audio..."):
        try:
            # Load and process the audio
            signal = process_audio(audio_file)

            # Save as temp file for prediction
            temp_file = "temp_resampled_file.wav"
            torchaudio.save(temp_file, signal, 16000)

            # Predict Emotion
            out_prob, score, index, text_lab = model.classify_file(temp_file)
            return f"{text_lab} (Confidence: {score.item():.4f})"
        except Exception as e:
            return f"Error: {str(e)}"

# Streamlit UI
st.title(" Emotion Recognition from Audio")

#st.write("Record your voice or upload a file to let the model predict your emotion!")

# # Functoin to record audio
# def audio_callback(frame: av.AudioFrame) -> np.ndarray:
#     """Converts audio frame to numpy array."""
#     audio = frame.to_ndarray()
#     return audio

# # Record Audio Section
# # ✅ Initialize session state for recording
# if "recorded_audio_path" not in st.session_state:
#     st.session_state.recorded_audio_path = None

# # ✅ Function to process and save recorded audio
# def save_audio(frames):
#     """Convert frames to WAV and save temporarily"""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
#         sound = AudioSegment(
#             data=frames[0].to_ndarray().tobytes(),
#             sample_width=2,  # 16-bit PCM
#             frame_rate=48000,
#             channels=1
#         )
#         sound.export(tmp_audio.name, format="wav")
#         st.session_state.recorded_audio_path = tmp_audio.name  # Store in session state

# # ✅ Record Audio Section
# st.subheader("🎤 Record Audio")

# webrtc_ctx = webrtc_streamer(
#     key="audio-recorder",
#     mode=WebRtcMode.SENDRECV,
#     audio_receiver_size=256,
#     media_stream_constraints={"audio": True, "video": False},
#     async_processing=True,
# )

# # ✅ Process and save audio when recording is detected
# if webrtc_ctx and webrtc_ctx.audio_receiver:
#     try:
#         audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#         if audio_frames:
#             save_audio(audio_frames)
#             st.success("✅ Recording saved!")
#             st.audio(st.session_state.recorded_audio_path, format="audio/wav")  # Play back the recorded audio
#     except Exception as e:
#         st.warning(f"❌ No audio received yet. Error: {e}")

# # ✅ Show "Detect Emotion" Button AFTER recording is done
# if st.session_state.recorded_audio_path:
#     if st.button("🔍 Detect Emotion"):
#         predicted_emotion = predict_emotion(st.session_state.recorded_audio_path)
#         st.success(f"🎭 Predicted Emotion: {predicted_emotion}")


# Upload Audio Section
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_file.getvalue())
        tmp_audio_path = tmp_audio.name

    st.audio(tmp_audio_path, format="audio/wav")

    if st.button("🔍 Detect Emotion for Uploaded File"):
        with st.spinner("Analyzing audio..."):
            predicted_emotion = predict_emotion(tmp_audio_path)
            st.success(f"Predicted Emotion: {predicted_emotion}")

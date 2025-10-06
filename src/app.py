import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
from streamlit_option_menu import option_menu

import torch
import tempfile

from utils import get_waveform, get_mfcc

from models.cnn import MobileNet
from models.wav2vecClassifier import Wav2vecClassifier
from st_audiorec import st_audiorec

from pydub import AudioSegment

import time
import os

@st.cache_resource
def load_model(name: str, path: str):
    state_d = torch.load(path,
                         weights_only=True)
    if name == "MobileNet":
        model = MobileNet(num_classes=2)
    elif name == "Wav2vec":
        model = Wav2vecClassifier(num_labels=2)
    else:
        assert "Invalid model name"
        st.stop()

    model.load_state_dict(state_d)
    return model

@st.cache_resource
def load_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        file_content = file.read()
    return file_content

selected = option_menu(None, ["Home", "Experiments"],
    icons=['house', 'file-earmark'],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#000000"},
        "icon": {"color": "red", "font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#ff5757"},
    }
)

if selected == "Home":
    st.title("Aphasia Detection from Audio")

    models_dict = {
        "MobileNet": os.path.join(BASE_DIR, "checkpoints", "mobilenet_chkp", "mobilenet_0.94_cpu.pt"),
        "Wav2vec": os.path.join(BASE_DIR, "checkpoints", "wav2vec_chkp", "wav2vec_0.75_0.9375.pt")
    }

    model_name = st.selectbox("Choose a model:", list(models_dict.keys()))

    model = load_model(model_name, models_dict[model_name])
    model.eval()

    status = st.radio("Select input style: ", ('Upload files (5 or less files)', 'Record your voice'))
    output_status = st.radio("Select mode: ", ('Display predictions for each audio', 'Display mean prediction'))

    rec = False
    if status == 'Upload files (5 or less files)':
        st.write("Upload an audio file (e.g., WAV) to detect signs of aphasia.")

        input_files = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"], accept_multiple_files=True)

    else:
        rec = True
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')
            rec = False
        input_files = [wav_audio_data]

    MAX_FILES = 5

    with torch.no_grad():
        if input_files:
            input_audio = []
            probs = []

            if len(input_files) > MAX_FILES:
                st.warning(f"You uploaded {len(input_files)} files. Please upload no more than {MAX_FILES}.")
            else:
                if (not rec) and st.button("Start Prediction"):
                    mean_time = 0
                    for ind, uploaded_file in enumerate(input_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            if status == 'Upload files (5 or less files)':
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            else:
                                try:
                                    tmp_file.write(uploaded_file)
                                    tmp_path = tmp_file.name

                                    audio = AudioSegment.from_file(tmp_path)
                                    audio = audio.set_channels(1)
                                    audio.export(tmp_path, format="wav")
                                except Exception as e:
                                    st.error(f"Error during voice recording: {e}")
                                    st.stop()
                        try:
                            if model_name == "MobileNet":
                                input_audio = get_mfcc(audio_path=tmp_path, sr=8_000, n_fft=512,
                                                       hop_length=256, win_length=512, n_mels=128,
                                                       n_mfcc=128)[None, ...]
                            elif model_name == "Wav2vec":
                                input_audio = get_waveform(audio_path=tmp_path, sr=8_000)
                            else:
                                assert "Invalid model name"
                                st.stop()
                        except Exception as e:
                            st.error(f"Error during preprocessing: {e}")
                            st.stop()
                        compute_time = 0
                        with st.spinner("Analyzing audio..."):
                            try:
                                model.eval()
                                with torch.no_grad():
                                    if model_name == "MobileNet":
                                        start_time = time.time()
                                        output = model(input_audio.float())
                                        compute_time = time.time() - start_time
                                    elif model_name == "Wav2vec":
                                        start_time = time.time()
                                        output = model(input_audio.float()).logits
                                        compute_time = time.time() - start_time
                                    else:
                                        assert "Invalid model name"
                                        st.stop()
                                    prob = torch.nn.functional.softmax(output, dim=-1).squeeze()[1]
                                    probs.append(prob)
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                                st.stop()
                        mean_time += compute_time
                        if output_status == 'Display predictions for each audio':
                            st.write(f"**Aphasia probability for #{ind + 1} file:** {prob:.2%}")
                            st.write(f"**Compute time:** {compute_time:.4f}s")

                            if prob > 0.5:
                                st.error("Signs of aphasia detected.")
                            else:
                                st.success("No signs of aphasia detected.")

                    if output_status == 'Display mean prediction':
                        st.success("Analysis complete.")
                        st.write(f"**Aphasia probability:** {sum(probs) / len(probs):.2%}")
                        st.write(f"**Compute time:** {mean_time / len(probs):.4f}s")

                        if sum(probs) / len(probs) > 0.5:
                            st.error("Signs of aphasia detected.")
                        else:
                            st.success("No signs of aphasia detected.")
else:
    text_content = load_readme()
    st.markdown(text_content, unsafe_allow_html=True)

import streamlit as st
import numpy as np
import librosa
import librosa.display
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Set background color to black using custom HTML/CSS
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: black;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, .markdown-text-container {
            color: white;
        }
        .stMarkdown, .stText, .stSubheader {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Music Genre Classifier")
st.markdown("Upload an audio file and the model will predict its genre!")

debug_mode = st.sidebar.checkbox("Enable Debug Mode")

classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

@st.cache_resource
def load_trained_model(path="music_genre_model.h5"):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features(audio_path, model, debug=False):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) > sr * 3:
            middle = len(y) // 2
            y = y[middle - (sr * 3) // 2:middle + (sr * 3) // 2]
        if debug:
            st.sidebar.write(f"Sample rate: {sr}")
            st.sidebar.write(f"Audio length: {len(y)} samples ({len(y)/sr:.2f} seconds)")
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmin=20, fmax=8000
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
        if debug:
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            plt.colorbar(img, format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            st.sidebar.pyplot(fig)
        resized = cv2.resize(norm_spec, (150, 150))
        input_shape = model.input_shape[1:]
        if len(input_shape) == 3 and input_shape[-1] == 3:
            resized = cv2.cvtColor(resized.astype('float32'), cv2.COLOR_GRAY2RGB)
            reshaped = resized.reshape(1, 150, 150, 3)
        else:
            reshaped = resized.reshape(1, 150, 150, 1)
        display_spec = norm_spec.copy()
        return reshaped, display_spec, sr
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        if debug:
            st.sidebar.write(f"Error details: {str(e)}")
        return None, None, None

def check_predictions(model, features, confidence_threshold=0.5):
    preds = model.predict(features)
    temperature = 1.5
    scaled_preds = tf.nn.softmax(preds[0] / temperature).numpy()
    probs = {classes[i]: float(scaled_preds[i]) for i in range(len(classes))}
    return probs, preds

def main():
    model = load_trained_model()
    if model is None:
        st.stop()

    st.info(f"Model can classify these genres: {', '.join(classes)}")

    with st.sidebar.expander("Advanced Options"):
        segment_duration = st.slider("Segment duration (seconds)", 1.0, 10.0, 3.0, 0.5)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name

        st.audio(uploaded_file, format='audio/' + uploaded_file.name.split('.')[-1])

        if st.button("🎶 Classify Genre"):
            with st.spinner("Extracting features..."):
                features, display_spec, sr = extract_features(audio_path, model, debug=debug_mode)

            if features is not None:
                with st.spinner("Predicting..."):
                    try:
                        class_probs, raw_preds = check_predictions(model, features, confidence_threshold)
                        pred_index = np.argmax(raw_preds[0])
                        pred_genre = classes[pred_index]
                        confidence = raw_preds[0][pred_index] * 100

                        st.markdown(f"### 🎧 Predicted Genre: <span style='color:lime'>{pred_genre.upper()}</span>", unsafe_allow_html=True)
                        st.write(f"Confidence: {confidence:.2f}%")

                        st.subheader("Mel Spectrogram")
                        try:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            if display_spec is not None and sr is not None:
                                img = librosa.display.specshow(
                                    display_spec, 
                                    sr=sr, 
                                    x_axis='time', 
                                    y_axis='mel', 
                                    ax=ax
                                )
                                plt.colorbar(img, format='%+2.0f dB')
                                plt.title('Mel Spectrogram')
                                st.pyplot(fig)
                            else:
                                st.warning("Could not display spectrogram")
                        except Exception as e:
                            st.warning(f"Could not display spectrogram: {e}")
                            normalized_for_display = (features[0, :, :, 0] - np.min(features[0, :, :, 0])) / (np.max(features[0, :, :, 0]) - np.min(features[0, :, :, 0]))
                            st.image(normalized_for_display, caption="Spectrogram used for prediction", use_container_width=True)

                        if debug_mode:
                            st.sidebar.subheader("Raw Predictions")
                            st.sidebar.write(raw_preds)
                            st.sidebar.write(f"Model input shape: {model.input_shape}")
                            st.sidebar.write(f"Features shape: {features.shape}")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        if debug_mode:
                            st.sidebar.write(f"Error details: {str(e)}")
                            import traceback
                            st.sidebar.code(traceback.format_exc())

        try:
            os.unlink(audio_path)
        except:
            pass

    with st.expander("🔧 Tips for Accurate Classification"):
        st.markdown("""
- **Select clean audio samples**: Choose songs with minimal background noise for best results.
- **Use representative segments**: The middle portion of a song often best represents its genre.
- **Genre boundaries are fuzzy**: Some songs blend multiple genres, making classification challenging.
- **Length matters**: 3-5 seconds of audio usually provides sufficient information for classification.
- **Consider audio quality**: Higher quality audio files (320kbps or lossless) generally yield better results.
- **Experiment with segments**: If one section misclassifies, try another section of the same song.
- **Enable debug mode**: Check the spectrogram to see what the model "sees" when classifying.
        """)

    with st.sidebar.expander("About Genre Classification"):
        st.markdown("""
This application uses deep learning to classify music into genres based on audio characteristics.
The model analyzes the mel spectrogram, which represents the frequency content of audio over time.

Different genres have distinctive spectral patterns:
- Classical: Complex harmonic structures, wider dynamic range
- Metal: High energy across spectrum, particularly in high frequencies
- Hip-hop: Strong bass and percussive elements
- Jazz: Rich mid-range harmonics with variable rhythmic patterns

The accuracy is typically 70-85% even in professional systems due to the subjective nature of genre boundaries.
        """)

if __name__ == "__main__":
    main()

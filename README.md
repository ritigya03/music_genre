# 🎵 Music Genre Classifier

A Streamlit-based web app that classifies music genres using a deep learning model trained on mel spectrograms.

🌐 **[Live Demo](https://musicgenre-zm2vlwg4wd3taaymeqhfk2.streamlit.app/)**  


---

## 🚀 Features

- 🎧 Upload MP3/WAV/OGG files
- 🔊 Converts audio into mel spectrograms
- 🧠 Predicts one of 10 music genres:
  - `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- 🎛️ Adjustable confidence thresholds and segment duration
- 📊 Visualizes mel spectrogram used in prediction
- 🐛 Optional debug mode to inspect intermediate outputs
- ☁️ Automatically downloads the model on first run via Google Drive

---

For testing - Some audio files present in audioSamples cd into it and use

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/ritigya03/music_genre
cd dsc_music
streamlit run app.py


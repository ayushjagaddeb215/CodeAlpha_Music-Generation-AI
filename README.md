# CodeAlpha_Music-Generation-AI
This project demonstrates how to generate music using a Long Short-Term Memory (LSTM) neural network trained on a dataset of MIDI files. The model learns patterns from musical sequences and produces new, AI-composed instrumental music in a classical piano style.

# 🎶 AI Music Generation using LSTM | CodeAlpha Internship - Task 3

This project demonstrates how to generate music using a **Long Short-Term Memory (LSTM)** neural network trained on a dataset of MIDI files. The model learns musical patterns and produces new, AI-composed instrumental music in a classical piano style.

---

## 🚀 Features

- 🎼 Parses MIDI files to extract notes and chords using `music21`
- 🔢 Prepares sequences for LSTM-based training
- 🧠 Trains an LSTM neural network using `TensorFlow/Keras`
- 🎹 Generates original musical compositions
- 🎵 Saves the output as a `.mid` file for playback

---

## 📂 Folder Structure
├── midi_files/ # Folder for input MIDI files (training data)
├── generated_music.mid # Output MIDI file (AI-generated)
└── music_generation.py # Main Python script

Install dependencies:
pip install music21 tensorflow numpy

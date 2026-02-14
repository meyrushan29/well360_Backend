import soundfile as sf
import numpy as np
import joblib
import librosa
import librosa.display
import Audio_feature_extraction as Afe
import tkinter as tk
from tkinter import filedialog
import json
import matplotlib.pyplot as plt
import os
import csv
import time

# =====================================================
# CONFIG
# =====================================================
MODEL_FILE = "Emotion_Model.pkl"
JSON_FILE = "recommendationA.json"
CSV_FILE = "audio_emotion_history.csv"

emotions = ['Fear', 'Angry', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']

json_key_map = {
    "Fear": "fearful",
    "Angry": "angry",
    "Disgust": "disgusted",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprise": "surprised"
}

# =====================================================
# LOAD MODEL
# =====================================================
loaded_model = joblib.load(MODEL_FILE)

# =====================================================
# LOAD RECOMMENDATIONS (SAFE UNICODE)
# =====================================================
with open(JSON_FILE, "r", encoding="utf-8") as f:
    recommendations = json.load(f)["base_phrases"]

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def get_features(file_path):
    with sf.SoundFile(file_path) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate

        chroma = Afe.feature_chromagram(waveform, sample_rate)
        mel = Afe.feature_melspectrogram(waveform, sample_rate)
        mfcc = Afe.feature_mfcc(waveform, sample_rate)

        features = np.hstack((chroma, mel, mfcc))

    return features, waveform, sample_rate

# =====================================================
# CSV FUNCTIONS (CONTINUOUS SAVE)
# =====================================================
def save_emotion_history(timestamp, emotion, audio_file):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "emotion", "audio_file"])
        writer.writerow([timestamp, emotion, os.path.basename(audio_file)])


def get_last_emotion():
    if not os.path.isfile(CSV_FILE):
        return None

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        return rows[-1][1]

# =====================================================
# SHOW RECOMMENDATIONS
# =====================================================
def show_recommendations(emotion):
    key = json_key_map.get(emotion)

    print(f"\n🧠 Average Emotion: {emotion}")
    print("💡 Recommendations:")

    if key in recommendations:
        for rec in recommendations[key][:3]:
            print(f" - {rec}")
    else:
        print(" - No recommendation available.")

# =====================================================
# AUDIO FILE PICKER (FRONT OF WINDOW)
# =====================================================
def select_audio_file():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        initialdir=r"C:\Users\Kavi\OneDrive\Desktop\FaceEmoV2 pp1 -- ok (2)\FaceEmoV2 pp1 -- ok\AudioModel\Sound test",
        filetypes=[("WAV files", "*.wav")]
    )

    root.destroy()
    return file_path

# =====================================================
# MAIN LOGIC
# =====================================================
def main():
    print("🎙️ Audio Emotion Prediction System")

    last_emotion = get_last_emotion()
    if last_emotion:
        print(f"\nLast detected average emotion: {last_emotion}")
        ans = input("Is this still valid? (yes/no): ").strip().lower()

        if ans == "yes":
            show_recommendations(last_emotion)
            return

    print("\nPlease select an audio file...")
    audio_file = select_audio_file()

    if not audio_file:
        print("❌ No file selected. Exiting.")
        return

    # Feature extraction
    features, waveform, sample_rate = get_features(audio_file)

    # Prediction
    prediction = loaded_model.predict([features])
    predicted_emotion = emotions[int(prediction[0]) - 1]

    print(f"\n🎯 Predicted Emotion: {predicted_emotion}")

    # Save to CSV (CONTINUOUS)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_emotion_history(timestamp, predicted_emotion, audio_file)

    # Recommendations
    show_recommendations(predicted_emotion)

    # =================================================
    # VISUALIZATION
    # =================================================
    plt.figure(figsize=(14, 10))

    # Waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title("Waveform")

    # MFCC
    mfcc_full = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
    plt.subplot(4, 1, 2)
    librosa.display.specshow(mfcc_full, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.subplot(4, 1, 3)
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")

    # Chromagram
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
    plt.subplot(4, 1, 4)
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title("Chromagram")

    plt.tight_layout()
    plt.show()

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()

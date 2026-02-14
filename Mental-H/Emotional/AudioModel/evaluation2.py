""" import numpy as np
import pandas as pd
import soundfile
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler
import Audio_feature_extraction as Afe

# -------------------------------
# Function to extract features from audio
# -------------------------------
def get_features(file):
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate

        chromagram = Afe.feature_chromagram(waveform, sample_rate)
        melspectrogram = Afe.feature_melspectrogram(waveform, sample_rate)
        mfcc = Afe.feature_mfcc(waveform, sample_rate)

        feature_matrix = np.hstack((chromagram, melspectrogram, mfcc))
    return feature_matrix

# -------------------------------
# Load dataset and extract features
# -------------------------------
def load_data(dataset_path):
    X, y = [], []
    count = 0
    emotions_map = {}
    for idx, emotion_folder in enumerate(os.listdir(dataset_path)):
        emotions_map[str(idx + 1).zfill(2)] = emotion_folder
        emotion_path = os.path.join(dataset_path, emotion_folder)
        if os.path.isdir(emotion_path):
            for file in glob.glob(os.path.join(emotion_path, "*.wav")):
                features = get_features(file)
                X.append(features)
                y.append(str(idx + 1).zfill(2))
                count += 1
                print(f'\rProcessed {count} audio samples', end=' ')
    print()
    return np.array(X), np.array(y), emotions_map

# -------------------------------
# Main Evaluation
# -------------------------------
if __name__ == "__main__":
    dataset_path = "DataSet"  # Path to your dataset
    X, y, emotions_map = load_data(dataset_path)

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load trained model
    model = joblib.load("Emotion_Model.pkl")

    # Predict
    y_pred = model.predict(X_scaled)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"\nTest Accuracy: {accuracy*100:.2f}% ({np.sum(y_pred == y)}/{len(y)})")    
      

      """

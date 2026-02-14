import soundfile as sf
import numpy as np
import joblib
import Audio_feature_extraction as Afe
import warnings
import re
from pydub import AudioSegment
import os

def get_features(file):
    with sf.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        # Compute features
        chromagram = Afe.feature_chromagram(waveform, sample_rate)
        melspectrogram = Afe.feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = Afe.feature_mfcc(waveform, sample_rate)
        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))

    return feature_matrix

# Trained Model
model_filename = "Emotion_Model.pkl"
loaded_model = joblib.load(model_filename)

# Input Audio for Test
input_audio_file = "UKTest.wav"

# Convert the WAV from stereo to mono
sound = AudioSegment.from_wav(input_audio_file)
sound = sound.set_channels(1)
output_audio_file = "output.wav"
sound.export(output_audio_file, format="wav")

# Extract features from the input and make predictions
input_features = get_features(output_audio_file)
prediction = loaded_model.predict([input_features])
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
predicted_emotion = emotions[int(prediction[0]) - 1]
print(f"Predicted Emotion: {predicted_emotion}")

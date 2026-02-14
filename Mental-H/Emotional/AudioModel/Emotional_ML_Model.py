import pandas as pd
import numpy as np
import soundfile
import os, glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')
import Audio_feature_extraction as Afe
import joblib

# load  soundfile and compute features
def get_features(file):
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype="float32")
        sample_rate = audio.samplerate
        chromagram = Afe.feature_chromagram(waveform, sample_rate)
        melspectrogram = Afe.feature_melspectrogram(waveform, sample_rate)
        mfc_coefficients = Afe.feature_mfcc(waveform, sample_rate)
        feature_matrix = np.array([])
        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))
        return feature_matrix

def load_data(dataset_path):
    X, y = [], []
    count = 0
    emotions = {}
    for idx, emotion_folder in enumerate(os.listdir(dataset_path)):
        emotions[str(idx + 1).zfill(2)] = emotion_folder
        emotion_path = os.path.join(dataset_path, emotion_folder)
        if os.path.isdir(emotion_path):
            for file in glob.glob(os.path.join(emotion_path, "*.wav")):
                features = get_features(file)
                X.append(features)
                y.append(str(idx + 1).zfill(2))  #
                count += 1
                print('\r' + f' Processed {count} audio samples', end=' ')
    return np.array(X), np.array(y)



# Define your dataset path
dataset_path = "DataSet"
features, emotions = load_data(dataset_path)
features_df = pd.DataFrame(features)


# unscaled features
scaler = StandardScaler()
features_scaled = features
features_scaled = scaler.fit_transform(features_scaled)

scaler = MinMaxScaler()
features_minmax = features
features_minmax = scaler.fit_transform(features_minmax)
features_scaled_df = pd.DataFrame(features_scaled)
features_minmax_df = pd.DataFrame(features_minmax)


# test/train set
X_train, X_test, y_train, y_test = train_test_split(
    features,
    emotions,
    test_size=0.2,
    random_state=69
)

X_train_scaled, X_test_scaled, _, _ = train_test_split(
    features_scaled,
    emotions,
    test_size=0.2,
    random_state=69
)

X_train_minmax, X_test_minmax, _, _ = train_test_split(
    features_scaled,
    emotions,
    test_size=0.2,
    random_state=69
)



#Random Forest
model = RandomForestClassifier(
    random_state=69
)

model.fit(X_train, y_train)

#Tuned Random Fores
model = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    warm_start=True,
    max_features='sqrt',
    oob_score=True,
    random_state=69
)

rm = model.fit(X_train, y_train)


# Save the model to a file
model_filename = "Emotion_Model.pkl"
joblib.dump(model, model_filename)

print(f'Model Accuracy  {100*model.score(X_test, y_test):.2f}%')

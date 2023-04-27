import librosa
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Extract features (mfcc, chroma,mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name)
    if chroma:
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:
        mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfcc))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions={
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07':  'disgust',
    '08':  'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data(test_size=0.3):
    x, y = [], []
    for file in glob.glob("RAVDESS/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=0)


# Spliting the Dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.3)


# Applying 6th Algorithm of Random Forest Classifier
print("\n\nRandom Forest\n\n")
RFClassifier = RandomForestClassifier(n_estimators=100)
RFClassifier.fit(x_train,y_train)
y_pred = RFClassifier.predict(x_test)
print(y_pred)
# Random Forest has 65-67% accuracy
print("\nThe Accurarcy of Random forest Classifier is : ", accuracy_score(y_pred, y_test)*100)
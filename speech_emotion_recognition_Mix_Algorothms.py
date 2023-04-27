import librosa
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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


# 1st Algortihm of Multi Layer Preceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
#train the Model
model.fit(x_train,y_train)
# Predict for the Test Set
y_pred = model.predict(x_test)
print("ORIGINAL Y NEEDED : \n",y_train)
print("PREDICTED Y NEEDED : \n",y_pred)
# Calculate the Accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# Accuracy by MLP is 55-62%
print("Accuracy using MLP CLASSIFIER : {:.2f}%".format(accuracy*100))


# Applying 2nd Algorithm of SVM Classifier
print("\n\n______SVM______\n\n")
machine = SVC(random_state=0)
machine.fit(x_train,y_train)
y_pred = machine.predict(x_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# SVM also gives 47.62% accuracy
print("\nThe Accurarcy of SVM is : ",accuracy_score(y_pred, y_test)*100,"\n")


# Applying 3rd Algorithm of Logistic regression Classifier
print("\n\n____Logistic Regression____")
lg = LogisticRegression(random_state=0 , multi_class='auto', solver='lbfgs', max_iter=1000)
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# THis Algorithm gives 62.77% accuracy with INTERATIONS REACHED LIMIT ERROR.
print("\nThe Accurarcy of LOGISTIC REGRESSION is : ", accuracy_score(y_pred, y_test)*100,"\n")


# Applying 4th Algorithm of K-neighbors Classifier
print("\n\n____KNN____\n\n")
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("\nPREDICTED Y based on given X_train & Y_train :\n ",y_pred)
# This has 62% accuracy
print("\nThe Accurarcy of KNN is : ", accuracy_score(y_pred, y_test)*100,"\n")


# Applying 5th Algorithm of Naive bayes Classifier
print("\n\n NAIVE BAYES ALGORITHM : \n\n")
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)
y_pred = NBClassifier.predict(x_test)
print(y_pred)
# Naive Bayes has 43.30% accuracy
print("\nThe Accurarcy of Naive Bayes is : ", accuracy_score(y_pred, y_test)*100, "\n")


# Applying 6th Algorithm of Random Forest Classifier
print("\n\nRandom Forest\n\n")
RFClassifier = RandomForestClassifier(n_estimators=100)
RFClassifier.fit(x_train,y_train)
y_pred = RFClassifier.predict(x_test)
print(y_pred)
# Random Forest has 65-67% accuracy
print("\nThe Accurarcy of Random forest Classfier is : ", accuracy_score(y_pred, y_test)*100)


# Applying 7th Algorithm of Decision Tree Classifier
print("\n\nDecision Tree Classifier\n\n")
DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(x_train, y_train)   # Providing algorithm the Training dataset
# Prediction by giving X_test Data to get correct y output..using DT Classifier Algorithm.
y_pred = DTClassifier.predict(x_test)
print(y_pred)
# DT Classifier Algorithm gives the 52.40% accuracy
print("\nThe Accurarcy of Decision Tree is : ", accuracy_score(y_pred, y_test))
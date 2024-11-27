import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Warnungen ignorieren
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Funktion zur Extraktion von MFCC-Features aus Audiodaten
def extract_features(audio, sr, n_mfcc=13, n_fft=416, hop_length=512, n_mels=64, max_pad_len=400):
    """
    Diese Funktion extrahiert MFCC (Mel-Frequency Cepstral Coefficients) aus den Audiodaten.
    Quelle: https://librosa.org/doc/latest/feature.html#mfcc
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    #print(f"MFCCs Shape (before padding/truncating): {mfccs.shape}")
    
    # Padding oder Kürzen der MFCC-Daten, um eine einheitliche Länge sicherzustellen
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs.flatten() #Shape Für SVM anpassen

# Augmentation: Add noise
def augment_audio(audio):
    noise = np.random.randn(len(audio)) * 0.005
    return audio + noise

#Merkmale eine Einzelne Datei EXtrahieren.
def process_file(file_path, label):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # Lower sample rate for speed
        features = [extract_features(audio, sr)]  # Original
        augmented_audio = augment_audio(audio)
        features.append(extract_features(augmented_audio, sr))  # Augmented
        labels = [label] * len(features)
        return features, labels
    except Exception as e:
        print(f"Fehler während der Bearbeitung des Dateien {file_path}: {e}")
        return [], []

# Funktion zum Laden der Audiodaten und Extrahieren der zugehörigen Merkmale und Labels
def load_data(audio_folder_path):
    files = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith(".wav")]
    results = Parallel(n_jobs=-1)(delayed(process_file)(
        file, 0 if "felix" in os.path.basename(file).lower() else 1 if "linelle" in os.path.basename(file).lower() else 2
    ) for file in files)
    
    features, labels = [], []
    for f, l in results:
        features.extend(f)
        labels.extend(l)
    
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Es wurde Kein Daten wegen labels in Datei gefunden")
    
    return np.array(features), np.array(labels)


# SVM-Modell mit den angegebenen Parametern erstellen
def create_svm_model():
    
    svm_model = SVC()
    svm_model.C=0.1
    svm_model.kernel='rbf'
    svm_model.degree=3
    svm_model.gamma='scale'
    svm_model.probability=True
    svm_model.class_weight='balanced'
    
    #diese Anderen Parameter des Support Vector Classifier sind für unseren Anforderung nicht relevant
    """svm_model.coef0=1.0, svm_model.shrinking=True, svm_model.tol=1e-3, svm_model.cache_size=250, svm_model.verbose=1,
    svm_model.max_iter=-1,svm_model.decision_function_shape='ovo, svm_model.break_ties=False, svm_model.random_state=None"""
    
    return svm_model


def train_svm_model(path):
    
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm_model = create_svm_model()
    svm_model.fit(X_train, y_train)
    
    accuracy = svm_model.score(X_test, y_test)
    print(f"Genauigkeit des SVM-Modells: {accuracy*100:.2f}")
    print()
    y_pred=svm_model.predict(X_test)
    
    #visualisierung der Daten mithilfe der Confusion Matrix
    #confusion_matrix(y_test,y_pred)
    # Classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Felix", "Linelle", "Unknown"], yticklabels=["Felix", "Linelle", "Unknown"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    
    return svm_model,scaler

# Classification report and confusion matrix
def plot_confusion_matrix(y_test,y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Felix", "Linelle", "Unknown"], yticklabels=["Felix", "Linelle", "Unknown"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Predict speaker
def predict_speaker(model, audio_file, scaler):
    try:
        audio, sr = librosa.load(audio_file, sr=16000)  # Lower sample rate for speed
        features = extract_features(audio, sr)
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        speaker = ["Felix", "Linelle", "Unknown"][prediction]
        print(f"File: {audio_file}, Predicted Speaker: {speaker}")
        return speaker
    except Exception as e:
        print(f"Erreur lors de la prédiction du fichier {audio_file}: {e}")
        return "Erreur"
    
    
def process_audio_file(file_path, model, scaler):
    if not os.path.isfile(file_path):
        print("Die angegebene Datei existiert nicht.")
        return

    audio, sr = librosa.load(file_path, sr=22050)
    segment_length = sr // 2  # Segmentlänge (0.5 Sekunden)
    overlap_factor = 0.5  # Überlappungsfaktor (50%)
    overlap_length = int(segment_length * overlap_factor)  # Berechne die Überlappung

    print(f"Datei analysieren: {file_path}")
    print(f"Gesamtdauer: {len(audio) / sr:.2f} Sekunden, Segmente werden verarbeitet.")

    current_speaker = None
    confirmed_speaker = None
    segment_start_time = 0
    buffer = []
    results = []

    # Berechne die Anzahl der Segmente unter Berücksichtigung der Überlappung
    start = 0
    while start < len(audio):
        end = start + segment_length
        segment = audio[start:end]

        if len(segment) < segment_length:  # Wenn das Segment kürzer ist, breche ab
            break


        #predict_speaker2(model, segment, scaler,sr=22050)
        mfccs = extract_features(segment, sr)
        
        #print(f"MFCCs pour le segment (start={start}, end={end}): {mfccs}")
        mfccs = scaler.transform([mfccs])
        prediction = model.predict(mfccs)
        speaker = ["Felix", "Linelle", "Julia", "Unbekannt"][prediction[0]]
        results.append((speaker, start / sr, end / sr))  # Zeit in Sekunden

        buffer.append(speaker)
        if len(buffer) > 2:
            buffer.pop(0)

        if len(buffer) == 2 and buffer[0] == buffer[1]:
            current_speaker = buffer[0]
            if current_speaker != confirmed_speaker:
                if confirmed_speaker is not None:
                    segment_end_time = start / sr
                    results.append((confirmed_speaker, segment_start_time, segment_end_time))
                    print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")
                confirmed_speaker = current_speaker
                segment_start_time = start / sr

        start += segment_length - overlap_length  # Aktualisiere den Start mit Überlappung

    if confirmed_speaker is not None:
        segment_end_time = len(audio) / sr
        results.append((confirmed_speaker, segment_start_time, segment_end_time))
        print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

    return results

# Hauptprogramm
if __name__ == "__main__":
    
    audio_path = r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen"
    model ,scaler= train_svm_model(audio_path)
   
    
    
    test_files=[
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Felix_1_1.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Felix_15_2.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Linelle_7_1.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Linelle_10_2.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\LinelleNew14.wav"
    ]
     
    for file in test_files:
        predict_speaker(model, file, scaler)
        # test mit Segmentierte Audio Dateien
        process_audio_file(file, model,scaler)
        print()


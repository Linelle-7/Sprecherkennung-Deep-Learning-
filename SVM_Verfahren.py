import librosa
import numpy as np
import pandas as pd
import os
import sounddevice as sd
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sounddevice as sd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.datasets import load_iris
from scipy.stats import uniform

# Optional: for parallel processing
from multiprocessing import Pool

#import für record_Audio methode
import signal
import time

# Optional: for hyperparameter optimization
#import optuna
#import lightgbm as lgb
#import flaml

# Optional: for machine learning models and training
#from flaml import AutoML


#Randomize optimisation Params sklearn

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


# Funktionen zur Erstellung und Suche nach besten Hyperparametern

#Hyperparameter-Tunning mit Randomize-search
def randomized_search_svm(X_train, y_train, n_iter=50, random_state=42):
    """
    Diese Funktion führt einen RandomizedSearchCV auf einem SVM-Modell durch,
    um die besten Hyperparameter zu finden.
    """
    print("    ***Starte RandomizedSearchCV...")
    
    # Definiere die Parameterbereiche für RandomizedSearch
    param_dist = {
        'C': uniform(0.1, 100),  # Von 0.1 bis 100
        'kernel': ['linear', 'rbf', 'poly'],  # Unterschiedliche Kernel , 'sigmoid'
        'gamma': ['scale', 'auto', 0.1, 1e-2, 'scale'],  # Gamma-Werte
        'degree': [1,2, 3, 4, 5],  # Grad für 'poly' Kernel
    }
    # Erstelle das SVM-Modell
    svm_model = SVC(probability=True, class_weight='balanced')
    
    # RandomizedSearchCV mit Cross-Validation
    randomized_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, 
                                           n_iter=n_iter, cv=5, verbose=1, n_jobs=-1, random_state=random_state)
    
    # Führe das RandomizedSearch durch
    randomized_search.fit(X_train, y_train)
    
    print(f"Beste Parameter: {randomized_search.best_params_}")
    print(f"Beste Kreuzvalidierungsgenauigkeit: {randomized_search.best_score_ * 100:.2f}%")
    
    return randomized_search.best_estimator_

# Hy perparameter-Tuning mit GridSearchCV
def hyperparameter_tuning_GridSsearchCV(X_train, y_train):
    
    print("   ***Starte Hyperparameter-Tuning...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10,1e-4,1e-3,1e-5,1e2,1e3,5e-2,5e-3],
        'kernel': ['linear', 'rbf', 'poly','sigmoid'],
        'gamma': ['scale', 'auto',0.1,1e-2,1e-3,1e-4,1e2,1e3],
        'degree': [2, 3, 4,5,1]
    }
    # GridSearchCV mit verschiedenen Metriken
    grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Beste Parameter: {grid_search.best_params_}")
    print(f"Beste Kreuzvalidierungsgenauigkeit: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

# SVM Modell trainieren
def train_svm_model(path):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Hyperparameter-Tuning und Modelltraining
    best_model = hyperparameter_tuning_GridSsearchCV(X_train, y_train)
    
    # SVM-Modell mit RandomizedSearchCV trainieren
    #best_model = randomized_search_svm(X_train, y_train)
    
    best_model.fit(X_train, y_train)
    accuracy = best_model.score(X_test, y_test)
    print(f"Genauigkeit des besten SVM-Modells: {accuracy*100:.2f}%")
    
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    
    y_pred = best_model.predict(X_test)
    
    # Evaluieren des Modells
    evaluate_model(y_test, y_pred)
    
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeit für die positive Klasse
    plot_roc_curve(y_test, y_pred_prob) #ROC-KURVE
    
    plot_precision_recall_curve(y_test, y_pred_prob)
    
    plot_learning_curve(best_model, X_train, y_train) #plot der learning Kurve
    
    # if best_model.kernel =='linear':
    #     feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
    #     plot_feature_importance(best_model, feature_names)
    
    return best_model, scaler

# Funktion zur Berechnung von mehreren Metriken
def evaluate_model(y_test, y_pred):
    """
    Berechnet mehrere Metriken zur Modellbewertung und gibt sie aus.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    print(f"Genauigkeit: {accuracy * 100:.2f}%")
    print(f"Präzision (Weighted): {precision * 100:.2f}%")
    print(f"Recall (Weighted): {recall * 100:.2f}%")
    print(f"F1-Score (Weighted): {f1 * 100:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.2f}")
    

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()

# plot der hilft zu visualisieren wie gut das modell 
# die verschiedene klassen unterscheidet
def plot_precision_recall_curve(y_test, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


#Feature Importance Kurve
def plot_feature_importance(svm_model, feature_names): 
    # Wenn ein linearer SVM verwendet wird, sind die Koeffizienten die Wichtigkeit der Merkmale
    importance = np.abs(svm_model.coef0.flatten())
    
    # Sortiere nach der Wichtigkeit der Merkmale
    sorted_idx = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[sorted_idx], align='center')
    plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Linear SVM')
    plt.show()

#learning Kurve
def plot_learning_curve(model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train Score", color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test Score", color='green')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

# Klassifikationsbericht und Confusion Matrix visualisieren
def plot_confusion_matrix(y_test, y_pred):
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
        print(f"Extrahierte Eigenschaften für Vorhersage: {features}") 
        features = scaler.transform([features])
        print(f"Extrahierte Eigenschaften für Vorhersage nach Scaler Transform: {features}") 
        prediction = model.predict(features)[0]
        speaker = ["Felix", "Linelle", "Unknown"][prediction]
        print(f"File: {audio_file}, Predicted Speaker: {speaker}")
        return speaker
    except Exception as e:
        print(f"Fehler während das Vorhersage des Dateis  {audio_file}: {e}")
        return "Fehler"


def process_audio_file2(file_path, model, scaler):
    if not os.path.isfile(file_path):
        print("Die angegebene Datei existiert nicht.")
        return []
    
    def predict_segment_speaker(model, audio_segment, scaler):
        try:
            audio_segment=myaugment_audio(audio_segment)
            features = extract_features(audio_segment, sr)
            features = scaler.transform([features])
            prediction = model.predict(features)[0]
            speaker = ["Felix", "Linelle", "Unknown"][prediction]
            return speaker
        except Exception as e:
            print(f"Fehler bei der Segmentvorhersage: {e}")
            return "Fehler"
    
    def myaugment_audio(audio):
    # Beispiel: Random Noise Hinzufügen
        noise_factor = 0.005
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return np.clip(augmented_audio, -1.0, 1.0)

    # Audio laden und augmentieren
    audio, sr = librosa.load(file_path, sr=16000)
    audio = augment_audio(audio)
    segment_length = sr // 2  # Segmentlänge (0.5 Sekunden)
    overlap_factor = 0.25
    overlap_length = int(segment_length * overlap_factor)

    print(f"Datei analysieren: {file_path}")
    print(f"Gesamtdauer: {len(audio) / sr:.2f} Sekunden, Segmente werden verarbeitet.")

    confirmed_speaker = None
    segment_start_time = 0
    buffer = []
    transcript = []

    start = 0
    while start < len(audio):
        end = start + segment_length
        segment = audio[start:end]

        if len(segment) < segment_length * 0.8:  # Zu kurze Segmente überspringen
            break

        speaker = predict_segment_speaker(model, segment, scaler)
        buffer.append(speaker)

        # Puffer verwenden, um einen Sprecher zu bestätigen
        if len(buffer) > 3:  # Gleitspeichergröße: 3
            buffer.pop(0)

        if len(buffer) == 3 and len(set(buffer)) == 1:  # Stabiler Sprecher erkannt
            current_speaker = buffer[0]
            if current_speaker != confirmed_speaker:
                if confirmed_speaker is not None:
                    segment_end_time = start / sr
                    transcript.append(
                        f"{confirmed_speaker} ({segment_start_time:.2f}s - {segment_end_time:.2f}s)"
                    )
                    print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")
                confirmed_speaker = current_speaker
                segment_start_time = start / sr

        start += segment_length - overlap_length

    # Letzten Sprecher speichern
    if confirmed_speaker is not None:
        segment_end_time = len(audio) / sr
        transcript.append(
            f"{confirmed_speaker} ({segment_start_time:.2f}s - {segment_end_time:.2f}s)"
        )
        print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

    return transcript


def process_audio_file3(file_path, model, scaler):
    if not os.path.isfile(file_path):
        print("Die angegebene Datei existiert nicht.")
        return []
    
    def predict_segment_speaker(model, audio_segment, scaler):
        try:
            audio_segment = myaugment_audio(audio_segment)
            features = extract_features(audio_segment, sr)
            features = scaler.transform([features])
            prediction = model.predict(features)[0]
            speaker = ["Felix", "Linelle", "Unknown"][prediction]
            return speaker
        except Exception as e:
            print(f"Fehler bei der Segmentvorhersage: {e}")
            return "Fehler"
    
    def myaugment_audio(audio):
        # Beispiel: Random Noise Hinzufügen
        noise_factor = 0.005
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return np.clip(augmented_audio, -1.0, 1.0)

    # Audio laden und augmentieren
    audio, sr = librosa.load(file_path, sr=16000)
    audio = myaugment_audio(audio)
    
    segment_length = sr // 2  # Segmentlänge (0.5 Sekunden)
    overlap_factor = 0.25
    overlap_length = int(segment_length * overlap_factor)

    print(f"Datei analysieren: {file_path}")
    print(f"Gesamtdauer: {len(audio) / sr:.2f} Sekunden, Segmente werden verarbeitet.")

    confirmed_speaker = None
    segment_start_time = 0
    buffer = []
    transcript = []

    start = 0
    while start < len(audio):
        end = start + segment_length
        segment = audio[start:end]

        if len(segment) < segment_length * 0.8:  # Zu kurze Segmente überspringen
            break

        speaker = predict_segment_speaker(model, segment, scaler)
        buffer.append(speaker)

        # Puffer verwenden, um einen Sprecher zu bestätigen
        if len(buffer) > 3:  # Gleitspeichergröße: 3
            buffer.pop(0)

        if len(buffer) == 3 and len(set(buffer)) == 1:  # Stabiler Sprecher erkannt
            current_speaker = buffer[0]
            if current_speaker != confirmed_speaker:
                if confirmed_speaker is not None:
                    segment_end_time = start / sr
                    transcript.append(
                        f"{confirmed_speaker} ({segment_start_time:.2f}s - {segment_end_time:.2f}s)"
                    )
                    print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")
                
                # Setze den Startzeitpunkt des ersten bestätigten Sprechers auf 0
                if confirmed_speaker is None:  # Falls es der erste Sprecher ist
                    segment_start_time = 0.0
                else:
                    segment_start_time = start / sr

                confirmed_speaker = current_speaker

        start += segment_length - overlap_length

    # Letzten Sprecher speichern
    if confirmed_speaker is not None:
        segment_end_time = len(audio) / sr
        transcript.append(
            f"{confirmed_speaker} ({segment_start_time:.2f}s - {segment_end_time:.2f}s)"
        )
        print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

    return transcript

def process_mp3_file(file_path, model,scaler):
    if not os.path.isfile(file_path):
        print("Die angegebene Datei existiert nicht.")
        return

    # MP3 in Audio-Daten und Samplingrate umwandeln
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Segmentieren der Audiodaten in 0.1 Sekunden Abschnitte
    segment_length = int(sr * 0.1)  # 0.1 Sekunden pro Segment
    num_segments = len(audio) // segment_length

    print(f"Datei analysieren: {file_path}")
    print(f"Gesamtdauer: {len(audio) / sr:.2f} Sekunden, {num_segments} Segmente werden verarbeitet.")

    current_speaker = None
    previous_speaker = None
    confirmed_speaker = None
    segment_start_time = 0  # Beginn des aktuellen Sprechersegments
    results = []  # Ergebnisse sammeln

    for i in range(0, num_segments, 10):  # Alle 1 Sekunde (10x 0.1s Abschnitte)
        # Die 10 0.1s Abschnitte für die aktuelle Sekunde zusammenfassen
        speaker_counts = {0: 0, 1: 0, 2: 0}  # Zähler für die Stimmen: Felix (0), Linelle (1), Julia (2)
        
        # Verarbeiten der 10 Abschnitte pro Sekunde
        for j in range(i, i + 10):
            start = j * segment_length
            end = start + segment_length
            segment = audio[start:end]

            # MFCCs extrahieren
            mfccs = extract_features(segment, sr)
            mfccs = scaler.transform([mfccs])  # Für das Modell passend vorbereiten

            # Vorhersage
            prediction = model.predict(mfccs)[0]
            predicted_label = np.argmax(prediction, axis=0)

            # Stimmenzählung
            speaker_counts[predicted_label] += 1

        # Am häufigsten erkannte Stimme für diese Sekunde
        most_common_speaker = max(speaker_counts, key=speaker_counts.get)

        # Bestimmen des Sprechers anhand der Häufigkeit der Erkennung
        if most_common_speaker == 0:
            speaker = "Felix"
        elif most_common_speaker == 1:
            speaker = "Linelle"
        elif most_common_speaker == 2:
            speaker = "Julia"
        else:
            speaker = "Unbekannt"

        # Sprecherwechsel prüfen
        if speaker != confirmed_speaker:
            # Wenn sich der Sprecher geändert hat, das Segment speichern
            if confirmed_speaker is not None:
                segment_end_time = (i + 10) * 0.1  # Ende des vorherigen Sprechersegments
                results.append((confirmed_speaker, segment_start_time, segment_end_time))
                print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

            # Neuen Sprecher bestätigen
            confirmed_speaker = speaker
            segment_start_time = i * 0.1  # Beginn des neuen Sprechersegments

    # Restsegment (letzter Sprecherabschnitt)
    if confirmed_speaker is not None:
        segment_end_time = num_segments * 0.1  # Ende des letzten Segments
        results.append((confirmed_speaker, segment_start_time, segment_end_time))
        print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

    return results

def record_audio(duration, sr=16000, device_id=None):
    """
    Nimmt Audio auf. Wenn 'device_id' angegeben wird, wird dieses Mikrofon verwendet,
    ansonsten wird das Standardmikrofon verwendet.
    
    Returns:
    - Audio-Daten als NumPy-Array.
    """
    # Beispielaufruf
    # Standardmikrofon verwenden:
    # audio_data = record_audio(5)

    # Benutzerdefiniertes Mikrofon mit device_id verwenden:
    # audio_data = record_audio(5, device_id=2)  # Ersetze 2 durch die gewünschte device_id

    print("Aufnahme gestartet...")
    
    # Wenn keine device_id angegeben ist, verwenden wir das Standardgerät
    if device_id is None:
        # Holen des Standard-Audioeingabegeräts
        device_info = sd.query_devices(kind='input')
        device_id = device_info['index']  # Standardgerät verwenden
        print(f"Verwendetes Mikrofon: {device_info['name']}")
    else:
        # Ausgabe, wenn eine spezifische device_id gewählt wurde
        device_info = sd.query_devices(device_id, kind='input')
        print(f"Verwendetes Mikrofon: {device_info['name']} (benutzerdefiniert)")

    # Aufnahme vom angegebenen Gerät (entweder Standard oder benutzerdefiniert)
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=device_id)
    sd.wait()
    print("Aufnahme abgeschlossen.")
    return audio_data.flatten()

# Unterbrechungsfunktion
def handle_interrupt(signum, frame):
    raise KeyboardInterrupt

# Echtzeit-Sprechererkennung mithilfe von record_audio
def continuous_recognition(model,scaler,duration=5, sr=16000):
    
    def predict_recorded_speaker(model, rec_features, scaler):
        try:
           
            #features = extract_features(audio, sr)
            #print(f"Extrahierte Eigenschaften für Vorhersage: {features}") 
            features = scaler.transform([rec_features])
            #print(f"Extrahierte Eigenschaften für Vorhersage nach Scaler Transform: {features}") 
            prediction = model.predict(features)[0]
            speaker = ["Felix", "Linelle", "Unknown"][prediction]
            #print(f"File: {audio_file}, Predicted Speaker: {speaker}")
            return speaker
        except Exception as e:
            print(f"Fehler während das Vorhersage des Dateis  {rec_features}: {e}")
            return "Fehler"

    
    print("Drücke 'CTRL+C' oder beende das Programm, um die Erkennung zu stoppen.")
    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        
        while True:
            audio_data = record_audio(duration, sr)
            features = extract_features(audio_data, sr)
            speaker = predict_recorded_speaker(model, features,scaler)
            print(f"Sprecher erkannt: {speaker}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Echtzeit-Erkennung beendet.")



# Hauptprogramm
if __name__ == "__main__":
    
    audio_path = r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen"
    model ,scaler= train_svm_model(audio_path)
        
    test_files=[
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Felix_1_1.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Felix_15_2.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Linelle_7_1.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Linelle_10_2.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\LinelleNew14.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen_NT\Linelle_NT.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen_NT\Linelle_NT2.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen_NT\Linelle_NT3.wav"
    ]
     
    for file in test_files:
        predict_speaker(model, file, scaler)
        # test mit Segmentierte Audio Dateien
        process_audio_file3(file, model,scaler)
        
        #process_mp3_file(file, model,scaler)
        print()
    
    print(" ***Continuous Recognition startet jetzt.")
    continuous_recognition(model,scaler, duration=5,sr=1600)

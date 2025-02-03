import librosa
import numpy as np
import pandas as pd
import os
import sounddevice as sd
from collections import Counter
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sounddevice as sd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_curve, auc,
    precision_recall_curve, roc_auc_score
    )
from sklearn.datasets import load_iris
from scipy.stats import uniform
from scipy.ndimage import uniform_filter1d
from hmmlearn import hmm
from matplotlib import cm
import queue

# Optional: for parallel processing
from multiprocessing import Pool

#import für record_Audio methode
import signal
import time
import speech_recognition as sr
import audioread 


# Warnungen ignorieren
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Funktion zur Extraktion von MFCC-Features aus Audiodaten
def extract_features(audio, sr=22050, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40, max_pad_len=400):
    """
    Diese Funktion extrahiert MFCC (Mel-Frequency Cepstral Coefficients) aus den Audiodaten.
    Quelle: https://librosa.org/doc/latest/feature.html#mfcc
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,fmax=sr//2)
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
def process_file(file_path, label, segment_length=0.1, sr=22050):
    try:
        audio, sr = librosa.load(file_path, sr=22050)  # Lower sample rate for speed
        # features = [extract_features(audio, sr)]  # Original
        # augmented_audio = augment_audio(audio)
        features=[]
        labels=[] 
        #print("process_file in Bearbeitung.......") 
        segment_samples = int(segment_length * sr)
        num_segments = len(audio) // segment_samples

        # Segmentiere die Audiodatei und extrahiere MFCCs
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            mfccs = extract_features(segment, sr)
            features.append(mfccs)
            labels.append(label)

        return features, labels
    except Exception as e:
        print(f"Fehler während der Bearbeitung des Dateien {file_path}: {e}")
        return [], []

# Funktion zum Laden der Audiodaten und Extrahieren der zugehörigen Merkmale und Labels
def load_data(path,label_map,segment_length, sr=22050):
    
    print("load_data in Bearbeitung.......")
    features, labels = [], []

    for speaker in label_map.keys():
        speaker_path = os.path.join(path, speaker)
        if not os.path.exists(speaker_path):
            print(f"Warnung: Ordner {speaker_path} existiert nicht.")
            continue

        files = [os.path.join(speaker_path, file) for file in os.listdir(speaker_path) if file.endswith(".mp3") or file.endswith(".wav")]
        #print(f"Verarbeite {len(files)} Dateien für Klasse '{speaker}' (Label {label_map[speaker]})")
        if len(files) == 0:
            print(f"Keine Dateien für {speaker} gefunden.")
            continue
        results = Parallel(n_jobs=-1)(delayed(process_file) (file, label_map[speaker],segment_length=0.1, sr=22050) for file in files)
        
        for f, l in results:
            features.extend(f)
            labels.extend(l)
            #print(f"  - {len(f)} Features für Datei hinzugefügt. Aktuelle Labels: {Counter(labels)}")

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Es wurde kein Daten wegen Labels in Datei gefunden")
    
    return np.array(features), np.array(labels)


# Funktionen zur Erstellung und Suche nach besten Hyperparametern

#Hyperparameter-Tunning mit Randomize-search
def randomized_search_svm(X_train, y_train, n_iter=5, random_state=42):
    """
    Diese Funktion führt einen RandomizedSearchCV auf einem SVM-Modell durch,
    um die besten Hyperparameter zu finden.
    """
    print("    ***Starte RandomizedSearchCV...")
    
    # Definiere die Parameterbereiche für RandomizedSearch
    param_dist = {
        'C': uniform(0.1, 10),  # Von 0.1 bis 10
        'kernel': ['linear', 'rbf', 'poly'],  # Unterschiedliche Kernel , 'sigmoid'
        'gamma': ['scale', 'auto', 0.1, 1e-2, 'scale'],  # Gamma-Werte
        'degree': [1,2, 3, 4, 5],  # Grad für 'poly' Kerne
        'probability': [True]
    }
    # Erstelle das SVM-Modell
    svm_model = SVC( class_weight='balanced')
    
    # RandomizedSearchCV mit Cross-Validation
    randomized_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, 
                                           n_iter=n_iter, cv=5, verbose=1, n_jobs=-1, random_state=random_state)
    
    # Führe das RandomizedSearch durch
    randomized_search.fit(X_train, y_train)
    print("fitting abgeschlossen")
    
    
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

import optuna
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from joblib import parallel_backend

def objective(trial,X_train, y_train):
    """
    Optuna-Ziel-Funktion für die Hyperparameter-Optimierung.
    """
    # Definieren der Hyperparameterbereiche
    with parallel_backend("threading"):
        C = trial.suggest_float("C", 1e-2, 10)  # Logarithmischer Bereich
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3

        # Erstellen eines Modells mit den vorgeschlagenen Hyperparametern
        model = Pipeline([
            ('scaler', StandardScaler()),  # Features skalieren
            ('svm', SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, class_weight='balanced'))
        ])

        # 5-fache Kreuzvalidierung zur Bewertung des Modells
        score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=4)
        return score.mean()



from sklearn.utils import shuffle
from sklearn.decomposition import PCA


def train_svm_model_optuna(path, label_map,segment_length, sr=22050):
    
    
    X, y = load_data(path,label_map,segment_length,  sr)
    X, y = shuffle(X,y,random_state=42)
    
    print(f"Feature-Shape: {X.shape}, Label-Shape: {y.shape}")
    print(f"Label-Verteilung: {Counter(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    #print(f"Unique classes in y_train: {np.unique(y_train)}")
    #print(f"y_train counts: {np.bincount(y_train)}")
    
    """scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)"""

    """pca = PCA(n_components=500)  # Auf 500 Dimensionen reduzieren
    myScaler=pca
    X_train= pca.fit_transform(X_train)
    X_test = pca.transform(X_test)"""
    
    
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(f_classif, k=1000)  # Wähle die 1000 besten Features
    myScaler=selector
    X_train= selector.fit_transform(X_train, y_train)
    X_test =selector.transform(X_test)

    
    
    # Optuna-Studie erstellen und optimieren
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5,interval_steps=2)
    study = optuna.create_study(direction="maximize",pruner=pruner)  # Ziel ist, die Genauigkeit zu maximiereN
    
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10, n_jobs=2)  # 20 Iterationen mit Parallelisierung
    end_time = time.time()
    print(f"Optimierung abgeschlossen in {end_time - start_time:.2f} Sekunden.")

    # Ergebnisse anzeigen
    print(f"Beste Hyperparameter: {study.best_params}")
    print(f"Beste Kreuzvalidierungsgenauigkeit: {study.best_value:.4f}")

    # Bestes Modell trainieren
    best_params = study.best_params
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=best_params["C"],
            kernel=best_params["kernel"],
            gamma=best_params["gamma"],
            degree=best_params["degree"] if best_params["kernel"] == "poly" else 3,
            class_weight='balanced'
        ))
    ])
    best_model.fit(X_train, y_train)
    # Testen des Modells
    accuracy = best_model.score(X_test, y_test)
    print(f"Test-Genauigkeit: {accuracy * 100:.2f}%")

    
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred,label_map)
    
    # Evaluieren des Modells
    evaluate_model(y_test, y_pred,label_map)
    
    plot_learning_curve(best_model, X_train, y_train) #plot der learning Kurve
    
    
    return best_model, myScaler

# SVM Modell trainieren
def train_svm_model(path, label_map, segment_length=0.1, sr=22050):
    
    X, y = load_data(path,label_map, segment_length=0.1, sr=22050)
    X, y = shuffle(X,y,random_state=42)
    
    print(f"Feature-Shape: {X.shape}, Label-Shape: {y.shape}")
    print(f"Label-Verteilung: {Counter(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    #print(f"Unique classes in y_train: {np.unique(y_train)}")
    #print(f"y_train counts: {np.bincount(y_train)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Hyperparameter-Tuning und Modelltraining
    #best_model = hyperparameter_tuning_GridSsearchCV(X_train, y_train)
    
    # SVM-Modell mit RandomizedSearchCV trainieren
    start_time = time.time()
    best_model = randomized_search_svm(X_train, y_train)
    end_time = time.time()
    print(f"Optimierung mit Randomize abgeschlossen in {end_time - start_time:.2f} Sekunden.")
    
    best_model.fit(X_train, y_train)
    accuracy = accuracy.score(X_test, y_test)
    print(f"Genauigkeit des besten SVM-Modells: {accuracy*100:.2f}%")
    
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred,label_map)
    
    # Evaluieren des Modells
    evaluate_model(y_test, y_pred)
    
    #y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeit für die positive Klasse
    #plot_roc_curve(y_test, y_pred_prob) #ROC-KURVE
    
    #plot_precision_recall_curve(y_test, y_pred_prob)
    
    plot_learning_curve(best_model, X_train, y_train) #plot der learning Kurve
    
    # if best_model.kernel =='linear':
    #     feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
    #     plot_feature_importance(best_model, feature_names)
    
    return best_model, scaler

# Funktion zur Berechnung von mehreren Metriken
def evaluate_model(y_test, y_pred,label_map):
    """
    Berechnet mehrere Metriken zur Modellbewertung und gibt sie aus.
    """
    #print(classification_report(y_test, y_pred))
    
    print(classification_report(y_test, y_pred, target_names=label_map))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    #roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

    print(f"Genauigkeit: {accuracy * 100:.2f}%")
    print(f"Präzision (Weighted): {precision * 100:.2f}%")
    print(f"Recall (Weighted): {recall * 100:.2f}%")
    print(f"F1-Score (Weighted): {f1 * 100:.2f}%")
    #print(f"ROC AUC Score: {roc_auc:.2f}")
    

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
def plot_confusion_matrix(y_test, y_pred,label_map):
    labels=[]
    for name in label_map:
        labels.append(name)
        
    labels.append("unbekannt")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
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
        speaker = ["Biden", "Moderator" ,"Trump","Unbekannt"][prediction]
        print(f"File: {audio_file}, Predicted Speaker: {speaker}")
        return speaker
    except Exception as e:
        print(f"Fehler während das Vorhersage des Dateis  {audio_file}: {e}")
        return "Fehler"

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
            speaker = ["Biden", "Moderator" ,"Trump","Unbekannt"][prediction]
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

# Echzeiterkennung mit Callback
def real_time_recognition(model,scaler):
    prediction =None #Speichert erkannte Sprecher während Programmsdurchlauf
    print("Echzeiterkennung fängt gleich an ...")
    # Predict speaker
    def predict_speaker(model, audio, scaler):
        try:
            sr=16000
            features = extract_features(audio, sr)
            print(f"Extrahierte Eigenschaften für Vorhersage: {features}") 
            features = scaler.transform([features])
            #print(f"Extrahierte Eigenschaften für Vorhersage nach Scaler Transform: {features}") 
            prediction = model.predict(features)[0]
            speaker = ["Biden", "Moderator" ,"Trump","Unbekannt"][prediction]
            #print(f"File: {audio}, Predicted Speaker: {speaker}")
            return speaker
        except Exception as e:
            print(f"Fehler während das Vorhersage des Dateis  {audio}: {e}")
            return "Fehler"
    
    def callback(indata, frames, time, status):
        nonlocal prediction
        if status:
            print(status)
        
        # Überprüfen, ob das Eingangsarray signifikant ist
        if np.max(np.abs(indata)) < 0.02:  # Schwellenwert für Stille
            prediction = "Unbekannt"
            return
        
        # Engabe in Mono unwandeln
        audio = indata[:, 0]
        # Vorhersage
        try:
            prediction =predict_speaker(model,audio, scaler)
            # if prediction == "unknown":
            #     print("erkannt : unbekannt ")
            # else:
            #     print(f"erkannt : {prediction}")
        except Exception as e:
            print(f"Erreur : {e}")

    with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=16000):
       print("Jetzt  können sie sprechen...")
       try:
           while True:
               if prediction is not None:
                   print(f"erkannt : {prediction}")
                   prediction=None
               time.sleep(.5)  # Wartezeit
        
       except KeyboardInterrupt:
           print("Erkennung beendet.")
 


def live_audio_analysis_svm(model, scaler, label_map, segment_length=0.1, sr=16000, window_size=3):
    """
    Führt eine Live-Sprechererkennung mit einem SVM-Modell durch und glättet die Ergebnisse.

    Parameter:
    - model (sklearn.SVC): Das trainierte SVM-Modell
    - scaler (StandardScaler): Der für das Modell verwendete Scaler
    - label_map (dict): Mapping von Sprechernamen zu Labels
    - segment_length (float): Länge der Segmente in Sekunden
    - sr (int): Sampling-Rate
    - window_size (int): Fenstergröße für die Glättung der Ergebnisse
    """
    buffer = queue.Queue()
    label_to_name = {v: k for k, v in label_map.items()}
    segment_samples = int(segment_length * sr)
    current_audio = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time, status):
        if status:
            print(status)
        # Aufgenommene Audiodaten in den Puffer speichern
        buffer.put(indata[:, 0])

    def format_time(seconds):
        """Hilfsfunktion, um Sekunden in mm:ss:msms-Format zu formatieren."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{m:02}:{s:02}:{ms:03}"

    with sd.InputStream(samplerate=sr, channels=1, callback=callback, blocksize=segment_samples):
        print("Live-Sprechererkennung gestartet. Drücke STRG+C, um zu beenden.")

        try:
            original_results = []
            start_time = 0

            while True:
                # Daten aus dem Puffer lesen und in den aktuellen Audio-Stream einfügen
                while not buffer.empty():
                    current_audio = np.append(current_audio, buffer.get())

                # Verarbeitung in segmentierten Blöcken
                while len(current_audio) >= segment_samples:
                    segment = current_audio[:segment_samples]
                    current_audio = current_audio[segment_samples:]

                    # MFCCs extrahieren und skalieren
                    mfccs = extract_features(segment, sr)  # Deine existierende Funktion für Feature-Extraktion
                    features_scaled = scaler.transform([mfccs])

                    # Vorhersage mit SVM
                    predicted_label = model.predict(features_scaled)[0]
                    original_results.append(predicted_label)

                    # Ausgabe des aktuellen Segments
                    speaker_name = label_to_name.get(predicted_label, "Unbekannt")
                    print(f"[{format_time(start_time)} - {format_time(start_time + segment_length)}] {speaker_name}")
                    start_time += segment_length

                # Ergebnisse glätten und anzeigen (alle x Sekunden)
                if len(original_results) >= window_size:
                    smoothed_results = []
                    padding = (window_size - 1) // 2
                    padded_results = [None] * padding + original_results + [None] * padding

                    for i in range(len(original_results)):
                        window = padded_results[i:i + window_size]
                        window = [label for label in window if label is not None]
                        most_common = max(set(window), key=window.count) if window else None
                        smoothed_results.append(most_common)

                    # Letzte geglättete Ergebnisse ausgeben
                    if smoothed_results:
                        speaker_name = label_to_name.get(smoothed_results[-1], "Unbekannt")
                        print(f"Glättung: {speaker_name}")

                # Wartezeit für Live-Streaming
                sd.sleep(int(segment_length * 1000))

        except KeyboardInterrupt:
            print("Erkennung beendet.")

def audio_to_text(audio_path):
    """
    Konvertiert eine Audiodatei in Text.
    
    :param audio_path: Der Dateipfad zur Audiodatei
    :return: Der erkannte Text (oder eine Fehlermeldung)
    """
    recognizer = sr.Recognizer()
    
    try:
        # Laden der Audiodatei
        with sr.AudioFile(audio_path) as source:
            print("Lade Audio...")
            audio_data = recognizer.record(source)  # Lesen der gesamten Audiodatei
            
        # Umwandeln von Audio zu Text
        print("Erkenne Text...")
        text = recognizer.recognize_sphinx(audio_data, language="de-DE")  # Deutsch
        if text:
            print("Erkannter Text:", text)
        else:
            print("Es wurde kein Text erkannt.")
        return text
    
    except sr.UnknownValueError:
        print("Die Sprache konnte nicht erkannt werden.")
        return None
    except sr.RequestError as e:
        print(f"Fehler bei der Anfrage an die Speech Recognition API: {e}")
        return None
    except FileNotFoundError:
        print(f"Die Datei {audio_path} wurde nicht gefunden.")
        return None


def segment_and_analyze_with_svm(audio_file, model, scaler, label_map, segment_length=0.25, sr=22050):
    """
    Segment the audio file into smaller chunks, classify each segment, and smooth predictions using HMM.
    
    Parameters:
        audio_file (str): Path to the audio file.
        model: Trained SVM model.
        scaler: StandardScaler instance used to normalize features.
        segment_length (float): Length of each segment in seconds.
        sr (int): Sampling rate for audio processing.

    Returns:
        transcript (list): List of recognized speaker intervals with timestamps.
    """
    # von "label":key (bisheriges forms von label_map) zu key:"label"
    label_map = {v: k for k, v in label_map.items()}
    
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"The file {audio_file} does not exist.")

    # Load the audio file
    audio, _ = librosa.load(audio_file, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples

    print(f"\nAnalyzing {os.path.basename(audio_file)}...")
    print(f"Segment length: {segment_length}s")

    # Initialize arrays to store results
    original_results = []

    # Segment the audio file and classify each segment
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]

        if len(segment) < segment_samples * 0.8:  # Skip incomplete segments
            break

        # Extract features and predict speaker
        features = extract_features(segment, sr)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        original_results.append(prediction)

    # Apply Hidden Markov Model for smoothing
    #smoothed_results = smooth_with_hmm(np.array(original_results), len(label_to_name))
    smoothed_results =  smooth_with_moving_average(np.array(original_results),3)

    # Generate speaker intervals with timestamps
    transcript = []
    current_speaker = None
    segment_start_time = 0.0
    
    for i, speaker_label in enumerate(smoothed_results):
        speaker_name = label_map.get(speaker_label, "Unknown")
        if speaker_name != current_speaker:
            if current_speaker is not None:
                segment_end_time = i * segment_length
                transcript.append((current_speaker, segment_start_time, segment_end_time))
                print(f"[{segment_start_time:.2f}s - {segment_end_time:.2f}s] {current_speaker}")
            current_speaker = speaker_name
            segment_start_time = i * segment_length

    # Add the last speaker interval
    if current_speaker is not None:
        segment_end_time = num_segments * segment_length
        transcript.append((current_speaker, segment_start_time, segment_end_time))
        print(f"[{segment_start_time:.2f}s - {segment_end_time:.2f}s] {current_speaker}")

    return transcript


def smooth_with_hmm(predictions, num_classes):
    """
    Smooth predictions using a Gaussian Hidden Markov Model (HMM).

    Parameters:
        predictions (np.array): Continuous numerical predictions (e.g., probabilities or features).
        num_classes (int): Number of unique speaker classes.

    Returns:
        np.array: Smoothed predictions.
    """
    from hmmlearn import hmm
    import numpy as np

    # Reshape predictions to match HMM input shape
    reshaped_predictions = predictions.reshape(-1, 1)

    # Initialize Gaussian HMM
    model = hmm.GaussianHMM(n_components=num_classes, covariance_type="diag", n_iter=100,init_params='', random_state=42)

    # Set realistic start probabilities
    model.startprob_ = np.full(num_classes, 1.0 / num_classes)

    # Set realistic transition probabilities
    transition_prob = 0.1
    model.transmat_ = np.full((num_classes, num_classes), transition_prob / (num_classes - 1))
    np.fill_diagonal(model.transmat_, 1.0 - transition_prob)

    # Fit the HMM model on the data (unsupervised)
    try:
        model.fit(reshaped_predictions)
    except ValueError as e:
        print(f"HMM fit error: {e}")
        return predictions  # Return the original predictions if HMM fails to fit

    # Decode the smoothed predictions using Viterbi algorithm
    smoothed_predictions = model.predict(reshaped_predictions)
    
    return smoothed_predictions

def smooth_with_moving_average(predictions, window_size=3):
    smoothed_predictions = uniform_filter1d(predictions, size=window_size, mode='nearest')
    return np.round(smoothed_predictions).astype(int)


def plot_speaker_timeline(transcript, audio_file):
    """
    Plots a speaker timeline with consistent colors for each label.

    Parameters:
        transcript: List of tuples (speaker, start_time, end_time).
        audio_file: Path to the audio file (used to get the duration).
    """
    # Load audio to determine the duration
    audio, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio) / sr

    # Eigene Farben definieren
    mycolors = ["blue", "orange", "red","pink", "yellow", "green", "gray"]

    # Einzigartige Sprecher extrahieren
    speakers = sorted(set([t[0] for t in transcript]))

    # Dictionary für konsistente Farbzuordnung erstellen
    speaker_colors = {speaker: mycolors[i % len(mycolors)] for i, speaker in enumerate(speakers)}

    # Plot the timeline
    fig, ax = plt.subplots(figsize=(12, 3))
    for speaker, start, end in transcript:
        ax.plot([start, end], [1, 1], color=speaker_colors[speaker], linewidth=6, label=speaker)

    # Remove duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))
    ax.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="upper right")

    plt.xlabel("Time (s)")
    plt.title("Speaker Timeline")
    plt.yticks([])
    plt.show()
import librosa, queue, time, speech_recognition as sr, optuna
import numpy as np, pandas as pd, os, sounddevice as sd, soundfile as sf,matplotlib.pyplot as plt, seaborn as sns, warnings
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV,cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    )
from sklearn.datasets import load_iris
from scipy.stats import uniform
from scipy.ndimage import uniform_filter1d
from hmmlearn import hmm
from matplotlib import cm
from collections import deque, Counter
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Warnungen ignorieren
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Funktion zur Extraktion von MFCC-Features aus Audiodaten
def extract_features(audio, sr=22050, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40, max_pad_len=400):
    
    """
    Extrahiert MFCC (Mel-Frequency Cepstral Coefficients) aus Audiodaten.

    Eingabe:
    - audio: Audiosignal als numpy-Array.
    - sr (int): Sampling-Rate (Standard: 22050 Hz).
    - n_mfcc (int): Anzahl der zu extrahierenden MFCCs (Standard: 13).
    - n_fft (int): Fensterlänge für die FFT (Standard: 1024).
    - hop_length (int): Schrittweite zwischen Fenstern (Standard: 512).
    - n_mels (int): Anzahl der Mel-Bänder (Standard: 40).
    - max_pad_len (int): Maximale Länge für die Padding.

    Ausgabe:
    - numpy.array: Flaches Array der MFCCs, das als Eingabe für die SVM verwendet werden kann.
    """
    
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

def process_file(file_path, label, sr=22050):
    """
    Extrahiert Merkmale (MFCCs) aus einer Audiodatei und augmentiert die Daten.

    Eingabe:
    - file_path (str): Pfad zur Audiodatei.
    - label (int): Label der Datei.
    - sr (int): Sampling-Rate (Standard: 22050 Hz).

    Ausgabe:
    - features (list): Liste der extrahierten Merkmale.
    - labels (list): Liste der Labels, die den Merkmalen entsprechen.
    """
    
    try:
        audio, sr = librosa.load(file_path,  sr=22050) # Lower sample rate for speed
        features = [extract_features(audio, sr)]  # Original
        augmented_audio = augment_audio(audio)
        features.append(extract_features(augmented_audio, sr))  # Augmented
        labels = [label] * len(features)
        return features, labels
    except Exception as e:
        print(f"Fehler während der Bearbeitung des Dateien {file_path}: {e}")
        return [], []

#Merkmale eine Einzelne Datei EXtrahieren.
def process_file_with_seg(file_path, label, segment_length=0.1, sr=22050):
    """
    Segmentiert eine Audiodatei in kleinere Abschnitte und extrahiert Merkmale aus jedem Segment.

    Eingabe:
    - file_path (str): Pfad zur Audiodatei.
    - label (int): Label der Datei.
    - segment_length (float): Länge jedes Segments in Sekunden.
    - sr (int): Sampling-Rate (Standard: 22050 Hz).

    Ausgabe:
    - features (list): Liste der extrahierten Merkmale aus jedem Segment.
    - labels (list): Liste der Labels, die den Segmenten entsprechen.
    """
    
    try:
        audio, sr = librosa.load(file_path, sr=22050)  # Lower sample rate for speed
        # features = [extract_features(audio, sr)]  # Original
        # augmented_audio = augment_audio(audio)
        features=[]
        labels=[] 
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
    """
    Lädt Audiodaten und extrahiert die entsprechenden Merkmale und Labels.

    Eingabe:
    - path (str): Pfad zum Ordner mit den Audiodateien.
    - label_map (dict): Mapping von Sprechernamen zu Labels.
    - segment_length (float): Länge der Segmente in Sekunden.
    - sr (int): Sampling-Rate (Standard: 22050 Hz).

    Ausgabe:
    - numpy.array: Merkmale der Audiodaten.
    - numpy.array: Labels der Audiodaten.
    """
    print("Lade Daten...")
    features, labels = [], []
    
    entscheidung = input("Trainingsdaten Segmentieren? (ja/nein): ").strip().lower()
         
    if entscheidung == "ja":
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
            
            
            results = Parallel(n_jobs=-1)(delayed(process_file_with_seg) (file, label_map[speaker],segment_length, sr=22050) for file in files)
        
            for f, l in results:
                features.extend(f)
                labels.extend(l)
                #print(f"  - {len(f)} Features für Datei hinzugefügt. Aktuelle Labels: {Counter(labels)}")               
    elif entscheidung == "nein":
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
            
            results = Parallel(n_jobs=-1)(delayed(process_file) (file, label_map[speaker], sr=22050) for file in files)
        
            for f, l in results:
                features.extend(f)
                labels.extend(l)
                #print(f"  - {len(f)} Features für Datei hinzugefügt. Aktuelle Labels: {Counter(labels)}")
    else:
            print("Ungültige Eingabe! Bitte 'ja' oder 'nein' eingeben.")

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Es wurde kein Daten wegen Labels in Datei gefunden")
    
    return np.array(features), np.array(labels)

# Funktionen zur Erstellung und Suche nach besten Hyperparametern
#Hyperparameter-Tunning mit Randomize-search
def randomized_search_svm(X_train, y_train, n_iter=10, random_state=42):
    """
    Hyperparameter Optimierug mit RandomizedSearchCV.

    Eingabeparameter:
    - X_train (numpy.array): Trainingsdaten (Merkmale).
    - y_train (numpy.array): Trainingslabels.
    - n_iter (int): Anzahl der Iterationen für die Suche
    - random_state (int): Zufallsseed für Reproduzierbarkeit

    Ausgabe:
    - best_estimator_: Das beste SVM-Modell.
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
                                           n_iter=n_iter, cv=StratifiedKFold(n_splits=5), verbose=1, n_jobs=-1,
                                           random_state=random_state, return_train_score=True)
    
    # Führe das RandomizedSearch durch
    randomized_search.fit(X_train, y_train)
    print("fitting abgeschlossen")
    print(f"Beste Parameter: {randomized_search.best_params_}")
    print(f"Beste Kreuzvalidierungsgenauigkeit: {randomized_search.best_score_ * 100:.2f}%")
    
    return randomized_search.best_estimator_


def objective(trial,X_train, y_train):
    """
    Optuna-Ziel-Funktion für die Hyperparameter-Optimierung.
    """
    # Definieren der Hyperparameterbereiche
    with parallel_backend("threading"):
        C = trial.suggest_float("C", 1, 100, log=True)  # Logarithmischer Bereich
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        gamma = trial.suggest_categorical('gamma', [0.1, 0.01, 'scale','auto'])
        probability = trial.suggest_categorical('probability', [True, False])
        
        # Erstellen eines Modells mit den vorgeschlagenen Hyperparametern abhängig von Kernel
        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5) 
            model = Pipeline([
                ('scaler', StandardScaler()),  # Features skalieren
                ('svm', SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, class_weight='balanced',probability=probability))
            ])
        else :
            model = Pipeline([
                ('scaler', StandardScaler()),  # Features skalieren
                ('svm', SVC(C=C, kernel=kernel, gamma=gamma, class_weight='balanced',probability=probability))
            ])
       
        

        # 5-fache Kreuzvalidierung zur Bewertung des Modells
        score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=4)
        return score.mean()

# SVM Modell trainieren
def train_svm_model_optuna(path, label_map,segment_length, sr=22050):
    """
    Hyperparameter-Optimierung mit Optuna

    Eingabeparameter:
    - path (str): Pfad zu den Audiodaten.
    - label_map (dict): Mapping von Sprechernamen zu Labels.
    - segment_length (float): Länge der Segmente in Sekunden.
    - sr (int): Sampling-Rate 

    Ausgabe:
    - best_model: Das trainierte und optimierte SVM-Modell.
    - myScaler: Der Skaler, der für die Transformation der Merkmale verwendet wurde.
    """
    
    # Beste gefundene Parameter von Rndomizesearch mit 50 fits als Startwerte
    initial_params = {'C': 3.845401188473625, 'degree': 5, 'gamma': 0.1, 'kernel': 'linear', 'probability': True}

    X, y = load_data(path,label_map,segment_length,  sr)
    X, y = shuffle(X,y,random_state=42)
    
    print(f"Feature-Shape: {X.shape}, Label-Shape: {y.shape}")
    print(f"Label-Verteilung: {Counter(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    #print(f"Unique classes in y_train: {np.unique(y_train)}")
    #print(f"y_train counts: {np.bincount(y_train)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    myScaler=scaler

    """pca = PCA(n_components=500)  # Auf 500 Dimensionen reduzieren
    myScaler=pca
    X_train= pca.fit_transform(X_train)
    X_test = pca.transform(X_test)"""
    
    #dies führt zur Overfitting
    """selector = SelectKBest(f_classif, k=1000)  # Wähle die 1000 besten Features
    X_train= selector.fit_transform(X_train, y_train)
    X_test =selector.transform(X_test)
    myScaler=selector"""

    # Optuna-Studie erstellen und optimieren
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5,interval_steps=2)
    study = optuna.create_study(direction="maximize",pruner=pruner)  # Ziel ist, die Genauigkeit zu maximiereN
    
    #Füge die besten bekannten Werte aus RandomizeSearch als ersten Versuch hinzu
    study.enqueue_trial(initial_params)
    
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=15, n_jobs=2)  # 15 Iterationen mit Parallelisierung
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

def train_svm_model(path, label_map, segment_length=0.1, sr=22050):
    """
    Ziel:
    Trainiert ein SVM-Modell mithilfe von RandomizedSearchCV.

    Eingabeparameter:
    - path (str): Pfad zu den Audiodaten.
    - label_map (dict): Mapping von Sprechernamen zu Labels.
    - segment_length (float): Länge der Segmente in Sekunden.
    - sr (int): Sampling-Rate

    Ausgabe:
    - best_model: Das trainierte und optimierte SVM-Modell.
    - scaler: Der Skaler, der für die Transformation der Merkmale verwendet wurde.
    """
    X, y = load_data(path,label_map, segment_length, sr)
    X, y = shuffle(X,y,random_state=42)
    
    print(f"Feature-Shape: {X.shape}, Label-Shape: {y.shape}")
    print(f"Label-Verteilung: {Counter(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    #print(f"Unique classes in y_train: {np.unique(y_train)}")
    #print(f"y_train counts: {np.bincount(y_train)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # SVM-Modell mit RandomizedSearchCV trainieren
    start_time = time.time()
    best_model = randomized_search_svm(X_train, y_train)
    end_time = time.time()
    print(f"Optimierung mit Randomize abgeschlossen in {end_time - start_time:.2f} Sekunden.")
    
    best_model.fit(X_train, y_train)
    accuracy =best_model.score(X_test, y_test)
    print(f"Genauigkeit des besten SVM-Modells: {accuracy*100:.2f}%")
    
    #Confusion Matrix
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred,label_map)
    
    # Evaluieren des Modells
    evaluate_model(y_test, y_pred,label_map)
    
    # Learningskurve zeigen
    plot_learning_curve(best_model, X_train, y_train) #plot der learning Kurve
    
    return best_model, scaler

def evaluate_model(y_test, y_pred,label_map):
    """
    Berechnet mehrere Metriken zur Bewertung eines Klassifikationsmodells.

    Eingabeparameter:
    - y_test (numpy.array): Wahre Labels der Testdaten.
    - y_pred (numpy.array): Vorhergesagte Labels des Modells.
    - label_map (dict): Mapping von Labels zu Klassen.

    Ausgabe:
    - Konsolenausgabe mit verschiedenen Bewertungsmetriken.
    """
    
    print(classification_report(y_test, y_pred, target_names=label_map))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Genauigkeit: {accuracy * 100:.2f}%")
    print(f"Präzision (Weighted): {precision * 100:.2f}%")
    print(f"Recall (Weighted): {recall * 100:.2f}%")
    print(f"F1-Score (Weighted): {f1 * 100:.2f}%")
    
#learning Kurve
def plot_learning_curve(model, X_train, y_train):
    """
    Zeigt die Lernkurve eines Modells, um Overfitting oder Underfitting zu analysieren.

    Eingabeparameter:
    - model: Das zu bewertende Modell.
    - X_train (numpy.array): Trainingsdaten.
    - y_train (numpy.array): Trainingslabels.

    Ausgabe:
    - Ein Diagramm mit Trainings- und Testgenauigkeiten in Abhängigkeit von der Trainingsgröße.
    """
    
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

#Confusion Matrix visualisieren
def plot_confusion_matrix(y_test, y_pred,label_map):
    """
    Visualisiert die Confusion-Matrix für die Modellbewertung.

    Eingabeparameter:
    - y_test (numpy.array): Wahre Labels der Testdaten.
    - y_pred (numpy.array): Vorhergesagte Labels des Modells.
    - label_map (dict): Mapping von Labels zu Klassen. 

    Ausgabe:
    - Ein Heatmap-Diagramm der Confusion-Matrix.
    """
    
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
    """
    Ziel:
    Nimmt eine Audiodatei als Eingabe, extrahiert Merkmale und sagt voraus, welcher Sprecher es ist.

    Eingabeparameter:
    - model: Das trainierte SVM-Modell.
    - audio_file (str): Pfad zur Audiodatei.
    - scaler: Der Skaler, der für die Merkmalsnormalisierung verwendet wurde.

    Ausgabe:
    - speaker (str): Name des vorhergesagten Sprechers.
    """
    try:
        audio, sr = librosa.load(audio_file, sr=16000)  # Lower sample rate for speed
        features = extract_features(audio, sr)
        # print(f"Extrahierte Eigenschaften für Vorhersage: {features}") 
        features = scaler.transform([features])
        # print(f"Extrahierte Eigenschaften für Vorhersage nach Scaler Transform: {features}") 
        prediction = model.predict(features)[0]
        speaker = ["Biden", "Moderator" ,"Trump","Unbekannt"][prediction]
        print(f"File: {audio_file}, Predicted Speaker: {speaker}")
        return speaker
    except Exception as e:
        print(f"Fehler während das Vorhersage des Dateis  {audio_file}: {e}")
        return "Fehler"

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

def audio_to_text2(audio_path,language):
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
        text = recognizer.recognize_sphinx(audio_data, language)
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

import os
import librosa
import numpy as np
import speech_recognition as sr

def audio_to_text(audio_file, transcript, language="en-US"):
    """
    Ziel:
    Verwendet das segmentierte `transcript`, um nur relevante Segmente zu analysieren und wandelt sie in Text um.

    Eingabeparameter:
    - audio_file (str): Pfad zur Audiodatei.
    - transcript (list): Liste mit Sprecher-Intervallen [(Sprecher, Startzeit, Endzeit)].
    - language (str): Sprachcode für die Spracherkennung (Standard: "en-US" für Englisch).

    Ausgabe:
    - Speichert das erkannte Transkript in einer `.txt`-Datei mit demselben Namen wie `audio_file`.
    - Gibt das erkannte Transkript als Zeichenkette zurück.
    """
    recognizer = sr.Recognizer()

    # Speicherpfad für das vollständige Transkript
    output_file = os.path.splitext(audio_file)[0] + "_full_transcript.txt"

    # Laden der gesamten Audiodatei mit librosa
    audio, sr_rate = librosa.load(audio_file, sr=16000)

    full_transcript = []

    for speaker, start_time, end_time in transcript:
        # Umwandlung von Zeit (Sekunden) in Sample-Indizes
        start_sample = int(start_time * sr_rate)
        end_sample = int(end_time * sr_rate)

        # Audio-Segment extrahieren
        segment_audio = audio[start_sample:end_sample]

        # Konvertiere Segment in eine WAV-Datei für SpeechRecognition
        temp_wav = "temp_segment.wav"
        #librosa.output.write_wav(temp_wav, segment_audio, sr_rate)
        sf.write(temp_wav, segment_audio, sr_rate)

        try:
            with sr.AudioFile(temp_wav) as source:
                #print(f" Processing Segment [{start_time:.2f}s - {end_time:.2f}s] for {speaker}...")
                audio_data = recognizer.record(source)

                # Speech Recognition durchführen
                text = recognizer.recognize_sphinx(audio_data, language)

                if text:
                    #print(f"Recognized for {speaker}: {text}")
                    full_transcript.append(f"[{start_time:.2f}s - {end_time:.2f}s] : {speaker} \"{text}\"")
                else:
                    #print(f" No speech detected for {speaker}.")
                    full_transcript.append(f"[{start_time:.2f}s - {end_time:.2f}s] : {speaker} \"(No Recognition)\"")

        except sr.UnknownValueError:
            print(f" No understandable speech detected in segment [{start_time:.2f}s - {end_time:.2f}s].")
            full_transcript.append(f"[{start_time:.2f}s - {end_time:.2f}s] : {speaker} \"(Unintelligible)\"")
        except sr.RequestError as e:
            print(f"API request error: {e}")

    # Speichern des Transkripts
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(full_transcript))

    print(f"\n Full transcript saved at: {output_file}")

    return full_transcript


def segment_and_analyze_with_svm(audio_file, model, scaler, label_map, segment_length=0.25, sr=22050):
    """
    Segmentiert eine Audiodatei in kleine Blöcke, klassifiziert jedes Segment mit einem SVM-Modell 
    und glättet die Vorhersagen mit einem Moving Average.

    Eingabeparameter:
    - audio_file (str): Pfad zur Audiodatei.
    - model: Trainiertes SVM-Modell.
    - scaler: StandardScaler-Instanz für die Normalisierung der Merkmale.
    - label_map (dict): Mapping von Labels zu Sprechernamen.
    - segment_length (float): Länge jedes Segments in Sekunden (Standard: 0.25s).
    - sr (int): Sampling-Rate für die Audioverarbeitung (Standard: 22050 Hz).

    Ausgabe:
    - transcript (list): Liste mit erkannten Sprecher-Intervallen und Zeitstempeln 'für Gantt-Diagramm erforderlich'.
    - speichert die Ergebnisse in einer `.txt`-Datei mit demselben Namen wie `audio_file`.
    """
    # von "label":key (bisheriges forms von label_map) zu key:"label"
    label_map = {v: k for k, v in label_map.items()}
    
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"The file {audio_file} does not exist.")

    audio, _ = librosa.load(audio_file, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples

    print(f"\nAnalyzing {os.path.basename(audio_file)}...")
    print(f"Segment length: {segment_length}s")

    # list zum sppeicherung der Ergebnisse
    original_results = []

    # Segmentierung und Klassifizierung jedes Segments
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]

        if len(segment) < segment_samples * 0.8:  # Skip incomplete segments
            break

        features = extract_features(segment, sr)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        original_results.append(prediction)

    # Anwenden eines Moving Average zur Glättung der Vorhersagen
    smoothed_results =  smooth_with_moving_average(np.array(original_results),3)

    # Erstellen der Sprecherintervalle mit Zeitstempeln
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

    # Letztes Sprecherintervall hinzufügen
    if current_speaker is not None:
        segment_end_time = num_segments * segment_length
        transcript.append((current_speaker, segment_start_time, segment_end_time))
        print(f"[{segment_start_time:.2f}s - {segment_end_time:.2f}s] {current_speaker}")
        
    # Ergebnis in Datei speichern
    output_file = os.path.splitext(audio_file)[0] + "_ausgabe.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join([f"[{start:.2f}s - {end:.2f}s] : {speaker}" for speaker, start, end in transcript]))

    return transcript

def smooth_with_hmm(predictions, num_classes):
    """
    Glättet Klassifikationsvorhersagen mit einem Hidden Markov Model (HMM),
    um unerwartete Wechsel zu reduzieren.

    Eingabeparameter:
    - predictions (np.array): Array mit numerischen Vorhersagen (z. B. Sprecher-Labels).
    - num_classes (int): Anzahl der einzigartigen Sprecherklassen.

    Ausgabe:
    - np.array: Geänderte Vorhersagen nach Anwendung des HMM.
    """
    # Umformen der Vorhersagen für das HMM-Modell
    reshaped_predictions = predictions.reshape(-1, 1)

    # Initialisierung des HMM-Modells
    model = hmm.GaussianHMM(n_components=num_classes, covariance_type="diag", n_iter=100, init_params='', random_state=42)

    # Setzen von Startwahrscheinlichkeiten
    model.startprob_ = np.full(num_classes, 1.0 / num_classes)

    # Übergangswahrscheinlichkeiten realistisch setzen
    transition_prob = 0.1
    model.transmat_ = np.full((num_classes, num_classes), transition_prob / (num_classes - 1))
    np.fill_diagonal(model.transmat_, 1.0 - transition_prob)

    # Training des HMM-Modells (ungelernt)
    try:
        model.fit(reshaped_predictions)
    except ValueError as e:
        print(f"HMM-Fehler: {e}")
        return predictions  # Rückgabe der Originalvorhersagen, falls das HMM fehlschlägt

    # Vorhersagen mit dem Viterbi-Algorithmus glätten
    smoothed_predictions = model.predict(reshaped_predictions)
    
    return smoothed_predictions

def smooth_with_moving_average(predictions, window_size=3):
    """
    Wendet einen gleitenden Mittelwertfilter an, um Sprünge in der Sprecherklassifikation zu reduzieren.

    Eingabeparameter:
    - predictions (np.array): Array mit den ursprünglichen Vorhersagen.
    - window_size (int): Größe des Fensters für den gleitenden Durchschnitt.

    Ausgabe:
    - np.array: Geänderte Vorhersagen nach Anwendung des Moving Average.
    """
    smoothed_predictions = uniform_filter1d(predictions, size=window_size, mode='nearest')
    return np.round(smoothed_predictions).astype(int)

def plot_speaker_timeline(transcript, audio_file):
    """
    Erstellt eine Zeitleiste mit den Sprechern und ihrer Sprechdauer basierend auf der Segmentierung.

    Eingabeparameter:
    - transcript (list): Liste mit Sprecher-Intervallen im Format (Sprecher, Startzeit, Endzeit).
    - audio_file (str): Pfad zur Audiodatei, um die Dauer des Audios zu bestimmen.

    Ausgabe:
    - Ein Diagramm mit den Sprechern und ihrer Sprechdauer.
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

    fig, ax = plt.subplots(figsize=(12, 3))
    for speaker, start, end in transcript:
        ax.plot([start, end], [1, 1], color=speaker_colors[speaker], linewidth=6, label=speaker)

    # Doppelte Labels in der Legende entfernen
    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))
    ax.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="upper right")

    plt.xlabel("Time (s)")
    plt.title("Speaker Timeline")
    plt.yticks([])
    plt.show()
    
def plot_speaker_Gantt(transcript, audio_file):
    """
    Erstellt ein Gantt-Diagramm mit den Sprechern und ihrer Sprechdauer basierend auf der Segmentierung.

    Eingabeparameter:
    - transcript (list): Liste mit Sprecher-Intervallen im Format (Sprecher, Startzeit, Endzeit).
    - audio_file (str): Pfad zur Audiodatei, um die Dauer des Audios zu bestimmen.

    Ausgabe:
    - Ein Gantt-Diagramm mit den Sprechern und ihrer Sprechdauer.
    """
    # Load audio to determine the duration
    audio, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio) / sr

    # Eigene Farben definieren
    mycolors = ["blue", "orange", "red", "pink", "yellow", "green", "gray"]

    # Einzigartige Sprecher extrahieren
    speakers = sorted(set([t[0] for t in transcript]))

    # Dictionary für konsistente Farbzuordnung erstellen
    speaker_colors = {speaker: mycolors[i % len(mycolors)] for i, speaker in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 6))
    for speaker, start, end in transcript:
        ax.barh(speaker, end - start, left=start, color=speaker_colors[speaker])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Speakers")
    plt.title("Speaker Timeline (Gantt Chart)")
    plt.show()

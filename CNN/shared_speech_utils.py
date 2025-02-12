import os
import queue
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import optuna
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow Logging konfigurieren
tf.get_logger().setLevel('ERROR')

def extract_mfccs(audio, sr=22050, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40, max_pad_len=400):
    """
    Extrahiert MFCC-Features aus Audiodaten mit fester Länge.

    Parameter:
    - audio (np.ndarray): Audiodaten
    - sr (int): Sampling-Rate
    - n_mfcc (int): Anzahl der MFCC-Features
    - n_fft (int): FFT-Fenstergröße
    - hop_length (int): Schrittweite für FFT
    - n_mels (int): Anzahl der Mel-Bänder
    - max_pad_len (int): Maximale Auffülllänge

    Rückgabe:
    - np.ndarray: Gepaddete MFCC-Matrix
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr // 2)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

def load_training_data(path, label_map):
    """
    Lädt Trainingsdaten aus einem Verzeichnis mit Unterordnern, die nach den Sprechern benannt sind.

    Parameter:
    - path (str): Pfad zum Datensatz
    - label_map (dict): Mapping von Sprechernamen zu Labels

    Rückgabe:
    - Tuple[np.ndarray, np.ndarray]: Features (X) und Labels (y)
    """
    X = []
    y = []

    for speaker, label in label_map.items():
        speaker_path = os.path.join(path, speaker)
        if not os.path.exists(speaker_path):
            print(f"Warnung: Ordner {speaker_path} existiert nicht.")
            continue

        audio_files = [file for file in os.listdir(speaker_path) if file.endswith(".wav") or file.endswith(".mp3")]
        for file in audio_files:
            try:
                audio, sr = librosa.load(os.path.join(speaker_path, file), sr=22050)
                mfccs = extract_mfccs(audio, sr)
                X.append(mfccs)
                y.append(label)
            except Exception as e:
                print(f"Fehler beim Laden von {file}: {e}")

    return np.array(X), np.array(y)

def create_cnn_model(input_shape, num_classes):
    """
    Erstellt ein CNN-Modell für die Sprachklassifikation.

    Parameter:
    - input_shape (tuple): Form der Eingabedaten
    - num_classes (int): Anzahl der Klassen

    Rückgabe:
    - tf.keras.Model: Kompiliertes CNN-Modell
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_optimized_cnn(trial, input_shape, num_classes):
    """
    Erstellt ein CNN-Modell mit hyperparametern, die durch Optuna optimiert werden.

    Parameter:
    - trial (optuna.trial.Trial): Optuna Trial-Objekt
    - input_shape (tuple): Form der Eingabedaten
    - num_classes (int): Anzahl der Klassen

    Rückgabe:
    - tf.keras.Model: Kompiliertes CNN-Modell
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Anzahl der Convolutional-Layers optimieren
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)

    for i in range(num_conv_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [8, 16, 32, 64, 128])
        kernel_size = trial.suggest_categorical(f"kernel_size_{i}", [1, 3, 5, 7])
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"))
        model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # Anzahl der Neuronen in Dense Layer optimieren
    dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64, 128])
    model.add(Dense(dense_units, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    # Optimizer optimieren
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    optimizer = getattr(tf.keras.optimizers, optimizer_name)()

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def objective(trial, X_train, y_train, X_test, y_test, input_shape, num_classes, epochs=20, batch_size=16):
    """
    Bewertet ein CNN-Modell mit verschiedenen Hyperparametern und gibt die Accuracy zurück.

    Parameter:
    - trial (optuna.trial.Trial): Optuna Trial-Objekt
    - X_train, y_train, X_test, y_test: Trainings- und Testdaten
    - input_shape (tuple): Form der Eingabedaten
    - num_classes (int): Anzahl der Klassen
    - epochs (int): Anzahl der Trainings-Epochen
    - batch_size (int): Batch-Größe für das Training

    Rückgabe:
    - float: Testgenauigkeit des trainierten Modells
    """
    model = create_optimized_cnn(trial, input_shape, num_classes)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

def train_optimized_model(X, y, num_classes, epochs=20, batch_size=16, n_trials=50):
    """
    Optimiert die Hyperparameter mit Optuna und trainiert das beste Modell.

    Parameter:
    - X (np.ndarray): Feature-Daten
    - y (np.ndarray): Labels
    - num_classes (int): Anzahl der Klassen
    - epochs (int): Anzahl der Trainings-Epochen
    - batch_size (int): Batch-Größe
    - n_trials (int): Anzahl der Optuna-Optimierungsversuche

    Rückgabe:
    - tf.keras.Model: Das beste trainierte CNN-Modell
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, input_shape, num_classes, epochs, batch_size), n_trials=n_trials)

    best_params = study.best_params
    print("Beste Hyperparameter:", best_params)

    # Trainiere das beste Modell erneut mit den besten Parametern
    best_model = create_optimized_cnn(optuna.trial.FixedTrial(best_params), input_shape, num_classes)
    history = best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # Modell speichern
    save_history(history, "CNN/Ausgaben/history_optuna.json")

    return best_model

def train_model(X, y, label_map, epochs=20, batch_size=16):
    """
    Trainiert ein CNN-Modell mit segmentierten Trainingsdaten.

    Parameter:
    - audio_path (str): Pfad zum Datensatz
    - label_map (dict): Mapping von Sprechernamen zu Labels
    - sr (int): Sampling-Rate
    - epochs (int): Anzahl der Trainings-Epochen
    - batch_size (int): Batch-Größe für das Training

    Rückgabe:
    - tf.keras.Model: Trainiertes CNN-Modell
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modell erstellen und trainieren
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_cnn_model(input_shape, num_classes=len(label_map))
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # Modell speichern
    save_history(history, "CNN/Ausgaben/history_standard.json")

    return model

def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

def load_history(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def plot():
    # Laden der gespeicherten Trainingsdaten
    history_standard = load_history("CNN/Ausgaben/history_standard.json")
    history_optuna = load_history("CNN/Ausgaben/history_optuna.json")

    # Zugriff auf Genauigkeitswerte
    acc_standard = history_standard['accuracy']
    val_acc_standard = history_standard['val_accuracy']
    acc_optuna = history_optuna['accuracy']
    val_acc_optuna = history_optuna['val_accuracy']

    # Anzahl der Epochen für die X-Achse
    epochs = range(1, len(acc_standard) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc_standard, 'bo-', label='Genauigkeit (Standard)')
    plt.plot(epochs, val_acc_standard, 'b--', label='Validierungsgenauigkeit (Standard)')
    plt.plot(epochs, acc_optuna, 'ro-', label='Genauigkeit (Optuna)')
    plt.plot(epochs, val_acc_optuna, 'r--', label='Validierungsgenauigkeit (Optuna)')

    # Achsenbeschriftungen
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit')
    plt.title('Tranings- und Validierungsgenauigkeit: Standard vs. Optuna')
    plt.legend()
    plt.grid()

    # Speichern des Diagramms
    plt.savefig("CNN/Ausgaben/plt_vergleich.png")
    plt.show()

def segment_and_analyze_with_output(audio_file, model, label_map, segment_length=0.1, window_size=3, sr=16000, optimiert=False):
    """
    Führt Sprechererkennung auf einer Audiodatei durch und segmentiert die Ergebnisse.
    Die Ergebnisse werden in eine Datei geschrieben, die denselben Namen wie die Eingabedatei trägt.

    Parameter:
    - audio_file (str): Pfad zur Audiodatei
    - model (tf.keras.Model): Trainiertes CNN-Modell
    - label_map (dict): Mapping von Sprechernamen zu Labels
    - segment_length (float): Länge jedes Segments in Sekunden
    - window_size (int): Fenstergröße für die Glättung der Vorhersagen
    - sr (int): Sampling-Rate
    """
    label_to_name = {v: k for k, v in label_map.items()}

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Die Datei {audio_file} existiert nicht.")

    audio, _ = librosa.load(audio_file, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples

    original_results = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]
        mfccs = extract_mfccs(segment, sr)
        mfccs = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(mfccs, verbose=0)
        original_results.append(np.argmax(prediction, axis=1)[0])

    # Glättung der Vorhersagen
    padding = (window_size - 1) // 2
    padded_results = [None] * padding + original_results + [None] * padding
    cleaned_results = []

    for i in range(len(original_results)):
        window = padded_results[i - padding:i + padding]
        window = [label for label in window if label is not None]
        cleaned_results.append(max(set(window), key=window.count) if window else None)

    current_speaker = None
    segment_start_time = 0

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{m:02}:{s:02}:{ms:03}"

    # Ausgabedatei mit demselben Namen wie die Eingabedatei erstellen
    if optimiert:
        modelname = "optuna"
    else:
        modelname = "standard"
    output_file_name = os.path.join("CNN", "Ausgaben", os.path.splitext(os.path.basename(audio_file))[0] + "_" + modelname + "_" + str(window_size) + ".txt")
    with open(output_file_name, 'w') as output_file:
        for i, speaker in enumerate(cleaned_results):
            speaker_name = label_to_name.get(speaker, "Unbekannt")
            if speaker_name != current_speaker:
                if current_speaker is not None:
                    end_time = i * segment_length
                    output_file.write(f"[{format_time(segment_start_time)} - {format_time(end_time)}] {current_speaker}\n")
                current_speaker = speaker_name
                segment_start_time = i * segment_length

        if current_speaker is not None:
            end_time = num_segments * segment_length
            output_file.write(f"[{format_time(segment_start_time)} - {format_time(end_time)}] {current_speaker}\n")

def live_audio_analysis(model, label_map, segment_length=0.1, sr=16000, window_size=3):
    """
    Führt eine Live-Sprechererkennung durch und glättet die Ergebnisse.

    Parameter:
    - model (tf.keras.Model): Das trainierte CNN-Modell
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
                    
                    # MFCCs extrahieren und vorhersagen
                    mfccs = extract_mfccs(segment, sr)
                    mfccs = np.expand_dims(mfccs, axis=0)
                    prediction = model.predict(mfccs, verbose=0)
                    predicted_label = np.argmax(prediction, axis=1)[0]
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
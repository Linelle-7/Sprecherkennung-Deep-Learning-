import os
import numpy as np  # type: ignore
import librosa  # type: ignore
import sounddevice as sd  # type: ignore
import tensorflow as tf  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense  # type: ignore
import time
import queue
import speech_recognition as sr  # type: ignore

# GPU-Speicherwachstum aktivieren
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = ALL, 1 = WARNING, 2 = ERROR, 3 = FATAL
tf.get_logger().setLevel('ERROR')

# Funktion zum Extrahieren von MFCCs mit fester Länge
def extract_mfccs(audio, sr=22050, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=40, max_pad_len=400):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr // 2)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Funktion zum Laden der Trainingsdaten
def load_training_data(path):
    X = []
    y = []
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}  # Neue Labels

    for speaker in label_map.keys():
        speaker_path = os.path.join(path, speaker)
        if not os.path.exists(speaker_path):
            print(f"Warnung: Ordner {speaker_path} existiert nicht.")
            continue

        audio_files = [file for file in os.listdir(speaker_path) if file.endswith(".mp3")]
        for file in audio_files:
            audio, sr = librosa.load(os.path.join(speaker_path, file), sr=22050)
            mfccs = extract_mfccs(audio, sr)
            X.append(mfccs)
            y.append(label_map[speaker])

    X = np.array(X)
    y = np.array(y)
    return X, y

# CNN-Modell erstellen
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Drei Klassen: Biden, Moderator, Trump
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Funktion zum Trainieren des Modells
def train_model(audio_path):
    X, y = load_training_data(audio_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_cnn_model(input_shape)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# Funktion zur segmentweisen Analyse der Audiodatei mit Sprechererkennung
def segment_and_analyze_with_output(audio_file, model, segment_length=0.1, window_size=3, sr=16000):
    label_to_name = {0: "Biden", 1: "Moderator", 2: "Trump"}  # Zuordnung der Labels zu Namen

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Die Datei {audio_file} existiert nicht.")

    audio, _ = librosa.load(audio_file, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples

    print(f"\nAnalyse von {os.path.basename(audio_file)}:")
    print(f"Segmentlänge: {segment_length}s, Fenstergröße: {window_size}")

    original_results = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]
        mfccs = extract_mfccs(segment, sr)
        mfccs = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(mfccs, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        original_results.append(predicted_label)

    padding = (window_size - 1) // 2
    padded_results = [None] * padding + original_results + [None] * padding
    cleaned_results = []

    for i in range(len(original_results)):
        window = padded_results[i:i + window_size]
        window = [label for label in window if label is not None]
        most_common = max(set(window), key=window.count) if window else None
        cleaned_results.append(most_common)

    current_speaker = None
    segment_start_time = 0

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{m:02}:{s:02}:{ms:03}"

    for i, speaker in enumerate(cleaned_results):
        speaker_name = label_to_name.get(speaker, "Unbekannt")
        if speaker_name != current_speaker:
            if current_speaker is not None:
                end_time = i * segment_length
                print(f"[{format_time(segment_start_time)} - {format_time(end_time)}] {current_speaker}")
            current_speaker = speaker_name
            segment_start_time = i * segment_length

    if current_speaker is not None:
        end_time = num_segments * segment_length
        print(f"[{format_time(segment_start_time)} - {format_time(end_time)}] {current_speaker}")

# Hauptprogramm
if __name__ == "__main__":
    audio_path = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf")
    model = train_model(audio_path)

    # Testdatei analysieren
    test_file = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf", "15-17.mp3")
    segment_and_analyze_with_output(test_file, model, window_size=5)
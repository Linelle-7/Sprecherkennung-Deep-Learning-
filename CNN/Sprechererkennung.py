import os
import numpy as np # type: ignore
import librosa # type: ignore
import sounddevice as sd # type: ignore
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense # type: ignore
import time
import queue
import speech_recognition as sr # type: ignore


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = ALL, 1 = WARNING, 2 = ERROR, 3 = FATAL
tf.get_logger().setLevel('ERROR')  # Setzen des Log-Levels von TensorFlow auf ERROR

# Funktion zum Extrahieren von MFCCs mit fester Länge
def extract_mfccs(audio, sr, n_mfcc=13, n_fft=88, hop_length=512, n_mels=40, max_pad_len=400):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Falls die Länge der MFCC-Daten kleiner als max_pad_len ist, auffüllen
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Falls die Länge größer ist, kürzen
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs

# Funktion zum Laden der Trainingsdaten
def load_training_data(path):
    X = []
    y = []
    audio_files = [file for file in os.listdir(path) if file.endswith(".wav")]

    for file in audio_files:
        audio, sr = librosa.load(os.path.join(path, file), sr=None)
        mfccs = extract_mfccs(audio, sr)  # Feste Länge der MFCCs
        X.append(mfccs)

        # Labels setzen (0 für Felix, 1 für Linelle)
        if file.lower().startswith("felix"):  
            y.append(0) 
        elif file.lower().startswith("linelle"):  
            y.append(1)

    X = np.array(X)  # In ein 3D-Array umwandeln für CNN (samples, features, timesteps)
    y = np.array(y)
    
    return X, y


# CNN-Modell erstellen
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicitly add the Input layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Drei Klassen: Felix, Linelle
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Funktion zum Trainieren des Modells
def train_model(audio_path):
    X, y = load_training_data(audio_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])  # Form der Eingabe
    model = create_cnn_model(input_shape)
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    return model

# Funktion zur Spracherkennung in Echtzeit
def recognize_speech(model):
    recognized = None  # Variable zur Speicherung der erkannten Stimme
    audio_buffer = queue.Queue()  # Buffer für die Audiodaten

    def callback(indata, frames, time, status):
        nonlocal recognized  # Nutzung der äußeren Variablen
        if status:
            print(status)
            
        audio_buffer .put(indata.copy()) # Aufgenommene Daten im Buffer speichern

        # Überprüfen, ob das Eingangsarray signifikant ist
        if np.max(np.abs(indata)) < 0.02:  # Schwellenwert für Stille
            recognized = "Unbekannt"
            return

        # MFCCs extrahieren und vorhersagen
        mfccs = extract_mfccs(indata[:, 0], sr=16000)
        mfccs = np.expand_dims(mfccs, axis=0)  # Für das CNN die Dimension anpassen
        prediction = model.predict(mfccs, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)

        # Bestimmen der erkannten Stimme
        if predicted_label[0] == 0:
            recognized = "Felix"
        elif predicted_label[0] == 1:
            recognized = "Linelle"
        else:
            recognized = "Unbekannt"

    with sd.InputStream(samplerate=16000, channels=1, blocksize=1024, device=None, callback=callback):
        print(f"Aufnahme läuft. Drücke STRG+C zum Beenden.")
        
        try:
          while True:
            # Ausgabe des Erkennungsergebnisses in regelmäßigen Abständen
            if recognized is not None:
                print(f"Erkannt: {recognized}")
                recognized = None  # Reset der erkannten Stimme
            time.sleep(1)  # Wartezeit von 1 Sekunde, um die Ausgabe zu steuern
            
            
        except KeyboardInterrupt:
          print("Erkennung beendet.")
          return audio_buffer
       
def audio_to_text(audio_buffer):
    recognizer = sr.Recognizer()

    # Nehmen Sie die gespeicherten Audiodaten aus dem Buffer und konvertieren Sie sie
    while not audio_buffer.empty():
        audio_data = audio_buffer.get()
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # Konvertieren Sie die numpy-Array-Audiodaten in AudioData für SpeechRecognition
        audio_data = sr.AudioData(audio_np.tobytes(), sample_rate=16000, sample_width=audio_np.itemsize)

        try:
            # Erkennen Sie den Text
            text = recognizer.recognize_sphynx(audio_data, language="de-DE")  # Für Deutsch
            print("Erkannter Text:", text)
        except sr.UnknownValueError:
            print("Die Sprache konnte nicht erkannt werden.")
        except sr.RequestError as e:
            print(f"Fehler bei der Anfrage an Sphynx Recognition API: {e}")

def segment_and_analyze_with_output(audio_file, model, segment_length=0.1, window_size=3, sr=16000):
    # Zuordnung der Labels zu Namen
    label_to_name = {0: "Felix", 1: "Linelle"}

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Die Datei {audio_file} existiert nicht.")

    # Audio laden
    audio, _ = librosa.load(audio_file, sr=sr)
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples

    # Ausgabe des Namens ohne Dateipfad der Audio-Datei
    print(f"\nAnalyse von {os.path.basename(audio_file)}:")

    # Ausgabe der Segement- und Fenstergröße
    print(f"Segmentlänge: {segment_length}s, Fenstergröße: {window_size}")
    
    # Ursprüngliche Ergebnisse
    original_results = []

    # Segmentweise Analyse
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]

        # MFCCs extrahieren und Vorhersage durchführen
        mfccs = extract_mfccs(segment, sr)
        mfccs = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(mfccs, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]

        original_results.append(predicted_label)

    # Padding für Bereinigung
    padding = (window_size - 1) // 2
    padded_results = [None] * padding + original_results + [None] * padding

    # Bereinigte Ergebnisse durch Fensterabstimmung
    cleaned_results = []
    for i in range(len(original_results)):
        window = padded_results[i:i + window_size]
        window = [label for label in window if label is not None]
        if window:
            most_common = max(set(window), key=window.count)
            cleaned_results.append(most_common)
        else:
            cleaned_results.append(None)

    # Sprecherwechsel analysieren und ausgeben
    current_speaker = None
    segment_start_time = 0

    def format_time(seconds):
        """Hilfsfunktion, um Sekunden in mm:ss:msms-Format zu formatieren."""
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

    # Ausgabe des letzten Segments
    if current_speaker is not None:
        end_time = num_segments * segment_length
        print(f"[{format_time(segment_start_time)} - {format_time(end_time)}] {current_speaker}")



# Hauptprogramm
if __name__ == "__main__":
    audio_path = os.path.join(os.path.dirname(__file__), "..", "Stimmen")
    model = train_model(audio_path)
    
    # MP3-Datei verarbeiten
    mp3_file = os.path.join(os.path.dirname(__file__), "..", "Stimmen\Felix_1_1.wav")
    segment_and_analyze_with_output(mp3_file, model, 0.5)
    mp3_file = os.path.join(os.path.dirname(__file__), "..", "Stimmen\Felix_15_2.wav")
    segment_and_analyze_with_output(mp3_file, model, 0.5)
    mp3_file = os.path.join(os.path.dirname(__file__), "..", "Stimmen\Linelle_7_1.wav")
    segment_and_analyze_with_output(mp3_file, model, 0.5)
    mp3_file = os.path.join(os.path.dirname(__file__), "..", "Stimmen\Linelle_10_2.wav")
    segment_and_analyze_with_output(mp3_file, model, 0.5)
    mp3_file = os.path.join(os.path.dirname(__file__), "..", "Stimmen\LinelleNew14.wav")
    segment_and_analyze_with_output(mp3_file, model, 0.5)
    # audio_buffer = recognize_speech(model) # Aufnahme und Spracherkennung in Echtzeit
    # audio_to_text(audio_buffer) # Methode zur Übersetzung der Audio Dateien in Text.
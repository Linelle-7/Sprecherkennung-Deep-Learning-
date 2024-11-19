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

        # Labels setzen (0 für Felix, 1 für Linelle, 2 für Julia)
        if file.lower().startswith("felix"):  
            y.append(0) 
        elif file.lower().startswith("linelle"):  
            y.append(1)
        elif file.lower().startswith("julia"):  
            y.append(2)

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
    model.add(Dense(3, activation='softmax'))  # Drei Klassen: Felix, Linelle, Julia
    
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
        elif predicted_label[0] == 2:
            recognized = "Julia"
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

def process_mp3_file(file_path, model):
    if not os.path.isfile(file_path):
        print("Die angegebene Datei existiert nicht.")
        return

    # MP3 in Audio-Daten und Samplingrate umwandeln
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Segmentieren der Audiodaten in halbe Sekunden
    segment_length = sr // 2  # 0.5 Sekunden pro Segment
    num_segments = len(audio) // segment_length

    print(f"Datei analysieren: {file_path}")
    print(f"Gesamtdauer: {len(audio) / sr:.2f} Sekunden, {num_segments} Segmente werden verarbeitet.")

    current_speaker = None
    previous_speaker = None
    confirmed_speaker = None
    segment_start_time = 0  # Beginn des aktuellen Sprechersegments
    buffer = []  # Buffer für vorläufig erkannte Sprecher
    results = []  # Ergebnisse sammeln

    for i in range(num_segments):
        # Extrahieren des aktuellen Segments
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]

        # MFCCs extrahieren
        mfccs = extract_mfccs(segment, sr)
        mfccs = np.expand_dims(mfccs, axis=0)  # Für das Modell passend vorbereiten

        # Vorhersage
        prediction = model.predict(mfccs, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Sprecher basierend auf der Vorhersage bestimmen
        if predicted_label == 0:
            speaker = "Felix"
        elif predicted_label == 1:
            speaker = "Linelle"
        elif predicted_label == 2:
            speaker = "Julia"
        else:
            speaker = "Unbekannt"

        # Sprecher in den Buffer speichern
        buffer.append(speaker)
        if len(buffer) > 2:  # Buffer auf zwei Elemente beschränken
            buffer.pop(0)

        # Sprecherwechsel prüfen, wenn der Buffer stabil ist
        if len(buffer) == 2 and buffer[0] == buffer[1]:
            current_speaker = buffer[0]

            if current_speaker != confirmed_speaker:
                # Ende des vorherigen Sprechersegments
                if confirmed_speaker is not None:
                    segment_end_time = i * 0.5
                    results.append((confirmed_speaker, segment_start_time, segment_end_time))
                    print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

                # Neuer Sprecher
                confirmed_speaker = current_speaker
                segment_start_time = i * 0.5

    # Restsegment (letzter Sprecherabschnitt)
    if confirmed_speaker is not None:
        segment_end_time = num_segments * 0.5
        results.append((confirmed_speaker, segment_start_time, segment_end_time))
        print(f"{confirmed_speaker}: {segment_start_time:.2f}s - {segment_end_time:.2f}s")

    # Prüfen, ob es einen Restabschnitt gibt
    remaining = len(audio) % segment_length
    if remaining > 0:
        start = num_segments * segment_length
        segment = audio[start:]
        
        # MFCCs extrahieren
        mfccs = extract_mfccs(segment, sr)
        mfccs = np.expand_dims(mfccs, axis=0)

        # Vorhersage
        prediction = model.predict(mfccs, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]

        if predicted_label == 0:
            speaker = "Felix"
        elif predicted_label == 1:
            speaker = "Linelle"
        elif predicted_label == 2:
            speaker = "Julia"
        else:
            speaker = "Unbekannt"

        if speaker == confirmed_speaker:
            # Verlängere das letzte Segment
            results[-1] = (confirmed_speaker, results[-1][1], (num_segments * 0.5 + remaining / sr))
        else:
            # Neues Segment
            start_time = num_segments * 0.5
            end_time = start_time + remaining / sr
            results.append((speaker, start_time, end_time))
            print(f"{speaker}: {start_time:.2f}s - {end_time:.2f}s")

    return results

# Hauptprogramm
if __name__ == "__main__":
    audio_path = os.path.join(os.path.dirname(__file__), "Stimmen")
    model = train_model(audio_path)
    
    # MP3-Datei verarbeiten
    mp3_file = os.path.join(os.path.dirname(__file__), "Stimmen\Felix_1_1.wav")
    process_mp3_file(mp3_file, model)
    mp3_file = os.path.join(os.path.dirname(__file__), "Stimmen\Felix_15_2.wav")
    process_mp3_file(mp3_file, model)
    mp3_file = os.path.join(os.path.dirname(__file__), "Stimmen\Linelle_7_1.wav")
    process_mp3_file(mp3_file, model)
    mp3_file = os.path.join(os.path.dirname(__file__), "Stimmen\Linelle_10_2.wav")
    process_mp3_file(mp3_file, model)
    # audio_buffer = recognize_speech(model) # Aufnahme und Spracherkennung in Echtzeit
    # audio_to_text(audio_buffer) # Methode zur Übersetzung der Audio Dateien in Text.
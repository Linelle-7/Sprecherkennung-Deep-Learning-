
import librosa
import numpy as np
import pandas as pd
import os
from flaml import AutoML
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb

from sklearn.metrics import accuracy_score
import sounddevice as sd
import signal
import time
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Funktion zur Extraktion von MFCC-Features aus Audiodaten
def extract_features(audio, sr, n_mfcc=13, n_fft=416, hop_length=512, n_mels=40, max_pad_len=400):
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
    
    #print("MFCCS's Laenge ist :",len(mfccs))
    """if len(mfccs.shape) == 1:
        mfccs = np.expand_dims(mfccs, axis=0)
    return mfccs"""
    #mfccs_flat = mfccs.flatten()  # Flatten the 2D array into a 1D array
    return mfccs



# Funktion zum Laden der Audiodaten und Extrahieren der zugehörigen Merkmale und Labels
def load_data(audio_folder_path):
    """
    Diese Funktion lädt die Audiodaten und extrahiert Merkmale und Labels.
    Quelle: https://librosa.org/doc/latest/feature.html
    """
    features = []
    labels = []
    audio_list = [file for file in os.listdir(audio_folder_path) if file.endswith(".wav")]
    for file in audio_list:
        audio, sr = librosa.load(os.path.join(audio_folder_path, file), sr=None)
        mfccs_features = extract_features(audio, sr)
        
        #print(f"Extracted features shape for {file}: {mfccs_features.shape}")
        
        features.append(mfccs_features)
        # Label-Zuweisung basierend auf dem Dateinamen
        if file.lower().startswith("felix"):
            labels.append(0)
        elif file.lower().startswith("linelle"):
            labels.append(1)
        #elif file.lower().startswith("linelleNew"):
            #labels.append(2)
        else:
            print(f"Unbekannter Sprecher in Datei: {file}")
    return np.array(features), np.array(labels)




# SVM-Modell mit den angegebenen Parametern erstellen
def create_svm_model(input_shape):
    
    svm_model = SVC()
    svm_model.C=0.7
    svm_model.kernel='poly'
    svm_model.degree=3
    svm_model.gamma=0.01001
    svm_model.coef0=1.0
    svm_model.shrinking=True
    svm_model.probability=False
    svm_model.tol=1e-3
    svm_model.cache_size=250
    svm_model.class_weight=None
    svm_model.verbose=1
    svm_model.max_iter=-1
    svm_model.decision_function_shape='ovr'
    svm_model.break_ties=True
    svm_model.random_state=2
    
    
    return svm_model


def train_svm_model(path):
    
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])  # Form der Eingabe
    svm_model = create_svm_model(input_shape)
    
    """ # Skalieren der Daten
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)"""
    
    # Reshape X_train and X_test to 2D
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1] * X_train.shape[2]  # Flatten each sample
    X_train = X_train.reshape(num_samples, num_features)
    
    num_samples_test = X_test.shape[0]
    num_features_test = X_test.shape[1] * X_test.shape[2]
    X_test = X_test.reshape(num_samples_test, num_features_test)
    
    svm_model.fit(X_train, y_train)
    accuracy = svm_model.score(X_test, y_test)
    print(f"Genauigkeit des SVM-Modells: {accuracy:.2f}")
    
    return svm_model
    





# Funktion zum Trainieren eines Modells mit FLAML
def create_model_flaml(time_budget=60, estimator_list=["svc"], metric="accuracy", seed=42,max_iter= 20 ):
    
    """Erstellt das AutoML-Modell mit FLAML für Klassifikationsaufgaben.
    Quelle: https://github.com/microsoft/FLAML
    """
    automl = AutoML()
    automl_settings = {
        "time_budget": time_budget,
        "task": "classification",
        "estimator_list": estimator_list,
        "metric": metric,
        "log_file_name": "flaml_automl.log",
        "seed": seed
    }
    
    return automl, automl_settings


def train_model_flaml(path):
    """
    Trainiert das AutoML-Modell und gibt das beste Modell sowie die Genauigkeit zurück.
    Quelle: https://github.com/microsoft/FLAML
    """
    
    X,Y=load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Reshape X_train to 2D-Array
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1] * X_train.shape[2]  # Total features after flattening each sample
    X_train = X_train.reshape(num_samples, num_features)
    
    # Reshape X_test to 2D-Array
    num_samples_test = X_test.shape[0]
    num_features_test = X_test.shape[1] * X_test.shape[2]
    X_test = X_test.reshape(num_samples_test, num_features_test)
    
    #print("Shape of X_train:",X_train.shape)
    #print()
    automl,mysetting=create_model_flaml(X_train)
    automl.fit(X_train, y_train, mysetting)

    # Cross-Validation nach dem Fit durchführen
    accuracy = cross_val_score(automl.model, X_train, y_train, cv=5, scoring="accuracy").mean()
    print(f"Cross-Validation Genauigkeit: {accuracy:.4f}")
    print("Beste Konfiguration aus AutoML:", automl.best_config)

    # Endgültige Modellbewertung auf dem Testset
    test_accuracy = automl.score(X_test, y_test)
    print(f"Testgenauigkeit: {test_accuracy:.4f}")
    
    return automl.best_config,automl



# Funktion zur Spracherkennung in Echtzeit
def recognize_speech(model):
    recognized = None  # Variable zur Speicherung der erkannten Stimme
    #audio_buffer=queue.Queue() #Buffer fuer Audio Dateien   
     
    def callback(indata, frames, time, status):
        nonlocal recognized  # Nutzung der äußeren Variablen
        if status:
            print(status)
        #audio_buffer.put(indata.copy()) #Aufgenommene Daten im Buffer Speichern

        # Überprüfen, ob das Eingangsarray signifikant ist
        if np.max(np.abs(indata)) < 0.01:  # Schwellenwert für Stille
            recognized = "Unbekannt"
            return

        # MFCCs extrahieren und vorhersagen
        mfccs = extract_features(indata[:, 0], sr=16000)
        
        #print("MFCCs Shape:", mfccs.shape)
        #print()
        #mfccs = np.expand_dims(mfccs, axis=0)  # Für das SVM die Dimension anpassen
        mfccs_reshaped = mfccs.reshape(1, -1)
        
        #print(" mfccs_reshaped Shape:",  mfccs_reshaped.shape)
        #print()
            
        prediction = model.predict( mfccs_reshaped)
        predicted_label =prediction[0]

        # Bestimmen der erkannten Stimme
        if predicted_label == 0:
            recognized = "Felix"
        elif predicted_label == 1:
            recognized = "Linelle"
        elif predicted_label == 2:
            recognized = "LinelleNew"
        else:
            recognized = "Unbekannt"
            

    with sd.InputStream(samplerate=16000, channels=1, callback=callback):
        print(f"Aufnahme läuft. Drücke STRG+C zum Beenden.")
        try:
            
            while True:
            # Ausgabe des Erkennungsergebnisses in regelmäßigen Abständen
                if recognized is not None:
                    print(f"Erkannt: {recognized}")
                    recognized = None  # Reset der erkannten Stimme
                time.sleep(5)  # Wartezeit von 1 Sekunde, um die Ausgabe zu steuern
        except KeyboardInterrupt:
            print("Erkennung beendet.")
            #return audio_buffer




# Hauptprogramm
if __name__ == "__main__":
    audio_path = "Stimmen"
    model = train_svm_model(audio_path)
    audio_buffer=recognize_speech(model)  # Aufnahme und Spracherkennung in Echtzeit
    duration2=2
    #audio_to_text2(audio_buffer,duration2)

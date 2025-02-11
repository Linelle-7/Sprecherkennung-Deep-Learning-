import os
import tensorflow as tf
from shared_speech_utils import (
    train_model,
    train_optimized_model,
    load_training_data,
    #segment_and_analyze_with_output,
    plot
)

if __name__ == "__main__":
    # Pfad zu den Trainingsdaten
    audio_path = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf")
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}
    segment_length = 0.5

    # Lade die Trainingsdaten
    X, y = load_training_data(audio_path, label_map)
    num_classes = len(label_map)

    # Trainiere das Standardmodell
    model_standard = train_model(X, y, audio_path, label_map, segment_length=segment_length)

    # Trainiere das optimierte Modell mit Optuna
    model_optuna = train_optimized_model(X, y, num_classes, n_trials=20)

    # Testdatei analysieren mit beiden Modellen
    #print("Teste Modelle")
    #test_file = os.path.join(audio_path, "15-17.mp3")
    #segment_and_analyze_with_output(test_file, model_standard, label_map, window_size=5)
    #segment_and_analyze_with_output(test_file, model_optuna, label_map, window_size=5, optimiert=True)

    # Plotte die Ergebnisse
    #print("Plotte Ergebnisse")
    #plot()
import os
import tensorflow as tf
from shared_speech_utils import (
    train_model,
    train_optimized_model,
    load_training_data,
    segment_and_analyze_with_output,
    plot
)

if __name__ == "__main__":
    # Pfad zu den Trainingsdaten
    audio_path = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf")
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}

    # Lade die Trainingsdaten
    X, y = load_training_data(audio_path, label_map)
    num_classes = len(label_map)

    # Trainiere das Standardmodell
    model_standard = train_model(X, y, label_map)

    # Trainiere das optimierte Modell mit Optuna
    model_optuna = train_optimized_model(X, y, num_classes, n_trials=20)

    # Testdatei analysieren mit beiden Modellen
    print("Teste Modelle")
    test_file = os.path.join(audio_path, "15-45.mp3")
    print("Standardmodell, Window Größe 3 (1/4):")
    segment_and_analyze_with_output(test_file, model_standard, label_map, segment_length=1, window_size=3)
    print("Fertig")
    print("Optimiertes Modell, Window Größe 3 (2/4):")
    segment_and_analyze_with_output(test_file, model_optuna, label_map, segment_length=1, window_size=3, optimiert=True)
    print("Fertig")
    print("Standardmodell, Window Größe 5 (3/4):")
    segment_and_analyze_with_output(test_file, model_standard, label_map, segment_length=1, window_size=5)
    print("Fertig")
    print("Optimiertes Modell, Window Größe 5 (4/4):")
    segment_and_analyze_with_output(test_file, model_optuna, label_map, segment_length=1, window_size=5, optimiert=True)
    print("Fertig")

    # Plotte die Ergebnisse
    print("Plotte Ergebnisse")
    plot()
import os
from shared_speech_utils import (
    train_model_segmented,
    segment_and_analyze_with_output,
    live_audio_analysis
)

if __name__ == "__main__":
    # Pfad zu den Trainingsdaten
    audio_path = os.path.join(os.path.dirname(__file__), "..", "Stimmen")
    label_map = {"Felix": 0, "Linelle": 1}
    segment_length = 0.5

    # Modell trainieren
    model = train_model_segmented(audio_path, label_map, segment_length=segment_length)

    # Testdateien analysieren
    audio_files = [
        "Felix/Felix_1_1.wav",
        "Felix/Felix_15_2.wav",
        "Linelle/Linelle_7_1.wav",
        "Linelle/Linelle_10_2.wav",
        "Linelle/LinelleNew14.wav",
    ]

    for file in audio_files:
        test_file = os.path.join(audio_path, file)
        print(f"Analysiere Datei: {file}")
        segment_and_analyze_with_output(test_file, model, label_map, segment_length=0.5)

    # live_audio_analysis(model, label_map, segment_length=0.5, sr=16000, window_size=5)
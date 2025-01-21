import os
import tensorflow as tf
from shared_speech_utils import (
    train_model_segmented,
    segment_and_analyze_with_output,
)

if __name__ == "__main__":
    # Pfad zu den Trainingsdaten
    audio_path = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf")
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}
    segment_length = 0.5

    # Modell trainieren
    model = train_model_segmented(audio_path, label_map, segment_length=segment_length)

    # Testdatei analysieren
    test_file = os.path.join(audio_path, "15-17.mp3")
    segment_and_analyze_with_output(test_file, model, label_map, window_size=5)
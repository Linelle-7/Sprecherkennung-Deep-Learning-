import os
from SVM_shared_utils import(
    train_svm_model_optuna,
    segment_and_analyze_with_svm,
    plot_speaker_timeline,
    live_audio_analysis_svm,
    train_svm_model,
    plot_speaker_Gantt
)

if __name__ == "__main__":
    audio_path = os.path.join(os.path.dirname(__file__), "..", "Stimmen")
    label_map = {"Felix": 0, "Linelle": 1, "Paul": 2}
    segment_length=0.5
    model ,scaler= train_svm_model_optuna(audio_path,label_map,segment_length=segment_length)
    # model ,scaler= train_svm_model(audio_path,label_map,segment_length=segment_length)
    test_files=[
        os.path.join(audio_path, "Linelle\LinelleNew16.wav"),
        os.path.join(audio_path, "Felix\Felix_17_2.wav"),
    ]
    
    for file in test_files:
        # Sprechererkennung mit Glättung durchführen
        transcript = segment_and_analyze_with_svm(file, model, scaler,label_map, segment_length=0.25, sr=16000)
        plot_speaker_timeline(transcript, file)
        plot_speaker_Gantt(transcript, file)

        #process_mp3_file(file, model,scaler)
        print()
        
    live_audio_analysis_svm(model, scaler, label_map, segment_length=segment_length, sr=16000, window_size=3)
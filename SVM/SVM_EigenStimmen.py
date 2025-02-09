
import os
from SVM_shared_utils import(
    train_svm_model_optuna,
    segment_and_analyze_with_svm,
    plot_speaker_timeline,
    live_audio_analysis_svm2,
    train_svm_model
)

if __name__ == "__main__":
    
    audio_path = r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen"
    label_map = {"Felix": 0, "Linelle": 1, "Paul": 2}
    segment_length=0.5
    model ,scaler= train_svm_model_optuna(audio_path,label_map,segment_length=segment_length)
    # model ,scaler= train_svm_model(audio_path,label_map,segment_length=segment_length)
    test_files=[
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Linelle\LinelleNew16.wav",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\Stimmen\Felix\Felix_17_2.wav",
    ]
    
    for file in test_files:
        #predict_speaker(model, file, scaler)
        # test mit Segmentierte Audio Dateien
        #process_audio_file3(file, model,scaler)
        #audio_to_text(file)
        
        # Sprechererkennung mit Glättung durchführen
        transcript = segment_and_analyze_with_svm(file, model, scaler,label_map, segment_length=0.25, sr=16000)
        plot_speaker_timeline(transcript, file)

        #process_mp3_file(file, model,scaler)
        print()
        
    live_audio_analysis_svm2(model, scaler, label_map, segment_length=segment_length, sr=16000, window_size=3)
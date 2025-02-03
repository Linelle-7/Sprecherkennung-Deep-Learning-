
import os
from SVM_shared_utils import(
    train_svm_model_optuna,
    segment_and_analyze_with_svm,
    plot_speaker_timeline,
    live_audio_analysis_svm
)


if __name__ == "__main__":
    
    audio_path = r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\US-Wahlkampf"
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}
    segment_length=0.5
    model ,scaler= train_svm_model_optuna(audio_path,label_map,segment_length=segment_length)
    
    test_files=[
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\US-Wahlkampf\15-25.mp3",
        r"C:\Spracherkennung\Spracherkennung-Deep-Learning-\US-Wahlkampf\15-17.mp3",
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
        
    live_audio_analysis_svm(model, scaler, label_map, segment_length=0.1, sr=16000, window_size=3)
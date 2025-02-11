import os
from SVM_shared_utils import(
    train_svm_model_optuna,
    segment_and_analyze_with_svm,
    plot_speaker_timeline,
    live_audio_analysis_svm,
    train_svm_model,
    plot_speaker_Gantt,
    audio_to_text
)

if __name__ == "__main__":
    audio_path = os.path.join(os.path.dirname(__file__), "..", "US-Wahlkampf") 
    label_map = {"Biden": 0, "Moderator": 1, "Trump": 2}
    segment_length=0.5
    model ,scaler= train_svm_model_optuna(audio_path,label_map,segment_length=segment_length)
    # model ,scaler= train_svm_model(audio_path,label_map,segment_length=segment_length)
    
    test_files=[
        os.path.join(audio_path, "15-17.mp3"),
        os.path.join(audio_path, "15-25.mp3")
       ] 
    
    for file in test_files:
        #predict_speaker(model, file, scaler)
        # Sprechererkennung mit Glättung durchführen
        transcript = segment_and_analyze_with_svm(file, model, scaler,label_map, segment_length=0.25, sr=16000)
        #plot_speaker_timeline(transcript, file)
        plot_speaker_Gantt(transcript, file)
        audio_to_text(file,transcript,"en-US")

        print()
        
    live_audio_analysis_svm(model, scaler, label_map, segment_length=segment_length, sr=16000, window_size=3)
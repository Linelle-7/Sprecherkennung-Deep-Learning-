🎤 Sprechererkennung mit SVM

📌 Projektbeschreibung

Dieses Projekt ermöglicht die automatische Sprechererkennung in Audiodateien und Live. Dabei werden Mel-Frequency Cepstral Coefficients (MFCCs) extrahiert und mit einer Support Vector Machine (SVM) klassifiziert. Zusätzlich können die Hyperparameter mit Optuna & RandomizedSearchCV optimiert werden. Eine Gantt-Chart-Visualisierung zeigt die Sprecherwechsel grafisch an.

🛠️ Verwendete Technologien

Python (3.x)

Librosa (Audioanalyse)

Matplotlib & Seaborn (Visualisierungen)

Scikit-learn (Maschinelles Lernen)

Optuna (Hyperparameter-Optimierung)

Joblib (Parallele Verarbeitung)

Sounddevice (Live-Audio-Erkennung)

HMMlearn (Hidden Markov Models)

📂 Projektstruktur

📁 projektordner
│-- 📄 main.py                   # Hauptskript zur Ausführung
│-- 📄 speaker_recognition.py     # Funktionen zur Feature-Extraktion & Modelltraining
│-- 📄 live_audio_analysis.py     # Live-Sprechererkennung
│-- 📄 visualization.py           # Gantt-Chart & Confusion Matrix
│-- 📂 audio/                    # Enthält Audiodateien
│-- 📂 models/                   # Trainierte SVM-Modelle
│-- 📄 README.md                 # Diese Datei

🚀 Installation & Ausführung

1️⃣ Voraussetzungen installieren

pip install librosa matplotlib numpy pandas scikit-learn optuna joblib sounddevice hmmlearn seaborn

2️⃣ Skript ausführen

main teil

🎨 Hauptfunktionen

1. Feature-Extraktion (MFCCs)

Extrahiert Mel-Frequency Cepstral Coefficients aus Audiodateien.

Unterstützt Segmentierung zur feineren Analyse.

2. Modelltraining mit SVM & Hyperparameter-Tuning

RandomizedSearchCV für eine schnelle Optimierung(ist aber nur für kleine  Datenmenge geeignet daher werden ihre besten parameter als anfangsparameter für Optuna angewendet).

Optuna für eine tiefgehende Hyperparameter-Suche.

3. Sprecheridentifikation aus Audiodateien

Mithilfe des trainiertes Modells, einen gegebenen audio datei segmentiren und in jedes segment der Sprecher erkennen und die teilstücken speichern.

Ergebnis kann in einem textdatei gespeichert und als Gantt-Chart visualisiert werden.

4. Live-Sprechererkennung (Mikrofon-Input)

Echtzeit-Speaker-Klassifikation mit gleitendem Durchschnitt (Moving Average).

📊 Ausgabebeispiel einen Gantt-Diagramm

|------ Sprecher A ------|
            |---- Sprecher B ----|  
                                 |---- Sprecher C ---|

🏆 Metriken zur Bewertung

Confusion Matrix zur Analyse der Vorhersagen.

Accuracy, Precision, Recall, F1-Score zur Modellbewertung.

📬 Kontakt

Bei Fragen oder Vorschlägen gerne melden! 😊

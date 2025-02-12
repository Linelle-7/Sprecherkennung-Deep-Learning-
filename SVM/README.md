 # Sprechererkennung mit SVM

## Projektbeschreibung

Dieses Projekt ermöglicht die automatische Sprechererkennung in Audiodateien und Live. Dabei werden Mel-Frequency Cepstral Coefficients (MFCCs) extrahiert und mit einer Support Vector Machine (SVM) klassifiziert. Zusätzlich können die Hyperparameter mit Optuna & RandomizedSearchCV optimiert werden. Eine Gantt-Chart-Visualisierung zeigt die Sprecherwechsel grafisch an.

## Verwendete Technologien

- Python (3.x)

- Librosa (Audioanalyse)

- Matplotlib & Seaborn (Visualisierungen)

- Scikit-learn (Maschinelles Lernen)

- Randomize Search und Optuna (Hyperparameter-Optimierung)

- Joblib (Parallele Verarbeitung)

- Sounddevice (Live-Audio-Erkennung)

- HMMlearn (Hidden Markov Models)

## Projektstruktur

### Installation & Ausführung

#### 1. Voraussetzungen installieren
  
      ```bash
       pip install librosa matplotlib numpy pandas scikit-learn optuna joblib sounddevice hmmlearn seaborn

#### 2. Skript ausführen

In SVM_sharaed_utils.py stehen alle hauptmethoden zum besseren Übersicht.
In SVM_US.py lassen sich stimmen aus der US_Wahlkamph 2020 zum training und Bearbeitung nutzen und 
In SVM_EigenStimmen nutzen wir unseren eigenen Stimmen.

### Hauptfunktionen

- 1. Feature-Extraktion (MFCCs)

Extrahiert Mel-Frequency Cepstral Coefficients aus Audiodateien.

Unterstützt Segmentierung zur feineren Analyse.

- 2. Modelltraining mit SVM & Hyperparameter-Tuning

RandomizedSearchCV für eine schnelle Optimierung(ist aber nur für kleine  Datenmenge geeignet daher werden ihre besten parameter als anfangsparameter für Optuna angewendet).

Optuna für eine tiefgehende Hyperparameter-Suche.

- 3. Sprecheridentifikation aus Audiodateien

Mithilfe des trainiertes Modells, einen gegebenen audio datei segmentiren und in jedes segment der Sprecher erkennen und die teilstücken speichern.

Ergebnis kann in einem textdatei gespeichert und als Gantt-Chart visualisiert werden.

- 4. Live-Sprechererkennung (Mikrofon-Input)

Echtzeit-Speaker-Klassifikation mit gleitendem Durchschnitt (Moving Average).


### Metriken zur Bewertung

- Confusion Matrix zur Analyse der Vorhersagen.

- Accuracy, Precision, Recall, F1-Score zur Modellbewertung.

Bei Fragen oder Vorschlägen gerne melden! 😊

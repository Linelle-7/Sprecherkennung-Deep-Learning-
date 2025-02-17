 # Sprechererkennung mit SVM (Linelle Meneckdem)
Indiesem Projektordner werden Sprechererkennung in einem Datei sowohl als Live mittels einer Support Vector Machine (SVM) als klasssifikationsmodell durchgeführt. Dabei werden Mel-Frequency Cepstral Coefficients (MFCCs) extrahiert und Zusätzlich können die Hyperparameter mit Optuna & RandomizedSearchCV optimiert werden. Eine Gantt-Chart-Visualisierung zeigt die Sprecherwechsel grafisch an.

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

###### SVM_sharaed_utils.py 
Hier stehen alle hauptmethoden zum besseren Übersicht.
###### SVM_US.py 
Hier lassen sich stimmen aus der US_Wahlkamph 2020 zum training und Bearbeitung nutzen.
###### SVM_EigenStimmen
Hier nutzen wir unseren eigenen Stimmen.

### Hauptfunktionen

- 1. Feature-Extraktion (MFCCs)

Extrahiert Mel-Frequency Cepstral Coefficients aus Audiodateien.

Unterstützt Segmentierung zur feineren Analyse.

- 2. Modelltraining mit SVM & Hyperparameter-Tuning

RandomizedSearchCV für eine schnelle Optimierung(ist aber nur für kleine  Datenmenge geeignet daher werden ihre besten parameter als anfangsparameter für Optuna angewendet).

Optuna für eine tiefgehende Hyperparameter-Suche. Gut geeignet für grosse Datensätze

- 3. Sprecheridentifikation aus Audiodateien

Mithilfe des trainiertes Modells, einen gegebenen audio datei segmentiren und in jedes segment der Sprecher erkennen.

Ergebnis kann in einem textdatei gespeichert und als Gantt-Chart visualisiert werden.

- 4. Live-Sprechererkennung (Mikrofon-Input)

Echtzeit-Speaker-Klassifikation mit gleitendem Durchschnitt (Moving Average).


### Metriken zur Bewertung

- Confusion Matrix zur Analyse der Vorhersagen.

- Accuracy, Precision, Recall, F1-Score zur Modellbewertung.

## Hilfreiche Quellen
- https://github.com/ksrvap/Audio-classification-using-SVM-and-CNN/blob/main/Audio_Classification.py​
- https://scikit-learn.org/stable/model_selection.html
- https://youtu.be/ZqpSb5p1xQo?si=vcfkGEIYfVlvj45h

Bei Fragen oder Vorschlägen gerne melden! 😊

# Convolutional Neural Network (Felix Thienel)
<p>In diesem Projekt-Ordner wird die Protokollierungssoftware mittels eines CNNs versucht umzusetzen.</p>

### Ausführung des CNNs
<p>Die Grundfunktionalität des CNNs ist in "shared_speech_utils.py" diese Datei dient jedoch nur als importierbare Bibliothek.</p> 
<p>Zum Testen des CNNs hat man zwei verschiedene möglichkeiten:</p>

#### Sprechererkennung_US.py
<p>Das ist die Hauptdatei in der alle Funktionalitäten, die bisher entwickelt auch implementiert wurden.</p>
<p>Dieses CNN wird auf die Stimmen von Trump, Biden und einem Moderator aus einem TV Duell trainiert. Es werden sowohl ein "normales" CNN (fixe Parameter), als auch ein "dynamisches" CNN (Hyperparameteroptimiert mittels Optuna) erzeugt und miteinander verglichen. Dafür werden sowohl die Trainingsdaten ausgegeben, gespeichert und geplotet, als auch die Ergebnisse von eingelesenen Audiodateien ausgegeben.</p>

### Sprechererkennung.py
<p>In dieser Datei wird ein CNN auf unsere eigenen Stimmen trainiert.</p>
<p>Jedoch wurde aufgrund von weniger Trainingsdaten die Hauptentwicklung auf die Datei mit den Stimmen der US-Wahlkampf-Stimmen umgelegt.</p>

## Ausgabe

<p>Alle Ausgaben (Textdateien, JSON History Dateien des Lernens, Bilddateien der Genauigkeitsvergleiche) werden in dem Unterordner "/Ausgaben" gespeichert.</p>

<i>Quellen: <br>
<a href=https://towardsdatascience.com/voice-classification-using-deep-learning-with-python-6eddb9580381>Idee der Nutzung der Daten des US-Wahlkampfs</a><br>
<a href=https://www.kaggle.com/datasets/headsortails/us-election-2020-presidential-debates>Datensatz</a><br>
<a href=https://youtu.be/9GJ6XeB-vMg>Idee zur Umsetzung</a><br>
<a href="https://youtube.com/playlist?list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P&amp;si=klCWu3GVtesWvIlK">Playlist zur Idee der Klassifikation</a></i>
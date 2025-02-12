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

<p>Alle Ausgaben (Textdatein, JSON History Datein des Lernens, Bilddatein der Genauigkeitsvergleiche) werden in dem Unterordner "/Ausgaben" gespeichert.</p>
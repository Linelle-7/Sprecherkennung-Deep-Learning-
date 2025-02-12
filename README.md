# Protokollierungssoftware
<p>Eine Software die imstande sein sollte Audiodateien einzulesen und zu diesen eine Textdatei mit den jeweiligen Sprechern (und dem Gesagtem)* auszugeben.</p>

## Aufbau
<p>Das jeweilige Modell (2 verschiedene Ansätze siehe weiter unten) wird mit Trainingsdaten versorgt, welche intern als MFCCs verarbeitet werden. Auf Grundlage dieser Trainingsdaten wird mithilfe von Optuna eine Hyperparameteroptimierung durchgeführt, um die höchst mögliche Genauigkeit zu erreichen.</p>

<p>Nach dem Training kann man nun Audiodateien einlesen, die in variable Segmentlängen unterteilt werden, um diese dann einzeln zu analysieren und eine Vorhersage des Sprechers abzugeben.</p>

<p>Nachdem alle Segmente analysiert wurden, wird über die gesamte Audiodatei ein dynamisches, variabel Langes Fenster verschoben, das die jeweiligen Segmente mit den jeweiligen Vor- und Nachgängern glättet.</p>

<p>Die geglätteten Segmente werden nun in einer Textdatei, mit den jeweiligen Zeitstempeln versehen, ausgegeben.</p>

<p>Die Verarbeitung von Live-Mikrophone-Aufnahmen wird derzeit schon teilweise unterstützt, es wurden aber noch nicht alle Funktionen der normalen Audioanalyse implementiert.</p>

<p>Die Funktionalität das Gesagte in Text umzuformen und auch mit in der ausgegebenen Datei abzuspeichern wird derzeit noch nicht unterstützt.</p>

### Verschiedene Ansätze:
#### Support Vector Machine (Linelle):
<p>Für weitere Informationen die jeweilige readme Datei lesen: "/SVM/README.md".</p>

#### Convolutional Neural Network (Felix):
<p>Für weitere Informationen die jeweilige readme Datei lesen: "/CNN/README.md".</p>


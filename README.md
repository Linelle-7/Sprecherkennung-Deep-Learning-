# Protokollierungssoftware
Eine Software die im Stande sein sollte Audiodatein einzulesen und zu diesen eine Textdatei mit den jeweiligen Sprechern (und dem Gesagtem)* auszugeben.
## Aufbau
<p>Das jeweilige Modell (2 verschiedene Ansätze siehe weiter unten) wird mit Trainingsdaten versorgt, welche intern als MFCCs verarbeitet werden. Auf grundlage dieser Trainingsdaten wird mit Hilfe von Optuna eine Hyperparameteroptimierung durchgeführt um die höchst mögliche Genauigkeit zu erreichen.</p>
<p>Nach dem Training kann man nun Audiodatein einlesen, die in variable Segmentlängen unterteilt werden, um diese dann einzeln zu analysieren und eine Vorhersage des Sprechers abzugeben.</p>
<p>Nachdem alle Segmente analysiert wurden, wird über die gesamte Audiodatei ein dynamisches, variabel Langes Fenster verschoben, das die jeweiligen Segmente mit den jeweiligen Vor- und Nachgängern glättet.</p>
<p>Die gegläteten Segmente werden nun in einer Textdatei, mit den jeweiligen Zeitstempeln versehen, ausgegeben.</p>
<p>Die verarbeitung von Live-Mikrophone-Aufnahmen wird derzeit schon teilweise unterstützt, es wurden aber noch nicht alle Funktionen der normalen Audioanalyse implementiert.</p>
<p>Die funktionalität das Gesagte in Text umzuformen und auch mit in der ausgegebenen Datei abzuspeichern wird derzeit noch nicht unterstützt.</p>

### Verschiedene Ansätze:
#### Support Vector Machine (Linelle):
Für weitere Informationen die jeweilige readme Datei lesen: "/SVM/README.md".
#### Convolutional Neural Network (Felix): 
Für weitere Informationen die jeweilige readme Datei lesen: "/CNN/README.md".


<i>*noch nicht komplett implementiert</i>
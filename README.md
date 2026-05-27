# live-gong-detection

Live-Webcam-App: erkennt eine Person, markiert sie mit einer Bounding-Box und
spielt einen Gong-Ton, wenn die Person eine Schlag-Bewegung macht. Die Wucht
(das Momentum) des Schlags steuert die Lautstärke – je schneller die Hand, desto
lauter der Gong.

## Funktionsweise

- **Personen-/Pose-Erkennung:** Ultralytics YOLO-Pose (`yolo11n-pose.pt`) liefert
  pro Frame die Personen-Box und die Handgelenk-Keypoints.
- **Schlagerkennung:** Das Handgelenk wird über die Frames verfolgt. Die
  Geschwindigkeit wird auf die Schulterbreite normiert (distanzunabhängig). Eine
  Zustandsmaschine mit Hysterese und Cooldown löst pro Geste genau einen Gong aus.
- **Momentum → Lautstärke:** Die Spitzengeschwindigkeit des Schlags wird perzeptuell
  auf eine Lautstärke abgebildet.
- **Gong-Klang:** synthetisch mit NumPy erzeugt (inharmonische, abklingende
  Sinus-Obertöne → metallischer Klang), non-blocking über `sounddevice` abgespielt.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Beim ersten Start lädt Ultralytics automatisch das Modell `yolo11n-pose.pt`
(Internet erforderlich).

## Starten

```powershell
.\.venv\Scripts\python.exe main.py
```

Eine Schlag-Bewegung vor der Kamera ausführen. Beenden mit `q` oder `ESC`.

## Kalibrieren

Alle Schwellwerte stehen in `config.py`. Das Overlay zeigt die aktuelle
(normierte) Geschwindigkeit und einen Balken – damit lassen sich
`V_ENTER_THRESH`, `V_EXIT_THRESH`, `V_MIN_INPUT` und `V_MAX_INPUT` an die eigene
Kamera/Umgebung anpassen.

## Projektstruktur

| Datei | Aufgabe |
|-------|---------|
| `main.py` | Capture-Loop, verbindet alle Komponenten |
| `config.py` | zentrale Tuning-Parameter |
| `pose_detector.py` | YOLO-Pose-Wrapper |
| `strike_detector.py` | Handgelenk-Tracking + Schlagerkennung + Momentum→Lautstärke |
| `gong_audio.py` | Gong-Synthese + non-blocking Audio-Thread |
| `visualizer.py` | Bounding-Box, Marker und Status-Overlay |

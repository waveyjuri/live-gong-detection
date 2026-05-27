"""Zentrale Konfiguration / Tuning-Parameter fuer die Gong-Erkennung.

Alle experimentell zu kalibrierenden Werte stehen hier an einer Stelle.
Beim Justieren hilft das Live-Overlay der aktuellen Geschwindigkeit (siehe
visualizer.py / main.py).
"""

# --- Kamera ---
CAMERA_INDEX = 0          # 0 = Standard-Webcam
FRAME_WIDTH = 1280        # gewuenschte Capture-Breite (Kamera entscheidet final)
FRAME_HEIGHT = 720
MIRROR_VIEW = True        # Spiegelbild (natuerlicher fuer den Nutzer)

# --- YOLO-Pose ---
MODEL_NAME = "yolo11n-pose.pt"   # Nano-Modell, wird beim ersten Lauf geladen
INFER_IMGSZ = 480          # Eingangsaufloesung der Inferenz (kleiner = schneller)
PERSON_CONF = 0.40         # Mindest-Confidence fuer eine Person
KP_CONF_THRESH = 0.50      # Mindest-Confidence eines Keypoints (z.B. Handgelenk)

# COCO-Keypoint-Indizes (Ultralytics Pose)
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# --- Schlagerkennung ---
# Geschwindigkeit ist normiert auf Schulterbreite -> Einheit ~ "Koerperbreiten/Sekunde".
V_ENTER_THRESH = 2.2       # ab hier beginnt ein Schlag (in_strike = True)
V_EXIT_THRESH = 1.0        # darunter gilt der Schlag als beendet -> Trigger
COOLDOWN_SEC = 0.6         # Mindestabstand zwischen zwei Gongs
SMOOTH_WINDOW = 3          # Anzahl Frames zum Glaetten der Geschwindigkeit
MIN_SCALE_PX = 30.0        # Fallback/Untergrenze fuer die Normierungsskala in px

# --- Momentum -> Lautstaerke ---
V_MIN_INPUT = 2.2          # Geschwindigkeit, die auf die Mindestlautstaerke abbildet
V_MAX_INPUT = 9.0          # Geschwindigkeit, ab der maximale Lautstaerke erreicht ist
VOL_FLOOR = 0.20           # Mindestlautstaerke eines ausgeloesten Gongs (0..1)
LOUDNESS_GAMMA = 0.6       # perzeptuelle Kurve (<1 hebt leise Schlaege an)

# --- Gong-Synthese ---
SAMPLE_RATE = 44100
GONG_DURATION = 2.5        # Sekunden
GONG_BASE_FREQ = 110.0     # Grundfrequenz in Hz
# Inharmonische Partials -> metallischer Gong-Klang (keine ganzzahligen Vielfachen)
GONG_PARTIAL_RATIOS = [1.0, 1.34, 1.78, 2.41, 3.05, 4.2]
GONG_PARTIAL_AMPS = [1.0, 0.7, 0.55, 0.4, 0.25, 0.15]
GONG_PARTIAL_DECAYS = [1.2, 1.0, 0.8, 0.6, 0.45, 0.3]   # Abklingzeit pro Partial (s)
GONG_ATTACK_SEC = 0.004    # kurzer Fade-In gegen Knacken
MASTER_GAIN = 0.9          # globaler Kopfraum, vermeidet Clipping

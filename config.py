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
VOL_FLOOR = 0.45           # Mindestlautstaerke eines ausgeloesten Gongs (0..1)
LOUDNESS_GAMMA = 0.5       # perzeptuelle Kurve (<1 hebt leise Schlaege an)

# --- Gong-Synthese (Tagesschau-Stil: tiefer, wuerdevoller Orchester-Gong/Tam-Tam) ---
SAMPLE_RATE = 44100
GONG_DURATION = 3.8        # Sekunden (langer, feierlicher Ausklang)
GONG_BASE_FREQ = 60.0      # tiefe Grundfrequenz in Hz (voller, dunkler Boden)
# Mischung aus tiefen, nahezu harmonischen Partials (Fundament) und hoeheren
# inharmonischen Partials (metallischer Schimmer-Wash).
GONG_PARTIAL_RATIOS = [1.0, 2.0, 2.74, 3.69, 4.81, 6.32, 8.15, 10.4]
GONG_PARTIAL_AMPS = [1.0, 0.65, 0.5, 0.45, 0.38, 0.3, 0.22, 0.15]
GONG_PARTIAL_DECAYS = [3.4, 2.8, 2.3, 1.9, 1.5, 1.1, 0.8, 0.55]   # Abklingzeit (s)
GONG_ATTACK_SEC = 0.003    # kurzer Fade-In gegen Knacken
MASTER_GAIN = 1.0          # voller Pegel (Normierung auf Peak verhindert Clipping)

# Bloom/Swell: der Gong "blueht" nach dem Anschlag kurz auf, statt sofort am
# lautesten zu sein (typisches Tam-Tam-Verhalten). 0 = aus (sofortiger Anschlag).
GONG_BLOOM_TAU = 0.10      # Anstiegszeit des Swells in s

# Pitch-Glide: beim Tagesschau-Gong nur ganz dezent abfallend (kein China-Glide).
GONG_GLIDE_START = 1.0     # Faktor direkt beim Schlag
GONG_GLIDE_END = 0.985     # leicht abfallend -> wirkt "satt"/setzt sich
GONG_GLIDE_TAU = 0.5       # Zeitkonstante des Glides in s

# Metallisches Schimmern: dezenter, etwas laenger ausklingender Rausch-Anteil.
GONG_SHIMMER_AMP = 0.18    # Anteil des Schimmerns (0 = aus)
GONG_SHIMMER_DECAY = 0.35  # Abklingzeit des Schimmerns in s

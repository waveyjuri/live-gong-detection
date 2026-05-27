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

# --- Kickroll (beidhaendiges Luft-Trommeln) ---
# Werden BEIDE Haende erkannt und macht die Person schnelle Abwaerts-Schlaege
# (Luft-Trommeln), wird pro Schlag ein harter Kick gespielt; der Gong pausiert.
KICKROLL_ENABLE = True
KICK_SMOOTH_WINDOW = 2     # kurze Glaettung -> bleibt fuer schnelles Trommeln reaktiv
V_KICK_ENTER = 1.1         # vertikale Abwaerts-Geschwindigkeit, ab der ein Kick zaehlt
V_KICK_EXIT = 0.4          # darunter ist die Hand "oben" -> bereit fuer den naechsten
KICK_COOLDOWN = 0.08       # Mindestabstand zweier Kicks DERSELBEN Hand (Uptempo)
KICKROLL_WINDOW = 1.0      # beide Haende muessen binnen dieser Zeit getroffen haben
KICK_V_MIN = 2.5           # Geschwindigkeit -> Mindestlautstaerke
KICK_V_MAX = 12.0          # Geschwindigkeit -> maximale Lautstaerke
KICK_VOL_FLOOR = 0.55      # Mindestlautstaerke eines Kicks (0..1)
KICK_LOUDNESS_GAMMA = 0.6  # perzeptuelle Kurve

# Bass-Synthese (Hardtechno-Bass: brutal uebersteuert, brettartig/verzerrt)
KICK_DURATION = 0.42       # Sekunden (kurz/treibend fuer Uptempo)
KICK_START_FREQ = 95.0     # Anfangsfrequenz in Hz (etwas Punch im Anschlag)
KICK_END_FREQ = 52.0       # Endfrequenz in Hz (tiefer, druckvoller Bass)
KICK_PITCH_TAU = 0.040     # Zeitkonstante des Pitch-Sweeps in s
KICK_DECAY = 0.28          # Abklingzeit des Bass-Koerpers in s
KICK_HARM2_AMP = 0.15      # 2. Harmonische (Distortion liefert die meisten Oberwellen)
KICK_CLICK_AMP = 0.0       # Klick-Transient AUS
KICK_CLICK_SEC = 0.004     # Laenge des Klicks in s (ungenutzt bei AMP=0)
KICK_DRIVE = 9.0           # tanh-Saturation -> Sinus wird fast Square (Hardtechno-Biss)
KICK_HARDCLIP = 0.55       # zusaetzliches Hard-Clipping (< 1.0 = brettartiger/aggressiver)
KICK_ATTACK_SEC = 0.003    # kurzer Fade-In gegen Knacken
KICK_GAIN = 1.0            # Gesamtpegel des Bass

# Tonfolge: pro Schlag wird der naechste Halbton-Versatz (relativ zur Basis)
# durchlaufen -> aus dem Trommeln wird eine treibende Bassline. Werte in
# Halbtonschritten; bei jedem neuen Kickroll startet die Folge von vorn.
KICK_SEQUENCE = [0, 0, 3, 0, 5, 3, 7, 5]  # Moll-Pentatonik-Groove
KICK_SEQUENCE_RESET = True  # True = jeder neue Kickroll beginnt am Anfang der Folge

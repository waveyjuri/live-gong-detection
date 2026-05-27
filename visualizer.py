"""Zeichnet Bounding-Box, Handgelenk-Marker und Status-Overlay auf den Frame."""

import time

import cv2

import config
from pose_detector import Person
from strike_detector import StrikeResult

_GREEN = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def draw(frame, person: Person | None, result: StrikeResult,
         last_trigger_volume: float, last_trigger_time: float):
    h = frame.shape[0]

    if person is not None:
        x1, y1, x2, y2 = (int(v) for v in person.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
        cv2.putText(frame, f"Person {person.conf:.2f}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2)

    # verfolgtes Handgelenk
    if result.wrist_pos is not None:
        wx, wy = (int(result.wrist_pos[0]), int(result.wrist_pos[1]))
        color = _RED if result.current_velocity > config.V_ENTER_THRESH else _YELLOW
        cv2.circle(frame, (wx, wy), 10, color, -1)

    # Status-Texte
    if not result.has_person:
        status = "Keine Person erkannt"
    elif result.wrist_pos is None:
        status = "Hand nicht sichtbar"
    else:
        status = f"Geschwindigkeit: {result.current_velocity:4.1f}"
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, _WHITE, 2)

    # Geschwindigkeits-Balken (relativ zur max. Eingangsgeschwindigkeit)
    bar_w = 300
    filled = int(min(result.current_velocity / config.V_MAX_INPUT, 1.0) * bar_w)
    cv2.rectangle(frame, (12, 40), (12 + bar_w, 56), (80, 80, 80), 1)
    cv2.rectangle(frame, (12, 40), (12 + filled, 56), _RED, -1)

    # kurzes "GONG!"-Feedback nach einem Trigger
    if time.time() - last_trigger_time < 0.6:
        cv2.putText(frame, f"GONG!  Vol {last_trigger_volume:.2f}",
                    (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 3)

    cv2.putText(frame, "q/ESC = beenden", (12, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)
    return frame

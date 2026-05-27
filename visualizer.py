"""Zeichnet Bounding-Box, Handgelenk-Marker und Status-Overlay auf den Frame."""

import time

import cv2

import config
from kickroll_detector import KickResult
from pose_detector import Person
from strike_detector import StrikeResult

_GREEN = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_MAGENTA = (220, 0, 220)


def draw(frame, person: Person | None, result: StrikeResult,
         kick_result: KickResult | None,
         last_trigger_volume: float, last_trigger_time: float,
         last_kick_time: float):
    h = frame.shape[0]
    kickroll_active = kick_result is not None and kick_result.active

    if person is not None:
        x1, y1, x2, y2 = (int(v) for v in person.bbox)
        box_color = _MAGENTA if kickroll_active else _GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"Person {person.conf:.2f}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    if kick_result is not None and kick_result.has_two_hands:
        # beide Handgelenke markieren (Kickroll-Tracking)
        for wpos in kick_result.wrist_positions:
            wx, wy = int(wpos[0]), int(wpos[1])
            cv2.circle(frame, (wx, wy), 10, _MAGENTA, -1)
    elif result.wrist_pos is not None:
        # einzelnes verfolgtes Handgelenk (Gong-Modus)
        wx, wy = (int(result.wrist_pos[0]), int(result.wrist_pos[1]))
        color = _RED if result.current_velocity > config.V_ENTER_THRESH else _YELLOW
        cv2.circle(frame, (wx, wy), 10, color, -1)

    # Status-Texte
    two_hands = kick_result is not None and kick_result.has_two_hands
    if not result.has_person:
        status = "Keine Person erkannt"
    elif kickroll_active:
        status = f"KICKROLL aktiv  v={kick_result.max_vy:4.1f}"
    elif two_hands:
        # beide Haende da, aber noch kein Schlag erkannt -> vert. Tempo anzeigen
        status = (f"2 Haende  v_vert={kick_result.max_vy:4.1f} "
                  f"(Kick ab {config.V_KICK_ENTER:.1f})")
    elif result.wrist_pos is None:
        status = "Hand nicht sichtbar"
    else:
        status = f"Geschwindigkeit: {result.current_velocity:4.1f}"
    status_color = _MAGENTA if kickroll_active else _WHITE
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2)

    # Geschwindigkeits-Balken (relativ zur max. Eingangsgeschwindigkeit)
    bar_w = 300
    filled = int(min(result.current_velocity / config.V_MAX_INPUT, 1.0) * bar_w)
    cv2.rectangle(frame, (12, 40), (12 + bar_w, 56), (80, 80, 80), 1)
    cv2.rectangle(frame, (12, 40), (12 + filled, 56), _RED, -1)

    # kurzes Trigger-Feedback (Kick hat Vorrang, da Uptempo)
    if time.time() - last_kick_time < 0.15:
        cv2.putText(frame, "KICK!", (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _MAGENTA, 3)
    elif time.time() - last_trigger_time < 0.6:
        cv2.putText(frame, f"GONG!  Vol {last_trigger_volume:.2f}",
                    (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 3)

    cv2.putText(frame, "q/ESC = beenden", (12, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)
    return frame

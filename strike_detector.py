"""Erkennung eines Gong-Schlags anhand der Handgelenk-Geschwindigkeit.

Fokus auf eine Person (groesste Bounding-Box). Die Handgelenk-Position wird
ueber Frames verfolgt; die Geschwindigkeit wird auf die Schulterbreite normiert,
damit der Schwellwert distanzunabhaengig ist. Eine Zustandsmaschine mit
Hysterese + Cooldown loest pro Geste genau einen Gong aus und liefert dessen
Spitzengeschwindigkeit fuer das Lautstaerke-Mapping.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import config
from pose_detector import Person


@dataclass
class StrikeResult:
    """Ergebnis der Frame-Verarbeitung."""
    triggered: bool = False        # wurde in diesem Frame ein Gong ausgeloest?
    volume: float = 0.0            # Lautstaerke (0..1) bei triggered
    peak_velocity: float = 0.0     # Spitzengeschwindigkeit des Schlags
    current_velocity: float = 0.0  # aktuelle (geglaettete) Geschwindigkeit (Overlay)
    wrist_pos: tuple | None = None  # verfolgte Handgelenk-Position (Overlay)
    has_person: bool = False


def _euclidean(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _map_velocity_to_volume(peak_velocity: float) -> float:
    """Bildet die Spitzengeschwindigkeit perzeptuell auf [VOL_FLOOR, 1.0] ab."""
    lo, hi = config.V_MIN_INPUT, config.V_MAX_INPUT
    t = (peak_velocity - lo) / max(hi - lo, 1e-6)
    t = min(max(t, 0.0), 1.0)
    t = t ** config.LOUDNESS_GAMMA
    return config.VOL_FLOOR + t * (1.0 - config.VOL_FLOOR)


class StrikeDetector:
    def __init__(self) -> None:
        self.prev_wrist_pos: tuple | None = None
        self.prev_time: float | None = None
        self.vel_history: deque[float] = deque(maxlen=config.SMOOTH_WINDOW)
        self.last_trigger_time: float = -1e9
        self.in_strike: bool = False
        self.peak_velocity: float = 0.0

    def _reset_tracking(self) -> None:
        """Tracking zuruecksetzen, ohne den Schlag-/Cooldown-Status zu verlieren."""
        self.prev_wrist_pos = None
        self.prev_time = None
        self.vel_history.clear()

    @staticmethod
    def _pick_primary(persons: list[Person]) -> Person:
        return max(persons, key=lambda p: p.bbox_area)

    def _select_wrist(self, person: Person):
        """Waehlt das Handgelenk mit ausreichender Confidence.

        Bei zwei gueltigen Haenden wird die genommen, die weiter vom letzten
        verfolgten Punkt entfernt liegt (= aktivere Hand), sonst das rechte.
        """
        candidates = []
        for idx in (config.KP_RIGHT_WRIST, config.KP_LEFT_WRIST):
            if person.kp_conf[idx] >= config.KP_CONF_THRESH:
                candidates.append(tuple(person.keypoints[idx]))
        if not candidates:
            return None
        if len(candidates) == 1 or self.prev_wrist_pos is None:
            return candidates[0]
        return max(candidates, key=lambda p: _euclidean(p, self.prev_wrist_pos))

    def _scale(self, person: Person) -> float:
        """Normierungsskala: Schulterbreite, sonst BBox-Diagonale, sonst Fallback."""
        ls, rs = config.KP_LEFT_SHOULDER, config.KP_RIGHT_SHOULDER
        if person.kp_conf[ls] >= config.KP_CONF_THRESH and \
                person.kp_conf[rs] >= config.KP_CONF_THRESH:
            shoulder = _euclidean(person.keypoints[ls], person.keypoints[rs])
            if shoulder >= config.MIN_SCALE_PX:
                return shoulder
        x1, y1, x2, y2 = person.bbox
        diag = math.hypot(x2 - x1, y2 - y1)
        return max(diag * 0.25, config.MIN_SCALE_PX)

    def process(self, persons: list[Person], now: float) -> StrikeResult:
        if not persons:
            self._reset_tracking()
            return StrikeResult(has_person=False)

        person = self._pick_primary(persons)
        wrist = self._select_wrist(person)
        result = StrikeResult(has_person=True, wrist_pos=wrist)

        if wrist is None:
            # Hand nicht sichtbar -> kein Fehl-Trigger
            self._reset_tracking()
            return result

        if self.prev_wrist_pos is None or self.prev_time is None:
            # erster gueltiger Frame: nur initialisieren, keine Geschwindigkeit
            self.prev_wrist_pos = wrist
            self.prev_time = now
            return result

        dt = max(now - self.prev_time, 1e-3)
        scale = self._scale(person)
        raw_disp = _euclidean(wrist, self.prev_wrist_pos)
        norm_velocity = (raw_disp / scale) / dt

        self.vel_history.append(norm_velocity)
        smoothed_v = sum(self.vel_history) / len(self.vel_history)
        result.current_velocity = smoothed_v

        self.prev_wrist_pos = wrist
        self.prev_time = now

        # --- Zustandsmaschine mit Hysterese ---
        if not self.in_strike:
            if smoothed_v > config.V_ENTER_THRESH:
                self.in_strike = True
                self.peak_velocity = smoothed_v
        else:
            self.peak_velocity = max(self.peak_velocity, smoothed_v)
            if smoothed_v < config.V_EXIT_THRESH:
                self.in_strike = False
                if now - self.last_trigger_time > config.COOLDOWN_SEC:
                    self.last_trigger_time = now
                    result.triggered = True
                    result.peak_velocity = self.peak_velocity
                    result.volume = _map_velocity_to_volume(self.peak_velocity)
                self.peak_velocity = 0.0

        return result

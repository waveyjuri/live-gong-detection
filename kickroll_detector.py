"""Erkennung eines beidhaendigen "Kickrolls" (Luft-Trommeln).

Werden BEIDE Handgelenke erkannt und macht die Person schnelle Abwaerts-Schlaege,
wird pro Schlag (pro Hand, mit eigenem Cooldown) ein harter Kick ausgeloest. Der
Kickroll-Modus gilt als aktiv, sobald BEIDE Haende innerhalb eines kurzen Fensters
getroffen haben -- so wird er von einer einzelnen Gong-Geste unterschieden.

Jede Hand wird unabhaengig verfolgt; betrachtet wird nur die vertikale
Geschwindigkeit (abwaerts = positiv), normiert auf die Schulterbreite, damit der
Schwellwert distanzunabhaengig ist.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import config
from pose_detector import Person


@dataclass
class KickResult:
    """Ergebnis der Frame-Verarbeitung des Kickroll-Detektors."""
    active: bool = False                       # Kickroll-Modus aktiv?
    kicks: list[float] = field(default_factory=list)   # Lautstaerken neuer Kicks
    wrist_positions: list = field(default_factory=list)  # [links, rechts] fuer Overlay
    has_two_hands: bool = False
    max_vy: float = 0.0                        # groesste vert. Geschwindigkeit (Overlay)


def _euclidean(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _map_velocity_to_volume(v: float) -> float:
    """Bildet die Abwaerts-Geschwindigkeit perzeptuell auf [KICK_VOL_FLOOR, 1.0] ab."""
    lo, hi = config.KICK_V_MIN, config.KICK_V_MAX
    t = (v - lo) / max(hi - lo, 1e-6)
    t = min(max(t, 0.0), 1.0)
    t = t ** config.KICK_LOUDNESS_GAMMA
    return config.KICK_VOL_FLOOR + t * (1.0 - config.KICK_VOL_FLOOR)


class _HandTracker:
    """Verfolgt eine Hand und erkennt einzelne Abwaerts-Schlaege (Hysterese)."""

    def __init__(self) -> None:
        self.prev_y: float | None = None
        self.prev_time: float | None = None
        self.vel_hist: deque[float] = deque(maxlen=config.KICK_SMOOTH_WINDOW)
        self.in_down: bool = False
        self.last_hit_time: float = -1e9
        self.last_v: float = 0.0   # zuletzt geglaettete vert. Geschwindigkeit (Overlay)

    def reset(self) -> None:
        self.prev_y = None
        self.prev_time = None
        self.vel_hist.clear()
        self.in_down = False
        self.last_v = 0.0

    def update(self, y: float, scale: float, now: float) -> tuple[bool, float]:
        """Aktualisiert mit neuer y-Position; gibt (hit, geschwindigkeit) zurueck."""
        if self.prev_y is None or self.prev_time is None:
            self.prev_y = y
            self.prev_time = now
            return False, 0.0

        dt = max(now - self.prev_time, 1e-3)
        vy = ((y - self.prev_y) / scale) / dt   # positiv = abwaerts (y waechst nach unten)
        self.prev_y = y
        self.prev_time = now

        self.vel_hist.append(vy)
        sv = sum(self.vel_hist) / len(self.vel_hist)
        self.last_v = sv

        hit, hit_v = False, 0.0
        if not self.in_down:
            if sv > config.V_KICK_ENTER:
                self.in_down = True
                if now - self.last_hit_time > config.KICK_COOLDOWN:
                    self.last_hit_time = now
                    hit, hit_v = True, sv
        else:
            if sv < config.V_KICK_EXIT:
                self.in_down = False
        return hit, hit_v


class KickrollDetector:
    def __init__(self) -> None:
        self.left = _HandTracker()
        self.right = _HandTracker()

    def _reset(self) -> None:
        self.left.reset()
        self.right.reset()

    @staticmethod
    def _pick_primary(persons: list[Person]) -> Person:
        return max(persons, key=lambda p: p.bbox_area)

    @staticmethod
    def _wrist(person: Person, idx: int):
        if person.kp_conf[idx] >= config.KP_CONF_THRESH:
            return tuple(person.keypoints[idx])
        return None

    @staticmethod
    def _scale(person: Person) -> float:
        ls, rs = config.KP_LEFT_SHOULDER, config.KP_RIGHT_SHOULDER
        if person.kp_conf[ls] >= config.KP_CONF_THRESH and \
                person.kp_conf[rs] >= config.KP_CONF_THRESH:
            shoulder = _euclidean(person.keypoints[ls], person.keypoints[rs])
            if shoulder >= config.MIN_SCALE_PX:
                return shoulder
        x1, y1, x2, y2 = person.bbox
        diag = math.hypot(x2 - x1, y2 - y1)
        return max(diag * 0.25, config.MIN_SCALE_PX)

    def process(self, persons: list[Person], now: float) -> KickResult:
        if not persons:
            self._reset()
            return KickResult()

        person = self._pick_primary(persons)
        lw = self._wrist(person, config.KP_LEFT_WRIST)
        rw = self._wrist(person, config.KP_RIGHT_WRIST)

        # Beide Haende noetig -> sonst kein Kickroll
        if lw is None or rw is None:
            self._reset()
            return KickResult(has_two_hands=False)

        scale = self._scale(person)
        lhit, lv = self.left.update(lw[1], scale, now)
        rhit, rv = self.right.update(rw[1], scale, now)

        # Beide Haende sind sichtbar (oben bereits geprueft); aktiv, sobald
        # IRGENDEINE Hand kuerzlich getroffen hat -> Kicks ab dem ersten Schlag.
        last_hit = max(self.left.last_hit_time, self.right.last_hit_time)
        active = (now - last_hit) < config.KICKROLL_WINDOW

        res = KickResult(active=active, has_two_hands=True,
                         wrist_positions=[lw, rw],
                         max_vy=max(self.left.last_v, self.right.last_v))
        if active:
            if lhit:
                res.kicks.append(_map_velocity_to_volume(lv))
            if rhit:
                res.kicks.append(_map_velocity_to_volume(rv))
        return res

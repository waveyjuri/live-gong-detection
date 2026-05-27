"""Synthetischer Gong-Klang + non-blocking Wiedergabe.

Der Gong wird aus inharmonischen, exponentiell abklingenden Sinus-Obertoenen
erzeugt (metallischer Klang). Ein Worker-Thread holt Lautstaerke-Werte aus einer
Queue und spielt den (skalierten) Gong ueber sounddevice ab, damit der
Video-Loop nie blockiert.
"""

import queue
import threading

import numpy as np

import config

_SENTINEL = object()


def synth_gong_base() -> np.ndarray:
    """Erzeugt den Basis-Gong (Volume 1.0), einmalig zu cachen."""
    sr = config.SAMPLE_RATE
    t = np.linspace(0.0, config.GONG_DURATION,
                    int(sr * config.GONG_DURATION), endpoint=False)
    signal = np.zeros_like(t)
    for ratio, amp, decay in zip(config.GONG_PARTIAL_RATIOS,
                                 config.GONG_PARTIAL_AMPS,
                                 config.GONG_PARTIAL_DECAYS):
        env = np.exp(-t / decay)
        signal += amp * env * np.sin(2.0 * np.pi * config.GONG_BASE_FREQ * ratio * t)

    # kurzer Attack-Fade gegen Knacken am Anfang
    attack = int(sr * config.GONG_ATTACK_SEC)
    if attack > 0:
        signal[:attack] *= np.linspace(0.0, 1.0, attack)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return (signal * config.MASTER_GAIN).astype(np.float32)


class GongPlayer:
    """Verwaltet einen Audio-Worker-Thread fuer non-blocking Wiedergabe."""

    def __init__(self) -> None:
        self._base = synth_gong_base()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._sd = None
        self._available = False
        try:
            import sounddevice as sd
            self._sd = sd
            self._available = True
        except Exception as exc:  # noqa: BLE001 - Audio darf optional fehlen
            print(f"[gong_audio] Audio nicht verfuegbar ({exc}); laeuft ohne Ton.")
        self._thread.start()

    @property
    def available(self) -> bool:
        return self._available

    def play(self, volume: float) -> None:
        """Lautstaerke (0..1) zur Wiedergabe einreihen (non-blocking)."""
        self._queue.put(float(volume))

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if not self._available:
                continue
            try:
                samples = (self._base * float(item)).astype(np.float32)
                # neuer Schlag unterbricht den vorherigen Gong
                self._sd.stop()
                self._sd.play(samples, config.SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                print(f"[gong_audio] Wiedergabefehler: {exc}")

    def close(self) -> None:
        self._queue.put(_SENTINEL)
        if self._available and self._sd is not None:
            try:
                self._sd.stop()
            except Exception:  # noqa: BLE001
                pass
        self._thread.join(timeout=1.0)

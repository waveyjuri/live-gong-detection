"""Synthetischer Gong-Klang + polyphone, non-blocking Wiedergabe.

Der Gong wird aus inharmonischen, exponentiell abklingenden Sinus-Obertoenen
erzeugt (metallischer/blecherner Klang). Die Wiedergabe laeuft ueber einen
sounddevice-OutputStream mit Misch-Callback: jeder Schlag fuegt eine "Stimme"
hinzu, mehrere Gongs klingen gleichzeitig und ueberlagern sich, ohne sich
gegenseitig abzubrechen. Der Video-Loop blockiert dabei nie.
"""

import threading

import numpy as np

import config

# Maximale Anzahl gleichzeitig klingender Gongs (Schutz vor Ueberlagerungs-Chaos)
MAX_VOICES = 12


def synth_gong_base() -> np.ndarray:
    """Erzeugt den Basis-Gong (Volume 1.0), einmalig zu cachen.

    China-Style: aufsteigender Pitch-Glide + metallisches Rausch-Schimmern.
    """
    sr = config.SAMPLE_RATE
    dt = 1.0 / sr
    t = np.linspace(0.0, config.GONG_DURATION,
                    int(sr * config.GONG_DURATION), endpoint=False)

    # Glide-Faktor: gleitet exponentiell von START nach END (Tonhoehe steigt).
    glide = config.GONG_GLIDE_END + (config.GONG_GLIDE_START - config.GONG_GLIDE_END) \
        * np.exp(-t / config.GONG_GLIDE_TAU)

    signal = np.zeros_like(t)
    for ratio, amp, decay in zip(config.GONG_PARTIAL_RATIOS,
                                 config.GONG_PARTIAL_AMPS,
                                 config.GONG_PARTIAL_DECAYS):
        # zeitvariable Frequenz -> Momentanphase als Integral (cumsum) der Frequenz
        inst_freq = config.GONG_BASE_FREQ * ratio * glide
        phase = 2.0 * np.pi * np.cumsum(inst_freq) * dt
        env = np.exp(-t / decay)
        # Bloom/Swell: Amplitude steigt erst an, dann Abklingen -> Tam-Tam-Charakter
        if config.GONG_BLOOM_TAU > 0:
            env = env * (1.0 - np.exp(-t / config.GONG_BLOOM_TAU))
        signal += amp * env * np.sin(phase)

    # metallisches Schimmern: schnell abklingender Rauschanteil beim Anschlag
    if config.GONG_SHIMMER_AMP > 0:
        rng = np.random.default_rng(42)   # fester Seed -> reproduzierbarer Klang
        noise = rng.standard_normal(t.shape)
        shimmer_env = np.exp(-t / config.GONG_SHIMMER_DECAY)
        signal += config.GONG_SHIMMER_AMP * shimmer_env * noise

    # kurzer Attack-Fade gegen Knacken am Anfang
    attack = int(sr * config.GONG_ATTACK_SEC)
    if attack > 0:
        signal[:attack] *= np.linspace(0.0, 1.0, attack)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return (signal * config.MASTER_GAIN).astype(np.float32)


def synth_kick_base(semitones: float = 0.0) -> np.ndarray:
    """Erzeugt einen Hardtechno-Bass (Volume 1.0), einmalig zu cachen.

    Pitch-Sweep gibt den Punch, tanh-Saturation + Hard-Clipping machen ihn
    bretthart. ``semitones`` verschiebt die Tonhoehe (fuer die Bassline-Folge).
    """
    sr = config.SAMPLE_RATE
    dur = config.KICK_DURATION
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)

    # Halbton-Versatz -> Frequenzfaktor (gleichstufige Stimmung)
    pitch = 2.0 ** (semitones / 12.0)

    # Tonhoehe gleitet schnell von START -> END (klassischer Kick-"Punch")
    f = (config.KICK_END_FREQ + (config.KICK_START_FREQ - config.KICK_END_FREQ)
         * np.exp(-t / config.KICK_PITCH_TAU)) * pitch
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    amp_env = np.exp(-t / config.KICK_DECAY)
    # Grundton + 2. Harmonische: der Bass bleibt auch auf kleinen Lautsprechern
    # praesent (Effekt der "fehlenden Grundfrequenz").
    body = np.sin(phase) + config.KICK_HARM2_AMP * np.sin(2.0 * phase)
    signal = body * amp_env

    # Klick-Transient: kurzer Rauschimpuls fuer den "harten" Anschlag
    if config.KICK_CLICK_AMP > 0:
        click_len = max(int(sr * config.KICK_CLICK_SEC), 1)
        rng = np.random.default_rng(7)
        click = rng.standard_normal(click_len) * np.linspace(1.0, 0.0, click_len)
        signal[:click_len] += config.KICK_CLICK_AMP * click

    # Saturation -> druckvoll/hart (hoher Drive treibt den Sinus Richtung Square)
    if config.KICK_DRIVE > 1.0:
        signal = np.tanh(signal * config.KICK_DRIVE)

    # zusaetzliches Hard-Clipping: kappt die Spitzen -> brettartiger Hardtechno-Bass
    if 0.0 < config.KICK_HARDCLIP < 1.0:
        peak = np.max(np.abs(signal))
        if peak > 0:
            lim = config.KICK_HARDCLIP * peak
            np.clip(signal, -lim, lim, out=signal)

    attack = int(sr * config.KICK_ATTACK_SEC)
    if attack > 0:
        signal[:attack] *= np.linspace(0.0, 1.0, attack)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return (signal * config.KICK_GAIN).astype(np.float32)


class GongPlayer:
    """Polyphone Wiedergabe ueber einen einzigen gemischten OutputStream."""

    def __init__(self) -> None:
        self._base = synth_gong_base()
        # Bassline: pro Schlag durchlaufene Tonfolge, Varianten einmal vorab erzeugen
        self._kick_seq = list(config.KICK_SEQUENCE) or [0]
        semis = set(self._kick_seq) | {config.KICK_MELODY_FILL_SEMI}
        self._kick_variants = {s: synth_kick_base(s) for s in semis}
        self._kick_idx = 0
        self._last_melody_time = -1e9   # fuer den Melodie-Mindestabstand
        self._voices: list[list] = []   # je Stimme: [samples, position]
        self._lock = threading.Lock()
        self._stream = None
        self._available = False
        try:
            import sounddevice as sd
            self._sd = sd
            self._stream = sd.OutputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - Audio darf optional fehlen
            print(f"[gong_audio] Audio nicht verfuegbar ({exc}); laeuft ohne Ton.")

    @property
    def available(self) -> bool:
        return self._available

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Mischt alle aktiven Stimmen fuer den naechsten Audio-Block."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            still_active = []
            for voice in self._voices:
                samples, pos = voice
                chunk = samples[pos:pos + frames]
                out[:len(chunk)] += chunk
                new_pos = pos + len(chunk)
                if new_pos < len(samples):
                    voice[1] = new_pos
                    still_active.append(voice)
            self._voices = still_active
        # Summe mehrerer Gongs kann > 1.0 werden -> hart begrenzen gegen Verzerrung
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out

    def _add_voice(self, samples: np.ndarray) -> None:
        """Stimme in den Misch-Buffer einreihen (parallele Wiedergabe)."""
        with self._lock:
            if len(self._voices) >= MAX_VOICES:
                # aelteste Stimme verdraengen, damit es nicht unbegrenzt waechst
                self._voices.pop(0)
            self._voices.append([samples, 0])

    def play(self, volume: float) -> None:
        """Neuen Gong (Lautstaerke 0..1) hinzufuegen; klingt parallel zu laufenden."""
        if not self._available:
            return
        self._add_voice((self._base * float(volume)).astype(np.float32))

    def play_kick(self, volume: float, now: float) -> None:
        """Bass abfeuern; rueckt die Melodie nur bei genug Abstand weiter.

        Liegt seit der letzten Melodie-Note >= KICK_MELODY_MIN_GAP, wird die
        naechste Note der Folge gespielt; schnellere Fuell-Schlaege bleiben auf
        dem Grundton und treiben die Melodie nicht durch.
        """
        if not self._available:
            return
        if now - self._last_melody_time >= config.KICK_MELODY_MIN_GAP:
            semi = self._kick_seq[self._kick_idx % len(self._kick_seq)]
            self._kick_idx += 1
            self._last_melody_time = now
        else:
            semi = config.KICK_MELODY_FILL_SEMI
        base = self._kick_variants[semi]
        self._add_voice((base * float(volume)).astype(np.float32))

    def reset_kick_sequence(self) -> None:
        """Tonfolge auf den Anfang setzen (z.B. bei neuem Kickroll)."""
        self._kick_idx = 0
        self._last_melody_time = -1e9

    def close(self) -> None:
        if self._available and self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass

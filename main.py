"""Live-Gong-Detection: Webcam -> Personen-/Pose-Erkennung -> Schlag -> Gong.

Beenden mit 'q' oder ESC.
"""

import sys
import time

import cv2

import config
import visualizer
from gong_audio import GongPlayer
from pose_detector import PoseDetector
from strike_detector import StrikeDetector


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main() -> int:
    print("Lade YOLO-Pose-Modell (erster Start kann dauern)...")
    detector = PoseDetector()
    strike = StrikeDetector()
    gong = GongPlayer()

    cap = open_camera()
    if not cap.isOpened():
        print(f"FEHLER: Webcam (Index {config.CAMERA_INDEX}) nicht verfuegbar.")
        gong.close()
        return 1

    print("Bereit. Schlag-Bewegung vor der Kamera ausfuehren. q/ESC zum Beenden.")
    last_trigger_volume = 0.0
    last_trigger_time = -1e9

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("FEHLER: Kein Frame von der Kamera.")
                break
            if config.MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            now = time.time()
            persons = detector.infer(frame)
            result = strike.process(persons, now)

            if result.triggered:
                gong.play(result.volume)
                last_trigger_volume = result.volume
                last_trigger_time = now
                print(f"GONG  v_peak={result.peak_velocity:5.2f}  "
                      f"vol={result.volume:.2f}")

            primary = max(persons, key=lambda p: p.bbox_area) if persons else None
            visualizer.draw(frame, primary, result,
                            last_trigger_volume, last_trigger_time)

            cv2.imshow("Live Gong Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q oder ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gong.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

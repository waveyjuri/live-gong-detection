"""Duenner Wrapper um Ultralytics YOLO-Pose.

Liefert pro Frame eine Liste erkannter Personen mit Bounding-Box,
Keypoints (COCO, 17 Punkte) und deren Confidences.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Person:
    """Eine erkannte Person."""
    bbox: tuple          # (x1, y1, x2, y2) in Pixeln
    conf: float          # Personen-Confidence
    keypoints: np.ndarray   # shape (17, 2) -> (x, y) in Pixeln
    kp_conf: np.ndarray     # shape (17,)  -> Confidence pro Keypoint

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class PoseDetector:
    def __init__(self) -> None:
        # Modell wird beim ersten Aufruf automatisch heruntergeladen.
        self.model = YOLO(config.MODEL_NAME)

    def infer(self, frame: np.ndarray) -> list[Person]:
        """Fuehrt eine Inferenz auf einem BGR-Frame aus."""
        results = self.model.predict(
            frame,
            imgsz=config.INFER_IMGSZ,
            conf=config.PERSON_CONF,
            verbose=False,
        )
        persons: list[Person] = []
        if not results:
            return persons

        r = results[0]
        if r.keypoints is None or r.boxes is None:
            return persons

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_conf = r.boxes.conf.cpu().numpy()
        kps_xy = r.keypoints.xy.cpu().numpy()          # (n, 17, 2)
        # Keypoint-Confidence ist optional vorhanden
        if r.keypoints.conf is not None:
            kps_conf = r.keypoints.conf.cpu().numpy()  # (n, 17)
        else:
            kps_conf = np.ones(kps_xy.shape[:2], dtype=float)

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            persons.append(
                Person(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(boxes_conf[i]),
                    keypoints=kps_xy[i],
                    kp_conf=kps_conf[i],
                )
            )
        return persons

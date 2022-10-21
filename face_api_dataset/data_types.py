from typing import Tuple, Dict
import numpy as np
from enum import Enum

Item = dict
InstanceId = int
LandmarkId = int
Landmark_2D = Tuple[float, float]
Landmark_3D = Tuple[float, float, float]

class OutOfFrameLandmarkStrategy(Enum):
    IGNORE = "ignore"
    CLIP = "clip"

    @staticmethod
    def clip_landmarks_(
        landmarks: Dict[LandmarkId, Landmark_2D], height: int, width: int
    ) -> Dict[LandmarkId, Landmark_2D]:
        clipped_landmarks: Dict[LandmarkId, Landmark_2D] = {}
        for name, (x, y) in landmarks.items():
            xx = np.clip(x, 0, width - 1)
            yy = np.clip(y, 0, height - 1)
            clipped_landmarks[name] = (float(xx), float(yy))
        return clipped_landmarks
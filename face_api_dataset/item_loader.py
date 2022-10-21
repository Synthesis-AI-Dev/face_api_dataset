from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union
from face_api_dataset.modality import Modality
from face_api_dataset.data_types import OutOfFrameLandmarkStrategy
import numpy as np

N_IBUG68_LANDMARKS = 68


class _ItemLoader(ABC):
    def __init__(
        self,
        root: Path,
        modalities: List[Modality],
        metadata_records: List[Dict[str, Union[int, str]]],
        out_of_frame_landmark_strategy: OutOfFrameLandmarkStrategy,
        body_segmentation_mapping: Dict[str, int],
        face_segmentation_classes: List[str],
        face_bbox_pad: int,
    ):
        self._modalities = modalities
        self._out_of_frame_landmark_strategy = out_of_frame_landmark_strategy
        self._body_segmentation_mapping = body_segmentation_mapping
        self._face_segmentation_classes = face_segmentation_classes
        self._face_bbox_pad = face_bbox_pad
        self._image_sizes: Dict[tuple, Tuple[int, int]] = {}

    @abstractmethod
    def get_item(self):
        return NotImplemented

    @staticmethod
    def _hom_to_euclidian(x):
        return x[:, :2] / np.expand_dims(x[:, 2], -1)

    @classmethod
    def get_euler_angles(
        cls, matrix: np.ndarray, order: str, degrees: bool
    ) -> np.ndarray:
        """
        Euler angles by matrix 4x4.

        :param np.ndarray matrix:  matrix 4x4.
        :param str order:  axes order ('xyz','yzx','zxy').
        :param bool degrees:  whether use degrees or radians.

        """
        from scipy.spatial.transform.rotation import Rotation

        return Rotation.from_matrix(matrix[:3, :3]).as_euler(order, degrees=degrees)

    @classmethod
    def get_quaternion(cls, matrix: np.ndarray) -> np.ndarray:
        """
        Quaternion by matrix 4x4.

        :param np.ndarray matrix:  matrix 4x4

        """
        from scipy.spatial.transform.rotation import Rotation

        return Rotation.from_matrix(matrix[:3, :3]).as_quat()

    @classmethod
    def get_shift(cls, matrix: np.ndarray) -> np.ndarray:
        """
        Return shift vector by matrix 4x4 which contains both rotation and shift.

        :param np.ndarray matrix:  matrix 4x4

        """
        return matrix[:3, 3]

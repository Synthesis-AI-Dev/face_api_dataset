import enum
import json
from pathlib import Path
from face_api_dataset.data_types import (
    InstanceId,
    LandmarkId,
    Item,
    Landmark_2D,
    Landmark_3D,
    OutOfFrameLandmarkStrategy,
)
from face_api_dataset.item_loader import N_IBUG68_LANDMARKS, _ItemLoader
from face_api_dataset.modality import Modality
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import tiffile
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class _Extension(str, Enum):
    INFO = "info.json"
    RGB = "rgb.png"
    NORMALS = "normals.tif"
    DEPTH = "depth.tif"
    ALPHA = "alpha.tif"
    SEGMENTS = "segments.png"
    MEDIAPIPE_DENSE_OBJ = "mediapipe_dense.obj"
    SAI_DENSE_OBJ = "sai_dense.obj"


def _modality_files(modality: Modality) -> List[_Extension]:
    return {
        Modality.SCENE_ID: [_Extension.INFO],
        Modality.RGB: [_Extension.RGB],
        Modality.NORMALS: [_Extension.NORMALS],
        Modality.DEPTH: [_Extension.DEPTH],
        Modality.ALPHA: [_Extension.ALPHA],
        Modality.BODY_SEGMENTATION: [_Extension.INFO, _Extension.SEGMENTS],
        Modality.CLOTHING_SEGMENTATION: None,
        Modality.INSTANCE_SEGMENTATION: [_Extension.INFO, _Extension.SEGMENTS],
        Modality.LANDMARKS_IBUG68: [_Extension.INFO],
        Modality.LANDMARKS_CONTOUR_IBUG68: [_Extension.INFO],
        Modality.LANDMARKS_KINECT_V2: [_Extension.INFO],
        Modality.LANDMARKS_COCO: [_Extension.INFO],
        Modality.LANDMARKS_MEDIAPIPE: [_Extension.INFO],
        Modality.LANDMARKS_MPEG4: [_Extension.INFO],
        Modality.LANDMARKS_3D_IBUG68: [_Extension.INFO],
        Modality.LANDMARKS_3D_KINECT_V2: [_Extension.INFO],
        Modality.LANDMARKS_3D_COCO: [_Extension.INFO],
        Modality.LANDMARKS_3D_MEDIAPIPE: [_Extension.INFO],
        Modality.LANDMARKS_3D_MPEG4: [_Extension.INFO],
        Modality.PUPILS: [_Extension.INFO],
        Modality.PUPILS_3D: [_Extension.INFO],
        Modality.IDENTITY: [_Extension.INFO],
        Modality.IDENTITY_METADATA: [_Extension.INFO],
        Modality.HAIR: [_Extension.INFO],
        Modality.FACIAL_HAIR: [_Extension.INFO],
        Modality.EXPRESSION: [_Extension.INFO],
        Modality.GAZE: [_Extension.INFO],
        Modality.FACE_BBOX: [_Extension.RGB, _Extension.INFO, _Extension.SEGMENTS],
        Modality.CAM_TO_HEAD: [_Extension.INFO],
        Modality.HEAD_TO_CAM: [_Extension.INFO],
        Modality.WORLD_TO_HEAD: [_Extension.INFO],
        Modality.HEAD_TO_WORLD: [_Extension.INFO],
        Modality.CAM_TO_WORLD: [_Extension.INFO],
        Modality.WORLD_TO_CAM: [_Extension.INFO],
        Modality.CAM_INTRINSICS: [_Extension.INFO],
        Modality.CAMERA_NAME: [_Extension.INFO],
        Modality.FRAME_NUM: [_Extension.INFO],
        Modality.LANDMARKS_3D_MEDIAPIPE_FACE: [
            _Extension.MEDIAPIPE_DENSE_OBJ,
            _Extension.INFO,
        ],
        Modality.LANDMARKS_3D_SAI: [_Extension.SAI_DENSE_OBJ, _Extension.INFO],
        Modality.LANDMARKS_MEDIAPIPE_FACE: [
            _Extension.MEDIAPIPE_DENSE_OBJ,
            _Extension.INFO,
        ],
        Modality.LANDMARKS_SAI: [_Extension.SAI_DENSE_OBJ, _Extension.INFO],
    }[modality]


class _ItemLoaderV1(_ItemLoader):
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
        super().__init__(
            root,
            modalities,
            metadata_records,
            out_of_frame_landmark_strategy,
            body_segmentation_mapping,
            face_segmentation_classes,
            face_bbox_pad,
        )
        filtered_modalities = []
        for m in self._modalities:
            if _modality_files(m) is None:
                logger.warn(f"Ignoring unsupported modality {m}")
            else:
                filtered_modalities.append(m)
        self._modalities = filtered_modalities
        self._needs_info = False
        for modality in self._modalities:
            if _Extension.INFO in _modality_files(modality):
                self._needs_info = True

        for record in metadata_records:
            scene_id = record["SCENE_ID"]
            camera = record["CAMERA"]
            frame = record["FRAME"]

            for modality in self._modalities:
                for extension in _modality_files(modality):
                    expected_file = (
                        root / f"{scene_id}.{camera}.{frame}.{extension.value}"
                    )
                    if not expected_file.exists():
                        raise ValueError(
                            f"Can't find file '{expected_file}' required for {modality.name} modality."
                        )

    def get_item(
        self, element_idx: Tuple[int, str, int], item_meta: pd.DataFrame
    ) -> Item:
        info = None
        if self._needs_info:
            info_file = Path(
                item_meta[item_meta.EXTENSION == _Extension.INFO].file_path.iloc[0]
            )
            with info_file.open("r") as f:
                info = json.load(f)

        ret: Item = {}
        for modality in self._modalities:
            ret[modality] = self._open_modality(modality, item_meta, element_idx, info)
        return ret

    def _open_modality(
        self,
        modality: Modality,
        item_meta: pd.DataFrame,
        element_idx: Tuple[int, str, int],  # (scene id, camera name, frame number)
        info: Optional[dict],
    ) -> Any:
        scene_id = item_meta.SCENE_ID.iloc[0]
        if modality == Modality.SCENE_ID:
            return scene_id

        if modality == Modality.RGB:
            file_path = item_meta[item_meta.EXTENSION == _Extension.RGB].file_path.iloc[
                0
            ]
            return self._read_rgb(file_path, element_idx)

        if modality == Modality.BODY_SEGMENTATION:
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.SEGMENTS
            ].file_path.iloc[0]
            segment_img, _ = self._read_body_segmentation(file_path, element_idx, info)
            return segment_img

        if modality == Modality.INSTANCE_SEGMENTATION:
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.SEGMENTS
            ].file_path.iloc[0]
            segment_img = self._read_instance_segmentation(file_path, element_idx, info)
            return segment_img

        if modality == Modality.NORMALS:
            normals_file = item_meta[
                item_meta.EXTENSION == _Extension.NORMALS
            ].file_path.iloc[0]
            img = tiffile.imread(str(normals_file))
            if img is None:
                raise ValueError(f"Error reading {normals_file}")
            if element_idx in self._image_sizes:
                if self._image_sizes[element_idx] != img.shape[1::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for SCENE_ID={scene_id}"
                    )
            else:
                self._image_sizes[element_idx] = img.shape[1::-1]
            return img

        if modality == Modality.ALPHA:
            alpha_file = item_meta[
                item_meta.EXTENSION == _Extension.ALPHA
            ].file_path.iloc[0]
            img = tiffile.imread(str(alpha_file))
            if element_idx in self._image_sizes:
                if self._image_sizes[element_idx] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for SCENE_ID={scene_id}"
                    )
            else:
                self._image_sizes[element_idx] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {alpha_file}")
            return img

        if modality == Modality.DEPTH:
            depth_file = item_meta[
                item_meta.EXTENSION == _Extension.DEPTH
            ].file_path.iloc[0]
            img = tiffile.imread(str(depth_file))
            if element_idx in self._image_sizes:
                if self._image_sizes[element_idx] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for SCENE_ID={scene_id}"
                    )
            else:
                self._image_sizes[element_idx] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {depth_file}")
            return img

        if modality in (Modality.LANDMARKS_IBUG68, Modality.LANDMARKS_CONTOUR_IBUG68):
            return self._read_face_landmarks_2d(info, modality, element_idx)

        if modality in (
            Modality.LANDMARKS_KINECT_V2,
            Modality.LANDMARKS_COCO,
            Modality.LANDMARKS_MEDIAPIPE,
            Modality.LANDMARKS_MPEG4,
        ):
            landmark_meta = self._read_body_landmarks(
                info, modality, self._image_sizes[element_idx]
            )
            return landmark_meta

        if modality in (
            Modality.LANDMARKS_3D_IBUG68,
            Modality.LANDMARKS_3D_KINECT_V2,
            Modality.LANDMARKS_3D_COCO,
            Modality.LANDMARKS_3D_MEDIAPIPE,
            Modality.LANDMARKS_3D_MPEG4,
        ):
            landmark_meta = self._read_landmarks_3d(info, modality)
            return landmark_meta

        if modality == Modality.PUPILS:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["screen_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["screen_space_pos"],
            ]
            if element_idx not in self._image_sizes:
                raise ValueError(
                    "Pupils can only be loaded with at least one image modality"
                )
            scale = self._image_sizes[element_idx]
            pupils_np = np.array(pupils, dtype=np.float64) * scale
            pupils = {
                "pupil_left": tuple(pupils_np[0]),
                "pupil_right": tuple(pupils_np[1]),
            }

            # v1 only has one character with ID 0
            pupils = {0: pupils}
            return pupils

        if modality == Modality.PUPILS_3D:
            pupils = {
                "pupil_left": tuple(
                    info["pupil_coordinates"]["pupil_left"]["camera_space_pos"]
                ),
                "pupil_right": tuple(
                    info["pupil_coordinates"]["pupil_right"]["camera_space_pos"]
                ),
            }

            # v1 only has one character with ID 0
            pupils = {0: pupils}
            return pupils

        if modality == Modality.IDENTITY:
            identity = info["identity_metadata"]["id"]
            # v1 only has one character with ID 0
            identity = {0: identity}
            return identity

        if modality == Modality.IDENTITY_METADATA:
            metadata = info["identity_metadata"]
            metadata = {0: metadata}
            return metadata

        if modality == Modality.HAIR:
            if "hair" in info["facial_attributes"]:
                hair = info["facial_attributes"]["hair"]
            else:
                hair = None

            # v1 only has one character with ID 0
            hair = {0: hair}
            return hair

        if modality == Modality.FACIAL_HAIR:
            if "facial_hair" in info["facial_attributes"]:
                facial_hair = info["facial_attributes"]["facial_hair"]
            else:
                facial_hair = None

            # v1 only has one character with ID 0
            facial_hair = {0: facial_hair}
            return facial_hair

        if modality == Modality.EXPRESSION:
            expression = info["facial_attributes"]["expression"]

            # v1 only has one character with ID 0
            expression = {0: expression}
            return expression

        if modality == Modality.GAZE:
            gaze = {
                "right": info["gaze_values"]["eye_right"]["gaze_vector"],
                "left": info["gaze_values"]["eye_left"]["gaze_vector"],
            }

            # v1 only has one character with ID 0
            gaze = {0: gaze}
            return gaze

        if modality == Modality.FACE_BBOX:
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.SEGMENTS
            ].file_path.iloc[0]
            segment_img, segment_mapping_int = self._read_body_segmentation(
                file_path, element_idx, info
            )

            segment_mapping = info["segments_mapping"]
            face_segments = set(self._face_segmentation_classes) & set(
                segment_mapping.keys()
            )
            face_seg_idxs = [
                segment_mapping_int[segment_mapping[s]] for s in face_segments
            ]
            face_mask = np.isin(segment_img, face_seg_idxs).astype(np.uint16)

            def get_bbox(img: np.ndarray):
                yxs = np.where(img != 0)
                if len(yxs[0]) == 0:
                    return (-1, -1, -1, -1)
                bbox = np.min(yxs[1]), np.min(yxs[0]), np.max(yxs[1]), np.max(yxs[0])
                return bbox

            height, width = self._image_sizes[element_idx]

            def expand_bbox(bbox: Tuple[int, int, int, int], padding: int):
                x0, y0, x1, y1 = bbox
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(x1 + padding, width)
                y1 = min(y1 + padding, height)

                return (x0, y0, x1, y1)

            face_bbox = get_bbox(face_mask)
            expanded_face_bbox = expand_bbox(face_bbox, self._face_bbox_pad)

            # v1 only has one character with ID 0
            expanded_face_bbox = {0: expanded_face_bbox}

            return expanded_face_bbox

        if modality == Modality.CAM_TO_WORLD:
            return np.array(
                info["camera"]["transform_cam2world"]["mat_4x4"], dtype=np.float64
            )

        if modality == Modality.WORLD_TO_CAM:
            return np.array(
                info["camera"]["transform_world2cam"]["mat_4x4"], dtype=np.float64
            )

        if modality == Modality.HEAD_TO_CAM:
            head2cam = np.array(
                info["head_transform"]["transform_head2cam"]["mat_4x4"],
                dtype=np.float64,
            )

            # v1 only has one character with ID 0
            head2cam = {0: head2cam}
            return head2cam

        if modality == Modality.CAM_TO_HEAD:
            head2cam = self._open_modality(
                Modality.HEAD_TO_CAM, item_meta, element_idx, info
            )
            cam2head = {}
            for instance_id, h2c in head2cam.items():
                cam2head[instance_id] = np.linalg.inv(h2c)

            return cam2head

        if modality == Modality.HEAD_TO_WORLD:
            head2world = np.array(
                info["head_transform"]["transform_head2world"]["mat_4x4"],
                dtype=np.float64,
            )

            # v1 only has one character with ID 0
            head2world = {0: head2world}
            return head2world

        if modality == Modality.WORLD_TO_HEAD:
            head2world = self._open_modality(
                Modality.HEAD_TO_WORLD, item_meta, element_idx, info
            )
            world2head = {}
            for instance_id, h2w in head2world.items():
                world2head[instance_id] = np.linalg.inv(h2w)

            return world2head

        if modality == Modality.CAM_INTRINSICS:
            return np.array(info["camera"]["intrinsics"], dtype=np.float64)

        if modality == Modality.CAMERA_NAME:
            return item_meta.CAMERA_NAME.iloc[0]

        if modality == Modality.FRAME_NUM:
            return item_meta.FRAME_NUM.iloc[0]

        if modality == Modality.LANDMARKS_3D_MEDIAPIPE_FACE:
            m = np.array(
                info["camera"]["transform_world2cam"]["mat_4x4"], dtype=np.float64
            )
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.MEDIAPIPE_DENSE_OBJ
            ].file_path.iloc[0]
            return self._read_obj(file_path, m)

        if modality == Modality.LANDMARKS_3D_SAI:
            m = np.array(
                info["camera"]["transform_world2cam"]["mat_4x4"], dtype=np.float64
            )
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.SAI_DENSE_OBJ
            ].file_path.iloc[0]
            return self._read_obj(file_path, m)

        if modality == Modality.LANDMARKS_SAI:
            sai_3d = self._open_modality(
                Modality.LANDMARKS_3D_SAI, item_meta, element_idx, info
            )
            intrinsics = self._open_modality(
                Modality.CAM_INTRINSICS, item_meta, element_idx, info
            )

            landmarks_sai = {}
            for instance_id, landmarks in sai_3d.items():
                landmarks_sai[instance_id] = dict(
                    enumerate(
                        self._hom_to_euclidian(
                            np.tensordot(landmarks * [1, -1, -1], intrinsics, axes=(-1, 1))
                        )
                    )
                )
            return landmarks_sai

        if modality == Modality.LANDMARKS_MEDIAPIPE_FACE:
            mediapipe_face_3d = self._open_modality(
                Modality.LANDMARKS_3D_MEDIAPIPE_FACE, item_meta, element_idx, info
            )
            intrinsics = self._open_modality(
                Modality.CAM_INTRINSICS, item_meta, element_idx, info
            )

            face_landmarks = {}
            for instance_id, landmarks in mediapipe_face_3d.items():
                face_landmarks[instance_id] = dict(
                    enumerate(
                        self._hom_to_euclidian(
                            np.tensordot(
                                landmarks * [1, -1, -1], intrinsics, axes=(-1, 1)
                            )
                        )
                    )
                )
            return face_landmarks

        raise ValueError("Unknown modality")

    def _read_body_segmentation(
        self, body_segmentation_file: str, element_idx: tuple, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(body_segmentation_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Error reading {body_segmentation_file}")
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[::-1]:
                raise ValueError(
                    f"Dimensions of different image modalities do not match."
                )
        else:
            self._image_sizes[element_idx] = img.shape[::-1]

        body_segmentation_mapping = info["segments_mapping"]
        body_segmentation_mapping_int = np.full(
            np.max(list(body_segmentation_mapping.values())) + 1,
            self._body_segmentation_mapping["default"],
            dtype=np.uint8,
        )
        for key, value in body_segmentation_mapping.items():
            if key in self._body_segmentation_mapping:
                body_segmentation_mapping_int[value] = self._body_segmentation_mapping[
                    key
                ]

        segment_img = body_segmentation_mapping_int[img]
        return segment_img, body_segmentation_mapping_int

    def _read_instance_segmentation(
        self, body_segmentation_file: str, element_idx: tuple, info: Dict[str, Any]
    ) -> np.ndarray:
        img = cv2.imread(str(body_segmentation_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Error reading {body_segmentation_file}")
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[::-1]:
                raise ValueError(
                    f"Dimensions of different image modalities do not match."
                )
        else:
            self._image_sizes[element_idx] = img.shape[::-1]

        segment_mapping = info["segments_mapping"]
        assert "background" in segment_mapping

        background_index = segment_mapping["background"]
        instance_img = img.copy()
        # there is only one character in v1, which has index 0
        instance_img[img == background_index] = -1
        instance_img[img != background_index] = 0

        return instance_img

    def _read_rgb(self, rgb_file: str, element_idx: tuple) -> np.ndarray:
        img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Error reading {rgb_file}")
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[1::-1]:
                raise ValueError("Dimensions of different modalities do not match")
        else:
            self._image_sizes[element_idx] = img.shape[1::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def _read_modality_meta(info: dict, mdt: Modality) -> dict:
        if mdt in (Modality.LANDMARKS_IBUG68, Modality.LANDMARKS_3D_IBUG68):
            meta_dict = info["landmarks"]["ibug68"]
        elif mdt is Modality.LANDMARKS_CONTOUR_IBUG68:
            meta_dict = info["landmarks"]["ibug68_contour"]
        elif mdt in (Modality.LANDMARKS_COCO, Modality.LANDMARKS_3D_COCO):
            meta_dict = info["landmarks"]["coco"]["whole_body"]
        elif mdt in (Modality.LANDMARKS_KINECT_V2, Modality.LANDMARKS_3D_KINECT_V2):
            meta_dict = info["landmarks"]["kinect_v2"]
        elif mdt in (Modality.LANDMARKS_MEDIAPIPE, Modality.LANDMARKS_3D_MEDIAPIPE):
            meta_dict = info["landmarks"]["mediapipe"]["body"]
        elif mdt in (Modality.LANDMARKS_MPEG4, Modality.LANDMARKS_3D_MPEG4):
            meta_dict = info["landmarks"]["mpeg4"]
        else:
            raise ValueError(f"Unrecognized modality for 3D landmarks {mdt}.")
        return meta_dict

    @classmethod
    def _read_body_landmarks(
        cls, info: dict, mdt: Modality, image_size: Tuple[int, int]
    ) -> Dict[InstanceId, Dict[LandmarkId, Landmark_2D]]:
        meta_dict = cls._read_modality_meta(info, mdt)
        result = {}

        for item in meta_dict:
            idd = item["id"]
            x, y = item["screen_space_pos"]
            w, h = image_size
            result[int(idd)] = (x * w, y * h)

        # v1 only has one character with ID 0
        result = {0: result}

        return result

    def _read_face_landmarks_2d(
        self, info: dict, mdt: Modality, element_idx: tuple
    ) -> Dict[InstanceId, Dict[LandmarkId, Landmark_2D]]:
        if element_idx not in self._image_sizes:
            raise ValueError(
                "Landmarks can only be loaded with at least one image modality"
            )

        meta_dict = self._read_modality_meta(info, mdt)

        if mdt is Modality.LANDMARKS_IBUG68:
            n_landmarks = len(meta_dict)
            if n_landmarks != N_IBUG68_LANDMARKS:
                msg = f"Error reading landmarks for item. Got {n_landmarks} landmarks instead of {N_IBUG68_LANDMARKS}"
                raise ValueError(msg)

        w, h = self._image_sizes[element_idx]
        landmarks: Dict[LandmarkId, Landmark_2D] = {}

        for landmark in meta_dict:
            x, y = landmark["screen_space_pos"]
            x, y = x * w, y * h
            landmarks[int(landmark["id"])] = (x, y)

        if self._out_of_frame_landmark_strategy is OutOfFrameLandmarkStrategy.CLIP:
            landmarks = OutOfFrameLandmarkStrategy.clip_landmarks_(landmarks, h, w)

        # v1 only has one character with ID 0
        landmarks = {0: landmarks}
        return landmarks

    @classmethod
    def _read_landmarks_3d(
        cls, info: dict, mdt: Modality
    ) -> Dict[LandmarkId, Landmark_3D]:
        meta_dict = cls._read_modality_meta(info, mdt)
        landmarks: Dict[LandmarkId, Landmark_3D] = {}

        for landmark in meta_dict:
            lmk_name = landmark["id"]
            landmarks[int(lmk_name)] = tuple(landmark["camera_space_pos"])

        # v1 only has one character with ID 0
        landmarks = {0: landmarks}
        return landmarks

    def _read_obj(self, obj_file: str, world_to_cam: np.ndarray):
        with open(obj_file, "r") as f:
            lines = f.readlines()
            result = np.empty((0, 3))
            for line in lines:
                if line.startswith("v "):
                    _, x, y, z = line.split()
                    x, y, z = float(x), float(y), float(z)
                    result = np.vstack((result, [x, y, z]))

        result = (
            np.tensordot(result, world_to_cam[:3, :3], axes=(-1, 1))
            + world_to_cam[:3, 3]
        )

        # v1 only has one character with ID 0
        result = {0: result}
        return result

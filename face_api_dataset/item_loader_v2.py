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
    INFO = "image_metadata.json"
    RGB = "rgb.npz"
    NORMALS = "surface_normals.npz"
    DEPTH = "depth.npz"
    BODY_SEGMENTATION = "body_segmentation.npz"
    CLOTHING_SEGMENTATION = "clothing_segmentation.npz"
    INSTANCE_SEGMENTATION = "instance_segmentation.npz"


def _modality_files(modality: Modality) -> List[_Extension]:
    return {
        Modality.SCENE_ID: [_Extension.INFO],
        Modality.RGB: [_Extension.RGB],
        Modality.NORMALS: [_Extension.NORMALS],
        Modality.DEPTH: [_Extension.DEPTH],
        Modality.ALPHA: None,
        Modality.BODY_SEGMENTATION: [_Extension.INFO, _Extension.BODY_SEGMENTATION],
        Modality.CLOTHING_SEGMENTATION: None,
        Modality.INSTANCE_SEGMENTATION: [_Extension.INSTANCE_SEGMENTATION],
        Modality.LANDMARKS_IBUG68: [_Extension.INFO],
        Modality.LANDMARKS_CONTOUR_IBUG68: None,
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
        Modality.FACIAL_HAIR: None,
        Modality.EXPRESSION: [_Extension.INFO],
        Modality.GAZE: [_Extension.INFO],
        Modality.FACE_BBOX: [
            _Extension.INFO,
            _Extension.BODY_SEGMENTATION,
            _Extension.INSTANCE_SEGMENTATION,
        ],
        Modality.CAM_TO_HEAD: [_Extension.INFO],
        Modality.HEAD_TO_CAM: [_Extension.INFO],
        Modality.WORLD_TO_HEAD: [_Extension.INFO],
        Modality.HEAD_TO_WORLD: [_Extension.INFO],
        Modality.CAM_TO_WORLD: [_Extension.INFO],
        Modality.WORLD_TO_CAM: [_Extension.INFO],
        Modality.CAM_INTRINSICS: [_Extension.INFO],
        Modality.CAMERA_NAME: [_Extension.INFO],
        Modality.FRAME_NUM: [_Extension.INFO],
        Modality.LANDMARKS_3D_MEDIAPIPE_FACE: None,
        Modality.LANDMARKS_3D_SAI: None,
        Modality.LANDMARKS_MEDIAPIPE_FACE: None,
        Modality.LANDMARKS_SAI: None,
    }[modality]


class _ItemLoaderV2(_ItemLoader):
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
                item_meta.EXTENSION == _Extension.BODY_SEGMENTATION
            ].file_path.iloc[0]
            segment_img, _ = self._read_body_segmentation(file_path, element_idx, info)
            return segment_img

        if modality == Modality.INSTANCE_SEGMENTATION:
            file_path = item_meta[
                item_meta.EXTENSION == _Extension.BODY_SEGMENTATION
            ].file_path.iloc[0]
            segment_img = self._read_instance_segmentation(file_path, element_idx)
            return segment_img

        if modality == Modality.NORMALS:
            normals_file = item_meta[
                item_meta.EXTENSION == _Extension.NORMALS
            ].file_path.iloc[0]
            img = np.load(normals_file)["arr_0"]
            if element_idx in self._image_sizes:
                if self._image_sizes[element_idx] != img.shape[1::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for SCENE_ID={scene_id}"
                    )
            else:
                self._image_sizes[element_idx] = img.shape[1::-1]
            return img

        if modality == Modality.DEPTH:
            depth_file = item_meta[
                item_meta.EXTENSION == _Extension.DEPTH
            ].file_path.iloc[0]
            img = np.load(depth_file)["arr_0"]
            print(img.shape, self._image_sizes[element_idx])
            if element_idx in self._image_sizes:
                if self._image_sizes[element_idx] != img.shape[1::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for SCENE_ID={scene_id}"
                    )
            else:
                self._image_sizes[element_idx] = img.shape[1::-1]

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
            if element_idx not in self._image_sizes:
                raise ValueError(
                    "Pupils can only be loaded with at least one image modality"
                )
            scale = self._image_sizes[element_idx]

            pupils = {}
            for human in info["humans"]:
                human_pupils = [
                    human["pupil_coordinates"]["pupil_left"]["screen_space_pos"],
                    human["pupil_coordinates"]["pupil_right"]["screen_space_pos"],
                ]

                pupils_np = np.array(human_pupils, dtype=np.float64) * scale
                human_pupils = {
                    "pupil_left": tuple(pupils_np[0]),
                    "pupil_right": tuple(pupils_np[1]),
                }

                pupils[human["instance_id"]] = human_pupils

            return pupils

        if modality == Modality.PUPILS_3D:
            pupils = {}
            for human in info["humans"]:
                human_pupils = {
                    "pupil_left": tuple(
                        human["pupil_coordinates"]["pupil_left"]["camera_space_pos"]
                    ),
                    "pupil_right": tuple(
                        human["pupil_coordinates"]["pupil_right"]["camera_space_pos"]
                    ),
                }
                pupils[human["instance_id"]] = human_pupils

            return pupils

        if modality == Modality.IDENTITY:
            identities = {
                human["instance_id"]: human["preset_id"] for human in info["humans"]
            }
            return identities

        if modality == Modality.IDENTITY_METADATA:
            metadata = {
                human["instance_id"]: human["identity_metadata"]
                for human in info["humans"]
            }
            return metadata

        if modality == Modality.HAIR:
            hair = {}

            for human in info["humans"]:
                hair[human["instance_id"]] = {
                    "style": human["identity_metadata"]["hair_style"]
                }

            return hair

        if modality == Modality.EXPRESSION:
            expression = {}

            for human in info["humans"]:
                expression[human["instance_id"]] = human["pose"]["expression"]

            return expression

        if modality == Modality.GAZE:
            gaze = {}

            for human in info["humans"]:
                gaze[human["instance_id"]] = {
                    "right": human["gaze_values"]["eye_right"]["gaze_vector"],
                    "left": human["gaze_values"]["eye_left"]["gaze_vector"],
                }

            return gaze

        if modality == Modality.FACE_BBOX:
            body_segmentation_file_path = item_meta[
                item_meta.EXTENSION == _Extension.BODY_SEGMENTATION
            ].file_path.iloc[0]
            (
                body_segmentation,
                body_segmentation_mapping_int,
            ) = self._read_body_segmentation(
                body_segmentation_file_path, element_idx, info
            )

            body_segmentation_mapping = info["body_segmentation_mapping"]
            face_segmentation_classes = set(
                self._face_segmentation_classes
            ).intersection(set(body_segmentation_mapping.keys()))
            face_seg_idxs = [
                body_segmentation_mapping_int[body_segmentation_mapping[s]]
                for s in face_segmentation_classes
            ]
            face_mask = np.isin(body_segmentation, face_seg_idxs).astype(np.uint16)

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

            instance_segmentation_file_path = item_meta[
                item_meta.EXTENSION == _Extension.INSTANCE_SEGMENTATION
            ].file_path.iloc[0]
            instance_segmentation = self._read_instance_segmentation(
                instance_segmentation_file_path, element_idx
            )

            face_bboxes = {}

            for human in info["humans"]:
                instance_id = human["instance_id"]
                human_face_mask = np.copy(face_mask)
                # we only care about the face mask for this particular instance
                human_face_mask[instance_segmentation != instance_id] = 0
                face_bbox = get_bbox(human_face_mask)
                expanded_face_bbox = expand_bbox(face_bbox, self._face_bbox_pad)

                face_bboxes[instance_id] = expanded_face_bbox

            return face_bboxes

        if modality == Modality.CAM_TO_WORLD:
            return np.array(
                info["camera"]["transform_cam2world"]["mat_4x4"], dtype=np.float64
            )

        if modality == Modality.WORLD_TO_CAM:
            return np.array(
                info["camera"]["transform_world2cam"]["mat_4x4"], dtype=np.float64
            )

        if modality == Modality.HEAD_TO_CAM:
            head2cam = {}

            for human in info["humans"]:
                head2cam[human["instance_id"]] = np.array(
                    human["head_transform"]["transform_head2cam"]["mat_4x4"],
                    dtype=np.float64,
                )

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
            head2world = {}

            for human in info["humans"]:
                head2world[human["instance_id"]] = np.array(
                    human["head_transform"]["transform_head2world"]["mat_4x4"],
                    dtype=np.float64,
                )

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
            return np.array(info["camera"]["intrinsics"]["mat_3x3"], dtype=np.float64)

        if modality == Modality.CAMERA_NAME:
            return item_meta.CAMERA_NAME.iloc[0]

        if modality == Modality.FRAME_NUM:
            return item_meta.FRAME_NUM.iloc[0]

        raise ValueError("Unknown modality")

    def _read_body_segmentation(
        self, body_segmentation_file: str, element_idx: tuple, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = np.load(body_segmentation_file)["arr_0"]
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[1::-1]:
                raise ValueError(
                    f"Dimensions of different image modalities do not match."
                )
        else:
            self._image_sizes[element_idx] = img.shape[1::-1]

        body_segmentation_mapping = info["body_segmentation_mapping"]
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

        body_segmentation = body_segmentation_mapping_int[img]
        return body_segmentation, body_segmentation_mapping_int

    def _read_instance_segmentation(
        self, instance_segmentation_file: str, element_idx: tuple
    ) -> np.ndarray:
        img = np.load(instance_segmentation_file)["arr_0"]
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[1::-1]:
                raise ValueError(
                    f"Dimensions of different image modalities do not match."
                )
        else:
            self._image_sizes[element_idx] = img.shape[1::-1]

        return img

    def _read_rgb(self, rgb_file: str, element_idx: tuple) -> np.ndarray:
        img = np.load(rgb_file)["arr_0"]
        if element_idx in self._image_sizes:
            if self._image_sizes[element_idx] != img.shape[1::-1]:
                raise ValueError("Dimensions of different modalities do not match")
        else:
            self._image_sizes[element_idx] = img.shape[1::-1]
        if img.shape[2] != 3:
            raise ValueError(
                f"Expected RGBA image to have 3 channels, got {img.shape[2]}"
            )

        return img

    @staticmethod
    def _read_modality_meta(info: dict, mdt: Modality) -> dict:
        if mdt in (Modality.LANDMARKS_IBUG68, Modality.LANDMARKS_3D_IBUG68):
            meta_dict = {
                human["instance_id"]: human["landmarks"]["ibug68"]
                for human in info["humans"]
            }
        elif mdt is Modality.LANDMARKS_CONTOUR_IBUG68:
            meta_dict = {
                human["instance_id"]: human["landmarks"]["ibug68_contour"]
                for human in info["humans"]
            }
        elif mdt in (Modality.LANDMARKS_COCO, Modality.LANDMARKS_3D_COCO):
            meta_dict = {
                human["instance_id"]: human["landmarks"]["coco_whole_body"]
                for human in info["humans"]
            }
        elif mdt in (Modality.LANDMARKS_KINECT_V2, Modality.LANDMARKS_3D_KINECT_V2):
            meta_dict = {
                human["instance_id"]: human["landmarks"]["kinect_v2"]
                for human in info["humans"]
            }
        elif mdt in (Modality.LANDMARKS_MEDIAPIPE, Modality.LANDMARKS_3D_MEDIAPIPE):
            meta_dict = {
                human["instance_id"]: human["landmarks"]["mediapipe"]["body"]
                for human in info["humans"]
            }
        elif mdt in (Modality.LANDMARKS_MPEG4, Modality.LANDMARKS_3D_MPEG4):
            meta_dict = {
                human["instance_id"]: human["landmarks"]["mpeg4_fba"]
                for human in info["humans"]
            }
        else:
            raise ValueError(f"Unrecognized modality for 3D landmarks {mdt}.")
        return meta_dict

    @classmethod
    def _read_body_landmarks(
        cls, info: dict, mdt: Modality, image_size: Tuple[int, int]
    ) -> Dict[InstanceId, Dict[LandmarkId, Landmark_2D]]:
        meta_dict = cls._read_modality_meta(info, mdt)
        result = {}

        for instance_id, landmarks in meta_dict.items():
            lm = {}
            for item in landmarks:
                idd = item["id"]
                x, y = item["screen_space_pos"]
                w, h = image_size
                lm[int(idd)] = (x * w, y * h)
            result[instance_id] = lm

        return result

    def _read_face_landmarks_2d(
        self, info: dict, mdt: Modality, element_idx: tuple
    ) -> Dict[InstanceId, Dict[LandmarkId, Landmark_2D]]:
        if element_idx not in self._image_sizes:
            raise ValueError(
                "Landmarks can only be loaded with at least one image modality"
            )

        w, h = self._image_sizes[element_idx]
        meta_dict = self._read_modality_meta(info, mdt)
        landmarks_2d: Dict[InstanceId, Dict[LandmarkId, Landmark_2D]] = {}

        for instance_id, landmarks in meta_dict.items():
            if mdt is Modality.LANDMARKS_IBUG68:
                n_landmarks = len(landmarks)
                if n_landmarks != N_IBUG68_LANDMARKS:
                    msg = f"Error reading landmarks for item. Got {n_landmarks} landmarks instead of {N_IBUG68_LANDMARKS}"
                    raise ValueError(msg)

            landmarks_2d[instance_id] = {}

            for landmark in landmarks:
                x, y = landmark["screen_space_pos"]
                x, y = x * w, y * h
                landmarks_2d[instance_id][int(landmark["id"])] = (x, y)

            if self._out_of_frame_landmark_strategy is OutOfFrameLandmarkStrategy.CLIP:
                landmarks_2d[instance_id] = OutOfFrameLandmarkStrategy.clip_landmarks_(
                    landmarks_2d[instance_id], h, w
                )

        return landmarks_2d

    @classmethod
    def _read_landmarks_3d(
        cls, info: dict, mdt: Modality
    ) -> Dict[LandmarkId, Landmark_3D]:
        meta_dict = cls._read_modality_meta(info, mdt)
        landmarks_3d: Dict[InstanceId, Dict[LandmarkId, Landmark_3D]] = {}

        for instance_id, landmarks in meta_dict.items():
            landmarks_3d[instance_id] = {}
            for landmark in landmarks:
                lmk_name = landmark["id"]
                landmarks_3d[instance_id][int(lmk_name)] = tuple(
                    landmark["camera_space_pos"]
                )

        return landmarks_3d

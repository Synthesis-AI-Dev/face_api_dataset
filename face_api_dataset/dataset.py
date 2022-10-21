import copy
import json
import os
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import List
from typing import (
    Optional,
    Any,
    Union,
    Dict,
    Callable,
    overload,
    TYPE_CHECKING,
    Tuple,
    Sequence,
)

import numpy as np
import pandas as pd
from pkg_resources import parse_version
from face_api_dataset.data_types import Item, OutOfFrameLandmarkStrategy
from face_api_dataset.item_loader_factory import _ItemLoaderFactory

from face_api_dataset.modality import Modality

CAM_NAME_PREFIX = "cam_"
FRAME_NUM_PREFIX = "f_"


if TYPE_CHECKING:
    Base = Sequence[dict]
else:
    Base = Sequence


class Grouping(Enum):
    """
    Different modalities of grouping Synthesis AI dataset.
    """

    NONE = "NONE"
    """
    Each image is treated independently.

    The size of the dataset is #scenes * #cameras * #frames  
    (assuming the same number of the cameras/frames per scene).
    """
    SCENE = "SCENE"
    """
    Items with the same scene are grouped into the list.

    The size of the dataset is #scenes. Each element is a `List[Item] with the same `SCENE_ID`.
    """
    CAMERA = "CAMERA"
    """
    Items with the same camera are grouped into the list.

    The size of the dataset is #cameras. Each element is a `List[Item] with the same `CAMERA_NAME`.
    """
    SCENE_CAMERA = "SCENE_CAMERA"
    """
    Items are grouped first by camera and then by scene.
    List of frames for a particular scene is indexed by `scene_id`.
    
    The size of the dataset is **#scenes** * **#cameras** , 
    each element is a `List[Item]` is a list of consecutive frames for given scene and camera.
    """


class FaceApiDataset(Base):
    """Synthesis AI dataset.

    This class provides access to all the modalities available in Synthesis AI generated datasets.
    """

    BODY_SEGMENTATION_MAPPING = {
        "default": 0,
        "background": 0,
        "beard": 1,
        "body": 2,
        "brow": 3,
        "cheek_left": 4,
        "cheek_right": 5,
        "chin": 6,
        "clothing": 7,
        "ear_left": 8,
        "ear_right": 9,
        "eye_left": 10,
        "eye_right": 11,
        "eyelashes": 12,
        "eyelid": 13,
        "eyes": 14,
        "forehead": 15,
        "glasses": 16,
        "hair": 17,
        "head": 18,
        "headphones": 19,
        "headwear": 20,
        "jaw": 21,
        "jowl": 22,
        "lip_lower": 23,
        "lip_upper": 24,
        "mask": 25,
        "mouth": 26,
        "mouthbag": 27,
        "mustache": 28,
        "neck": 29,
        "nose": 30,
        "nose_outer": 31,
        "nostrils": 32,
        "shoulders": 33,
        "smile_line": 34,
        "teeth": 35,
        "temples": 36,
        "tongue": 37,
        "undereye": 38,
        "eyebrows": 89,
        "torso_lower_left": 40,
        "torso_lower_right": 41,
        "torso_mid_left": 42,
        "torso_mid_right": 43,
        "torso_upper_left": 44,
        "torso_upper_right": 45,
        "arm_lower_left": 46,
        "arm_lower_right": 47,
        "arm_upper_left": 48,
        "arm_upper_right": 49,
        "hand_left": 50,
        "hand_right": 51,
        "finger1_mid_bottom_left": 52,
        "finger1_mid_bottom_right": 53,
        "finger1_mid_left": 54,
        "finger1_mid_right": 55,
        "finger1_mid_top_left": 56,
        "finger1_mid_top_right": 57,
        "finger2_mid_bottom_left": 58,
        "finger2_mid_bottom_right": 59,
        "finger2_mid_left": 60,
        "finger2_mid_right": 61,
        "finger2_mid_top_left": 62,
        "finger2_mid_top_right": 63,
        "finger3_mid_bottom_left": 64,
        "finger3_mid_bottom_right": 65,
        "finger3_mid_left": 66,
        "finger3_mid_right": 67,
        "finger3_mid_top_left": 68,
        "finger3_mid_top_right": 69,
        "finger4_mid_bottom_left": 70,
        "finger4_mid_bottom_right": 71,
        "finger4_mid_left": 72,
        "finger4_mid_right": 73,
        "finger4_mid_top_left": 74,
        "finger4_mid_top_right": 75,
        "finger5_mid_bottom_left": 76,
        "finger5_mid_bottom_right": 77,
        "finger5_mid_left": 78,
        "finger5_mid_right": 79,
        "finger5_mid_top_left": 80,
        "finger5_mid_top_right": 81,
        "nails_left": 82,
        "nails_right": 83,
        "leg_lower_left": 84,
        "leg_lower_right": 85,
        "leg_upper_left": 86,
        "leg_upper_right": 87,
        "foot_left": 88,
        "foot_right": 89,
        "pupil_left": 90,
        "pupil_right": 91,
        "eyelashes_left": 92,
        "eyelashes_right": 93,
        "eyelid_left": 92,
        "eyelid_right": 93,
        "eyebrow_left": 94,
        "eyebrow_right": 95,
        "glasses_frame": 96,
        "glasses_lens_left": 97,
        "glasses_lens_right": 98,
        "undereye_left": 99,
        "undereye_right": 100,
        "sclera_left": 101,
        "sclera_right": 102,
        "cornea_left": 103,
        "cornea_right": 104,
    }
    """
    Default body segmentation mapping.
    """

    FACE_SEGMENTATION_CLASSES = [
        "brow",
        "cheek_left",
        "cheek_right",
        "chin",
        "eye_left",
        "eye_right",
        "eyelid_left",
        "eyelid_right",
        "eyes",
        "jaw",
        "jowl",
        "lip_lower",
        "lip_upper",
        "mouth",
        "mouthbag",
        "nose",
        "nose_outer",
        "nostrils",
        "smile_line",
        "teeth",
        "undereye",
        "eyelashes_left",
        "eyelashes_right",
        "eyebrow_left",
        "eyebrow_right",
        "undereye_left",
        "undereye_right",
    ]
    "Segmentation classes included in the face bounding box."

    def __init__(
        self,
        root: Union[str, os.PathLike],
        modalities: Optional[List[Modality]] = None,
        body_segmentation_mapping: Optional[Dict[str, int]] = None,
        face_segmentation_classes: Optional[List[str]] = None,
        face_bbox_pad: int = 0,
        grouping: Grouping = Grouping.NONE,
        out_of_frame_landmark_strategy: OutOfFrameLandmarkStrategy = OutOfFrameLandmarkStrategy.IGNORE,
        transform: Optional[
            Callable[[Dict[Modality, Any]], Dict[Modality, Any]]
        ] = None,
    ) -> None:
        """
        Initializes FaceApiDataset from the data in :attr:`root` directory, loading listed :attr:`modalities`.
        All dataset files should be located in the root directory, which should either look like this::

            root
            ├── metadata.jsonl
                0.cam_default.f_1.exr
                0.cam_default.f_1.rgb.png
                0.cam_default.f_1.info.json
                0.cam_default.f_1.segments.png
                0.cam_default.f_1.alpha.tif
                0.cam_default.f_1.depth.tif
                0.cam_default.f_1.normals.tif
                1.cam_default.f_1.exr
                1.cam_default.f_1.rgb.png
                1.cam_default.f_1.info.json
                1.cam_default.f_1.segments.png
                1.cam_default.f_1.alpha.tif
                1.cam_default.f_1.depth.tif
                1.cam_default.f_1.normals.tif

        Or it should look like this::

            root
            ├── metadata.jsonl
                0.cam_default.f_1.exr
                0.cam_default.f_1.rgba.npz
                0.cam_default.f_1.image_metadata.json
                0.cam_default.f_1.body_segmentation.npz
                0.cam_default.f_1.depth.npz
                0.cam_default.f_1.surface_normals.npz
                1.cam_default.f_1.exr
                1.cam_default.f_1.rgba.npz
                1.cam_default.f_1.image_metadata.json
                1.cam_default.f_1.body_segmentation.npz
                1.cam_default.f_1.depth.npz
                1.cam_default.f_1.surface_normals.npz

        No extra files are allowed, but all files which are not needed to load modalities listed may be absent.

        For instance, if only `RGB` and `BODY_SEGMENTATION` modalities are needed, then only one of the following sets of files are enough::

            root
            ├── 0.cam_default.f_1.rgb.png
                0.cam_default.f_1.info.json
                0.cam_default.f_1.segments.png
                1.cam_default.f_1.rgb.png
                1.cam_default.f_1.info.json
                1.cam_default.f_1.segments.png

            OR

            root
            ├── 0.cam_default.f_1.rgba.npz
                0.cam_default.f_1.image_metadata.json
                0.cam_default.f_1.body_segmentation.npz
                1.cam_default.f_1.rgba.npz
                1.cam_default.f_1.image_metadata.json
                1.cam_default.f_1.body_segmentation.npz

        If any of the required files are absent for at least one image, or any redundant files are located in the directory, :class:`ValueError` is raised.

        To work with segment modalities, :attr:`body_segmentation_mapping` parameter is used.
        This shows how to map a segmentation class name to integer representation.
        For example, to work with `background` (0)/`face` (1)/`hair` (2)/`body` (3) segmentation it may look like this::

            body_segmentation_mapping = {
                'default': 0,
                'background': 0,
                'beard': 1,
                'body': 3,
                'brow': 1,
                'cheek_left': 1,
                'cheek_right': 1,
                'chin': 1,
                'clothing': 3,
                'ear_left': 1,
                'ear_right': 1,
                'eye_left': 1,
                'eye_right': 1,
                'eyelashes': 1,
                'eyelid': 1,
                'eyes': 1,
                'forehead': 1,
                'glasses': 0,
                'hair': 2,
                'head': 1,
                'headphones': 0,
                'headwear': 0,
                'jaw': 1,
                'jowl': 1,
                'lip_lower': 1,
                'lip_upper': 1,
                'mask': 0,
                'mouth': 1,
                'mouthbag': 1,
                'mustache': 1,
                'neck': 3,
                'nose': 1,
                'nose_outer': 1,
                'nostrils': 1,
                'shoulders': 3,
                'smile_line': 1,
                'teeth': 1,
                'temples': 1,
                'tongue': 1,
                'undereye': 1
            }

        In addition :attr:`transform` function may be provided. If it is not `None` it will be applied to modality dict after each :meth:`__get__` call.

        For example, to flip rgb image and its segmentation::

            def transform(item: Dict[Modality, Any]) -> Dict[Modality, Any]:
                ret = item.copy()
                ret[Modality.RGB] = flip(modalities[Modality.RGB])
                ret[Modality.BODY_SEGMENTATION] = flip(modalities[Modality.BODY_SEGMENTATION])
                return ret

        :param Union[str,bytes,os.PathLike] root: Dataset root. All image files (ex. `0.cam_default.f_1.rgb.png`) should be located directly in this directory.
        :param Optional[List[Modality]] modalities: List of modalities to load. If None all the modalities are loaded.
        :param Optional[Dict[str,int]] body_segmentation_mapping: Mapping from object names to segmentation id. If `None` :attr:`BODY_SEGMENTATION_MAPPING` mapping is used.
        :param Optional[List[str]] face_segmentation_classes: List of object names considered to incorporate a face. If `None` :attr:`FACE_SEGMENTATION_CLASSES` is used.
        :param int face_bbox_pad: Extra area in pixels to pad around height and width of face bounding box.
        :param Optional[Callable[[Dict[Modality,Any]],Dict[Modality,Any]]] transform: Additional transforms to apply to modalities.
        """
        if body_segmentation_mapping is None:
            body_segmentation_mapping = self.BODY_SEGMENTATION_MAPPING
        if modalities is None:
            modalities = list(Modality)
        self._body_segmentation_mapping = body_segmentation_mapping
        if face_segmentation_classes is None:
            face_segmentation_classes = self.FACE_SEGMENTATION_CLASSES
        self._face_segmentation_classes = face_segmentation_classes
        self._face_bbox_pad = face_bbox_pad
        self._modalities = sorted(modalities, key=lambda x: x.value)

        self._root = Path(root)
        if not (
            self._root.exists() and self._root.is_dir() and len(os.listdir(root)) > 0
        ):
            raise ValueError(
                f"{root} directory either doesn't exist or is not a directory or doesn't have files"
            )
        self._grouping = grouping

        metadata_records = []
        version_checked = False
        for file_path in self._root.glob("*"):
            if file_path.name == "metadata.jsonl" or file_path.suffix == ".exr":
                continue
            [
                scene_id,
                camera,
                frame,
                modality_name,
                modality_extension,
            ] = file_path.name.split(".")
            if not version_checked and modality_name == "info":
                with open(file_path, "r") as f:
                    info = json.load(f)
                dataset_version = info["version"]
                if parse_version(dataset_version) != parse_version("1.5"):
                    raise ValueError(
                        f"The version of this dataset is {dataset_version} which is not compatible with current face_api_dataset version. You could use an earlier version of face_api_dataset(<=1.0.4)."
                    )
                version_checked = True
            frame_num = int(frame.split(FRAME_NUM_PREFIX)[-1])
            cam_name = camera.split(CAM_NAME_PREFIX)[-1]
            extension = f"{modality_name}.{modality_extension}"
            record = {
                "SCENE_ID": int(scene_id),
                "CAMERA": camera,
                "CAMERA_NAME": cam_name,
                "FRAME": frame,
                "FRAME_NUM": int(frame_num),
                "EXTENSION": extension,
                "file_path": str(file_path),
            }

            if not scene_id.isdigit():
                raise ValueError(f"Unexpected file {file_path} in the dataset")

            metadata_records.append(record)

        self._item_loader = _ItemLoaderFactory.get_item_loader(
            self._root,
            self._modalities,
            metadata_records,
            out_of_frame_landmark_strategy,
            self._body_segmentation_mapping,
            self._face_segmentation_classes,
            self._face_bbox_pad,
        )
        self._metadata = pd.DataFrame.from_records(metadata_records)

        if grouping is Grouping.NONE:
            group_columns = ["SCENE_ID", "CAMERA_NAME", "FRAME_NUM"]
        elif grouping is Grouping.SCENE:
            group_columns = ["SCENE_ID"]
        elif grouping is Grouping.CAMERA:
            group_columns = ["CAMERA_NAME"]
        elif grouping is Grouping.SCENE_CAMERA:
            group_columns = ["SCENE_ID", "CAMERA_NAME"]
        else:
            raise ValueError(f"Invalid grouping parameter {grouping}")

        self._metadata.set_index(
            ["SCENE_ID", "CAMERA_NAME", "FRAME_NUM"], inplace=True, drop=False
        )
        self._metadata.sort_index(inplace=True)
        self._group_columns = group_columns
        self._group_meta = self._metadata.groupby(level=group_columns)
        self._group_index: List[tuple] = list(self._group_meta.indices.keys())

        self._out_of_frame_landmark_strategy = out_of_frame_landmark_strategy
        self._transform = transform

    @property
    def body_segmentation_mapping(self) -> Dict[str, int]:
        """
        Segment mapping for the dataset.

        :type: Dict[str, int]
        """
        return self._body_segmentation_mapping

    @property
    def modalities(self) -> List[Modality]:
        """
        List of the loaded modalities.

        :type: List[Modality]
        """
        return self._modalities

    def __len__(self) -> int:
        return len(self._group_meta)

    @overload
    def __getitem__(self, index: int) -> dict:
        pass

    @overload
    def __getitem__(self, index: slice) -> "FaceApiDataset":
        pass

    def __getitem__(self, i: Union[int, slice]) -> Union[dict, "FaceApiDataset"]:
        if isinstance(i, slice):
            ret = copy.copy(self)
            ret._group_index = ret._group_index[i]
            indices = np.concatenate(
                [self._group_meta.indices[idx] for idx in ret._group_index]
            )
            ret._metadata = self._metadata.iloc[indices]
            ret._group_meta = ret._metadata.groupby(level=self._group_columns)
            return ret
        else:
            if self._transform is None:
                return self._get(i)
            else:
                return self._transform(self._get(i))

    def get_group_index(self) -> pd.DataFrame:
        if self._grouping is self._grouping.NONE:
            raise ValueError(
                "Group index is unavailable when groping is set to Grouping.NONE"
            )
        return pd.DataFrame(self._group_index, columns=self._group_columns)

    def _get(self, i: int) -> Union[Item, List[Item]]:
        if i > len(self):
            raise ValueError(f"Index {i} is out of bounds")
        group_idx = self._group_index[i]
        group_meta: pd.DataFrame = self._group_meta.get_group(group_idx)

        if self._grouping is Grouping.NONE:
            res = self._item_loader.get_item(group_idx, group_meta)
        elif self._grouping is Grouping.SCENE:
            scene_id = group_idx
            res = []
            for cam in group_meta.CAMERA_NAME.unique():
                cam_group_meta = group_meta[group_meta.CAMERA_NAME == cam]
                for frame in group_meta.FRAME_NUM.unique():
                    item_meta = cam_group_meta[cam_group_meta.FRAME_NUM == frame]
                    element_idx = (scene_id, cam, frame)
                    res.append(self._item_loader.get_item(element_idx, item_meta))
        elif self._grouping is Grouping.CAMERA:
            cam = group_idx
            res = []
            for scene_id in group_meta.SCENE_ID.unique():
                scene_group_meta = group_meta[group_meta.SCENE_ID == scene_id]
                for frame in scene_group_meta.FRAME_NUM.unique():
                    item_meta = scene_group_meta[scene_group_meta.FRAME_NUM == frame]
                    element_idx = (scene_id, cam, frame)
                    res.append(self._item_loader.get_item(element_idx, item_meta))
        elif self._grouping is Grouping.SCENE_CAMERA:
            scene_id, cam = group_idx
            res = []
            for frame in group_meta.FRAME_NUM.unique():
                item_meta = group_meta[group_meta.FRAME_NUM == frame]
                element_idx = (scene_id, cam, frame)
                res.append(self._item_loader.get_item(element_idx, item_meta))
        else:
            raise ValueError(f"Invalid grouping {self._grouping}.")
        return res

import json
import json
import os
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import List
from typing import Sequence, Optional, Any, Union, Dict, Callable, TYPE_CHECKING, Tuple

import numpy as np

from face_api_dataset.modality import Modality

Id = str
Landmark_2D = Tuple[float, float]
Landmark_3D = Tuple[float, float, float]
CAMERA = str
Render_Id = str
Frame_Mapping = "Dict[int, str]"
Item = Dict


class _Extension(str, Enum):
    INFO = "info.json"
    RGB = "rgb.png"
    NORMALS = "normals.tif"
    DEPTH = "depth.tif"
    ALPHA = "alpha.tif"
    SEGMENTS = "segments.png"
    CAM_NAME_PREFIX = "cam_"
    FRAME_NO_PREFIX = "f_"


def _modality_files(modality: Modality) -> List[_Extension]:
    return {
        Modality.RENDER_ID: [_Extension.INFO],
        Modality.RGB: [_Extension.RGB],
        Modality.NORMALS: [_Extension.NORMALS],
        Modality.DEPTH: [_Extension.DEPTH],
        Modality.ALPHA: [_Extension.ALPHA],
        Modality.SEGMENTS: [_Extension.INFO, _Extension.SEGMENTS],
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
        Modality.FRAME_NUM: [_Extension.INFO]
    }[modality]


def _check_import(modality: Modality) -> None:
    if {_Extension.NORMALS, _Extension.DEPTH, _Extension.ALPHA}.intersection(
            _modality_files(modality)
    ):
        if find_spec("tiffile") is None:
            raise ImportError(f"Module tiffile is needed to load {modality}")
        if find_spec("imagecodecs") is None:
            raise ImportError(f"Module imagecodecs is needed to load {modality}")
    if {_Extension.SEGMENTS, _Extension.RGB}.intersection(_modality_files(modality)):
        if find_spec("cv2") is None:
            raise ImportError(f"Module cv2 is needed to load {modality}")


class OutOfFrameLandmarkStrategy(Enum):
    IGNORE = "ignore"
    CLIP = "clip"

    @staticmethod
    def clip_landmarks_(landmarks: Dict[Id, Landmark_2D], height: int, width: int) -> Dict[Id, Landmark_2D]:
        clipped_landmarks: Dict[Id, Landmark_2D] = {}
        for name, (x, y) in landmarks.items():
            xx = np.clip(x, 0, width - 1)
            yy = np.clip(y, 0, height - 1)
            clipped_landmarks[name] = (float(xx), float(yy))
        return clipped_landmarks


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
    Each items is a plain `Item` dict.
    """
    SCENE = "SCENE"
    """
    Frames from the same scene are grouped and can be indexed by `camera_name`.

    The size of the dataset is #scenes , each element `List[Item]` 
    is a list of frames from all cameras from the same scene.
    """
    CAMERA = "CAMERA"
    """
    Frames from the same camera are grouped and can be indexed by `frame_id`.

    The size of the dataset is #cameras , each element `List[Item]` 
    is is a list of frames from all scenes with the same camera.
    """
    CAMERA_SCENE = "CAMERA_SCENE"
    """
    Frames are groped first by camera and then by scene.
    List of frames for a particular scene is indexed by `scene_id`.
    
    The size of the dataset is #cameras , each element is a `Dict[scene_id, List[Item]]`, with #scenes items,
    where `[List[Item]` is a list of consecutive frames for given camera and scene.
    """
    SCENE_CAMERA = "SCENE_CAMERA"
    """
    Frames are groped first by scene and then by camera.
    List of frames for a particular scene is indexed by `camera_name`.
    
    The size of the dataset is #scenes , each element is a `Dict[scene_id, List[Item]]` with #cameras items, 
    where `[List[Item]` is a list of consecutive frames for given scene and camera.
    """


class FaceApiDataset(Base):
    """Synthesis AI face dataset.

    This class provides access to all the modalities available in Synthesis AI generated datasets.
    """

    SEGMENTS = {
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
        "undereye_right": 100
    }
    """
    Default segmentation mapping.
    """

    FACE_SEGMENTS = ["brow", "cheek_left", "cheek_right", "chin",
                     "eye_left", "eye_right", "eyelid_left", "eyelid_right",
                     "eyes", "jaw", "jowl", "lip_lower", "lip_upper",
                     "mouth", "mouthbag", "nose", "nose_outer", "nostrils",
                     "smile_line", "teeth", "undereye", "eyelashes_left", "eyelashes_right"]
    "Segments included in the bounding box."

    N_LANDMARKS = 68

    def __init__(
            self,
            root: Union[str, os.PathLike],
            modalities: Optional[List[Modality]] = None,
            segments: Optional[Dict[str, int]] = None,
            face_segments: Optional[List[str]] = None,
            face_bbox_pad: int = 0,
            grouping: Grouping = Grouping.NONE,
            out_of_frame_landmark_strategy: OutOfFrameLandmarkStrategy = OutOfFrameLandmarkStrategy.IGNORE,
            transform: Optional[
                Callable[[Dict[Modality, Any]], Dict[Modality, Any]]
            ] = None,
    ) -> None:
        """
        Initializes FaceApiDataset from the data in :attr:`root` directory, loading listed :attr:`modalities`.
        All dataset files should be located in the root directory::

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

        No extra files are allowed, but all files which are not needed to load modalities listed may be absent.

        For instance, if only `RGB` and `SEGMENTS` modalities are needed the following files are enough::

            root
            ├── 0.cam_default.f_1.rgb.png
                0.cam_default.f_1.info.json
                0.cam_default.f_1.segments.png
                1.cam_default.f_1.rgb.png
                1.cam_default.f_1.info.json
                1.cam_default.f_1.segments.png

        If any of the required files are absent for at least one image, or any redundant files are located in the directory, :class:`ValueError` is raised.

        To work with segment modalities, :attr:`segments` parameter is used.
        This shows how to map segments name to integer representation.
        For example, to work with `background` (0)/`face` (1)/`hair` (2)/`body` (3) segmentation it may look like this::

            segments = { 'default': 0,
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

            def transform(modalities: List[Modality]) -> List[Modality]:
                ret = modalities.copy()
                ret[Modality.RGB] = flip(modalities[Modality.RGB])
                ret[Modality.SEGMENTS] = flip(modalities[Modality.SEGMENTS])
                return ret

        :param Union[str,bytes,os.PathLike] root: Dataset root. All image files (ex. `0.cam_default.f_1.rgb.png`) should be located directly in this directory.
        :param Optional[List[Modality]] modalities: List of modalities to load. If None all the modalities are loaded.
        :param Optional[Dict[str,int]] segments: Mapping from object names to segmentation id. If `None` :attr:`SEGMENTS` mapping is used.
        :param Optional[List[str]] face_segments: List of object names considered to incorporate a face. If `None` :attr:`FACE_SEGMENTS` mapping is used.
        :param int face_bbox_pad: Extra area in pixels to pad around height and width of face bounding box.
        :param grouping: Grouping dataset items grouping mode (default Grouping.None).
        :param Optional[Callable[[Dict[Modality,Any]],Dict[Modality,Any]]] transform: Additional transforms to apply to modalities.
        """
        if segments is None:
            segments = self.SEGMENTS
        if modalities is None:
            modalities = list(Modality)
        self._segments = segments
        if face_segments is None:
            face_segments = self.FACE_SEGMENTS
        self._face_segments = face_segments
        self._face_bbox_pad = face_bbox_pad
        self._modalities = sorted(modalities, key=lambda x: x.value)
        for modality in self._modalities:
            _check_import(modality)
        self._needs_info = False
        for modality in self._modalities:
            if _Extension.INFO in _modality_files(modality):
                self._needs_info = True
        self._root = Path(root)

        image_names = set()
        self._camera_to_render_id: Dict[CAMERA, Dict[Render_Id, Frame_Mapping]] = dict()
        self._render_id_to_camera: Dict[Render_Id, Dict[CAMERA, Frame_Mapping]] = dict()
        for file_path in self._root.glob("*"):
            if file_path.name == "metadata.jsonl":
                continue
            file_name = ".".join(file_path.name.split(".")[:3])
            render_id = file_name.split(".")[0]
            if not render_id.isdigit():
                raise ValueError(f"Unexpected file {file_path} in the dataset")
            # self._render_ids.add(render_id)
            if file_name in image_names:
                continue
            for modality in modalities:
                for extension in _modality_files(modality):
                    if not (self._root / f"{file_name}.{extension.value}").exists():
                        raise ValueError(
                            f"Can't find file '{file_name}.{extension.value}' "
                            f"required for {modality.name} modality"
                        )
            image_names.add(file_name)
            frame_num = int(file_name.split("f_")[-1])
            camera_name = file_name.split(".")[1]

            if grouping in (Grouping.CAMERA, Grouping.CAMERA_SCENE):
                render_id_to_frames = self._camera_to_render_id.setdefault(camera_name, dict())
                frame_to_filename = render_id_to_frames.setdefault(render_id, dict())
                frame_to_filename[frame_num] = file_name
            if grouping in (Grouping.SCENE, Grouping.SCENE_CAMERA):
                camera_to_frames = self._render_id_to_camera.setdefault(render_id, dict())
                frame_to_filename = camera_to_frames.setdefault(camera_name, dict())
                frame_to_filename[frame_num] = file_name

        self._image_names = sorted(list(image_names), key=lambda n: int(n.split(".")[0]))
        self._camera_names = list(self._camera_to_render_id.keys())
        self._render_ids = list(self._render_id_to_camera.keys())
        self._image_sizes: Dict[str, Tuple[int, int]] = {}
        self._out_of_frame_landmark_strategy = out_of_frame_landmark_strategy
        self._transform = transform
        self._grouping = grouping

    @property
    def segments(self) -> Dict[str, int]:
        """
        Segment mapping for the dataset.

        :type: Dict[str, int]
        """
        return self._segments

    @property
    def modalities(self) -> List[Modality]:
        """
        List of the loaded modalities.

        :type: List[Modality]
        """
        return self._modalities

    def __len__(self) -> int:
        len_by_group = {Grouping.NONE: len(self._image_names),
                        Grouping.CAMERA_SCENE: len(self._camera_names),
                        Grouping.SCENE_CAMERA: len(self._render_ids),
                        Grouping.CAMERA: len(self._camera_names),
                        Grouping.SCENE: len(self._render_ids)}
        return len_by_group[self._grouping]

    def __getitem__(self, i: int) -> Item:
        if self._transform is None:
            return self._get(i)
        else:
            return self._transform(self._get(i))

    def _get(self, i: int) -> Union[Item, List[Item], Dict[str, List[Item]]]:
        if i > len(self):
            raise ValueError(f"Index {i} is out of bounds")

        if self._grouping is Grouping.NONE:
            file_name = self._image_names[i]
            item = self._get_by_file_name(file_name)
            return item

        elif self._grouping in (Grouping.CAMERA, Grouping.CAMERA_SCENE):
            camera_name = self._camera_names[i]
            scene_frames: Dict[Render_Id, Frame_Mapping] = self._camera_to_render_id[camera_name]
            scene_frames_item: Dict[Render_Id, List[Item]] = self._flatten_frames(scene_frames)

            if self._grouping is Grouping.CAMERA:
                flat_frames = self._extract_flat_frames(scene_frames_item)
                return flat_frames
            return scene_frames_item

        elif self._grouping in (Grouping.SCENE, Grouping.SCENE_CAMERA):
            render_id = self._render_ids[i]
            camera_frames: Dict[CAMERA, Frame_Mapping] = self._render_id_to_camera[render_id]
            scene_camera_item: Dict[CAMERA, List[Item]] = self._flatten_frames(camera_frames)

            if self._grouping is Grouping.SCENE:
                flat_frames = self._extract_flat_frames(scene_camera_item)
                return flat_frames
            return scene_camera_item

        else:
            raise ValueError(f"Invalid grouping option {self._grouping}")

    def _flatten_frames(self, item: "Dict[str, Dict[int, str]]") -> Dict[str, List[Item]]:
        ret = {}
        for key, frame2fn in sorted(item.items()):
            ret[key] = []
            for frame_no, file_name in sorted(frame2fn.items()):
                ret[key].append(self._get_by_file_name(file_name))
        return ret

    @staticmethod
    def _extract_flat_frames(attr_frames: Dict[str, List[Item]]) -> List[Item]:
        flat_frames = []
        for scene, frames in attr_frames.items():
            flat_frames.extend(frames)
        return flat_frames

    def _get_by_file_name(self, file_name: str) -> Item:
        info = None
        if self._needs_info:
            info_file = self._root / f"{file_name}.{_Extension.INFO}"
            with info_file.open("r") as f:
                info = json.load(f)

        ret = {}
        for modality in self._modalities:
            ret[modality] = self._open_modality(modality, file_name, info)
        return ret

    def _open_modality(
            self, modality: Modality, file_name: str, info: Optional[dict]
    ) -> Any:
        render_id = int(file_name.split(".")[0])
        if modality == Modality.RENDER_ID:
            return render_id

        if modality == Modality.RGB:
            return self._read_rgb(file_name)

        if modality == Modality.SEGMENTS:
            segment_img, _ = self._read_segments(file_name, info)
            return segment_img

        if modality == Modality.NORMALS:
            import tiffile

            normals_file = (
                    self._root / f"{file_name}.{_modality_files(Modality.NORMALS)[0]}"
            )
            img = tiffile.imread(str(normals_file))
            if img is None:
                raise ValueError(f"Error reading {normals_file}")
            if file_name in self._image_sizes:
                if self._image_sizes[file_name] != img.shape[1::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={render_id}"
                    )
            else:
                self._image_sizes[file_name] = img.shape[1::-1]
            return img

        if modality == Modality.ALPHA:
            import tiffile

            alpha_file = self._root / f"{file_name}.{_modality_files(Modality.ALPHA)[0]}"
            img = tiffile.imread(str(alpha_file))
            if file_name in self._image_sizes:
                if self._image_sizes[file_name] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={render_id}"
                    )
            else:
                self._image_sizes[file_name] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {alpha_file}")
            return img

        if modality == Modality.DEPTH:
            import tiffile

            depth_file = self._root / f"{file_name}.{_modality_files(Modality.DEPTH)[0]}"
            img = tiffile.imread(str(depth_file))
            if file_name in self._image_sizes:
                if self._image_sizes[file_name] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={file_name}"
                    )
            else:
                self._image_sizes[file_name] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {depth_file}")
            return img

        if modality in (Modality.LANDMARKS_IBUG68, Modality.LANDMARKS_CONTOUR_IBUG68):
            return self._read_face_landmarks_2d(info, modality, file_name)

        if modality in (Modality.LANDMARKS_KINECT_V2, Modality.LANDMARKS_COCO,
                        Modality.LANDMARKS_MEDIAPIPE, Modality.LANDMARKS_MPEG4):
            landmark_meta = self._read_body_landmarks(info, modality, self._image_sizes[file_name])
            return landmark_meta

        if modality in (Modality.LANDMARKS_3D_IBUG68, Modality.LANDMARKS_3D_KINECT_V2,
                        Modality.LANDMARKS_3D_COCO, Modality.LANDMARKS_3D_MEDIAPIPE,
                        Modality.LANDMARKS_3D_MPEG4):
            landmark_meta = self._read_landmarks_3d(info, modality)
            return landmark_meta

        if modality == Modality.LANDMARKS_3D_IBUG68:
            landmarks = []
            for landmark in info["landmarks"]:
                landmarks.append(landmark["camera_space_pos"])
            return np.array(landmarks, dtype=np.float64)

        if modality == Modality.PUPILS:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["screen_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["screen_space_pos"],
            ]
            if file_name not in self._image_sizes:
                raise ValueError(
                    "Pupils can only be loaded with at least one image modality"
                )
            scale = self._image_sizes[file_name]
            pupils_np = np.array(pupils, dtype=np.float64) * scale
            return {"pupil_left": tuple(pupils_np[0]),
                    "pupil_right": tuple(pupils_np[1])}

        if modality == Modality.PUPILS_3D:
            pupils = {
                "pupil_left": tuple(info["pupil_coordinates"]["pupil_left"]["camera_space_pos"]),
                "pupil_right": tuple(info["pupil_coordinates"]["pupil_right"]["camera_space_pos"])}
            return pupils

        if modality == Modality.IDENTITY:
            return info["identity_metadata"]["id"]

        if modality == Modality.IDENTITY_METADATA:
            return info["identity_metadata"]

        if modality == Modality.HAIR:
            if "hair" in info["facial_attributes"]:
                return info["facial_attributes"]["hair"]
            else:
                return None

        if modality == Modality.FACIAL_HAIR:
            if "facial_hair" in info["facial_attributes"]:
                return info["facial_attributes"]["facial_hair"]
            else:
                return None

        if modality == Modality.EXPRESSION:
            return info["facial_attributes"]["expression"]

        if modality == Modality.GAZE:
            return info["facial_attributes"]["gaze"]

        if modality == Modality.FACE_BBOX:
            segment_img, segment_mapping_int = self._read_segments(file_name, info)

            segment_mapping = info["segments_mapping"]
            face_segments = set(self._face_segments) & set(segment_mapping.keys())
            face_seg_idxs = [segment_mapping_int[segment_mapping[s]] for s in face_segments]
            face_mask = np.isin(segment_img, face_seg_idxs).astype(np.uint16)

            def get_bbox(img: np.ndarray):
                yxs = np.where(img != 0)
                if len(yxs[0]) == 0:
                    return (-1, -1, -1, -1)
                bbox = np.min(yxs[1]), np.min(yxs[0]), np.max(yxs[1]), np.max(yxs[0])
                return bbox

            height, width, _ = self._read_rgb(file_name).shape

            def expand_bbox(bbox: Tuple[int, int, int, int], padding: int):
                x0, y0, x1, y1 = bbox
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(x1 + padding, width)
                y1 = min(y1 + padding, height)

                return (x0, y0, x1, y1)

            face_bbox = get_bbox(face_mask)
            expanded_face_bbox = expand_bbox(face_bbox, self._face_bbox_pad)

            return expanded_face_bbox

        if modality == Modality.CAM_TO_WORLD:
            return np.array(info["camera"]["transform_cam2world"]["mat_4x4"], dtype=np.float64)

        if modality == Modality.WORLD_TO_CAM:
            return np.array(info["camera"]["transform_world2cam"]["mat_4x4"], dtype=np.float64)

        if modality == Modality.HEAD_TO_CAM:
            return np.array(info["head_transform"]["transform_head2cam"]["mat_4x4"], dtype=np.float64)

        if modality == Modality.CAM_TO_HEAD:
            return np.linalg.inv(self._open_modality(Modality.HEAD_TO_CAM, file_name, info))

        if modality == Modality.HEAD_TO_WORLD:
            return np.array(info["head_transform"]["transform_head2world"]["mat_4x4"], dtype=np.float64)

        if modality == Modality.WORLD_TO_HEAD:
            return np.linalg.inv(self._open_modality(Modality.HEAD_TO_WORLD, file_name, info))

        if modality == Modality.CAM_INTRINSICS:
            return np.array(info["camera"]["intrinsics"], dtype=np.float64)

        if modality == Modality.CAMERA_NAME:
            return file_name.split(".")[1]

        if modality == Modality.FRAME_NUM:
            return int(file_name.split("f_")[-1])

        raise ValueError("Unknown modality")

    def _read_segments(self, file_name: str, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        segments_file = (
                self._root / f"{file_name}.{_modality_files(Modality.SEGMENTS)[1]}"
        )
        img = cv2.imread(str(segments_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Error reading {segments_file}")
        if file_name in self._image_sizes:
            if self._image_sizes[file_name] != img.shape[::-1]:
                raise ValueError(
                    f"Dimensions of different image modalities do not match for render_id={file_name}"
                )
        else:
            self._image_sizes[file_name] = img.shape[::-1]

        segment_mapping = info["segments_mapping"]
        segment_mapping_int = np.full(
            np.max(list(segment_mapping.values())) + 1,
            self.segments["default"],
            dtype=np.uint8,
        )
        for key, value in segment_mapping.items():
            if key in self.segments:
                segment_mapping_int[value] = self.segments[key]

        segment_img = segment_mapping_int[img]
        return segment_img, segment_mapping_int

    def _read_rgb(self, file_name: str) -> np.ndarray:
        import cv2

        rgb_file = self._root / f"{file_name}.{_Extension.RGB}"
        img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Error reading {rgb_file}")
        if file_name in self._image_sizes:
            if self._image_sizes[file_name] != img.shape[1::-1]:
                raise ValueError("Dimensions of different modalities do not match")
        else:
            self._image_sizes[file_name] = img.shape[1::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def _read_modality_meta(info: dict, mdt: Modality) -> dict:
        if mdt in (Modality.LANDMARKS_IBUG68, Modality.LANDMARKS_3D_IBUG68):
            meta_dict = info["landmarks"]
        elif mdt is Modality.LANDMARKS_CONTOUR_IBUG68:
            meta_dict = info["contour_landmarks"]
        elif mdt in (Modality.LANDMARKS_COCO, Modality.LANDMARKS_3D_COCO):
            meta_dict = info["body_landmarks"]["coco"]["whole_body"]
        elif mdt in (Modality.LANDMARKS_KINECT_V2, Modality.LANDMARKS_3D_KINECT_V2):
            meta_dict = info["body_landmarks"]["kinect_v2"]
        elif mdt in (Modality.LANDMARKS_MEDIAPIPE, Modality.LANDMARKS_3D_MEDIAPIPE):
            meta_dict = info["body_landmarks"]["mediapipe"]["body"]
        elif mdt in (Modality.LANDMARKS_MPEG4, Modality.LANDMARKS_3D_MPEG4):
            meta_dict = info["body_landmarks"]["mpeg4"]
        else:
            raise ValueError(f"Unrecognized modality for 3D landmarks {mdt}.")
        return meta_dict

    @classmethod
    def _read_body_landmarks(cls, info: dict, mdt: Modality, image_size: Tuple[int, int]
                             ) -> Dict[Id, Landmark_2D]:
        meta_dict = cls._read_modality_meta(info, mdt)
        result = {}
        for item in meta_dict:
            if mdt in (Modality.LANDMARKS_MPEG4, Modality.LANDMARKS_3D_MPEG4):
                # TODO https://synthesisai.atlassian.net/browse/ENG-357
                idd = item["name"]
            else:
                idd = str(item["id"])
            x, y = item["screen_space_pos"]
            w, h = image_size
            result[idd] = (x * w, y * h)

        return result

    def _read_face_landmarks_2d(self, info: dict, mdt: Modality, file_name: str) -> Dict[Id, Landmark_2D]:
        meta_dict = self._read_modality_meta(info, mdt)
        if file_name not in self._image_sizes:
            raise ValueError(
                "Landmarks can only be loaded with at least one image modality"
            )

        if mdt is Modality.LANDMARKS_IBUG68:
            n_landmarks = len(meta_dict)
            if n_landmarks != self.N_LANDMARKS:
                json_file = self._root / f"{file_name}.{_modality_files(Modality.LANDMARKS_IBUG68)[0]}"
                msg = (f"Error reading landmarks for item with name: {file_name} from file: {json_file}\n",
                       f"Got {n_landmarks} landmarks instead of {self.N_LANDMARKS}")
                raise ValueError(msg)

        w, h = self._image_sizes[file_name]
        landmarks: Dict[Id, Landmark_2D] = {}
        for landmark in meta_dict:
            x, y = landmark["screen_space_pos"]
            x, y = x * w, y * h
            landmarks[str(landmark["ptnum"])] = (x, y)

        if self._out_of_frame_landmark_strategy is OutOfFrameLandmarkStrategy.CLIP:
            landmarks = OutOfFrameLandmarkStrategy.clip_landmarks_(landmarks, h, w)
        return landmarks

    @classmethod
    def _read_landmarks_3d(cls, info: dict, mdt: Modality) -> Dict[Id, Landmark_3D]:
        meta_dict = cls._read_modality_meta(info, mdt)

        landmarks: Dict[Id, Landmark_3D] = {}
        for landmark in meta_dict:
            if mdt is Modality.LANDMARKS_3D_IBUG68:
                lmk_name = str(landmark["ptnum"])
            elif mdt is Modality.LANDMARKS_3D_MPEG4:  # TODO https://synthesisai.atlassian.net/browse/ENG-357
                lmk_name = landmark["name"]
            else:
                lmk_name = str(landmark["id"])
            landmarks[lmk_name] = tuple(landmark["camera_space_pos"])

        return landmarks

import importlib
import json
import os
from collections import Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Callable
import numpy as np


class Modality(Enum):
    """
    Different modalities of Synthesis AI dataset.
    All image modalities are in `[y][x][channel]` format, with axis going as follows::
        ┌-----> x
        |
        |
        v
        y
    """

    RENDER_ID = auto()
    """
    Render ID (image number). 
    
    **Type**: `int`.
    """
    RGB = auto()
    """
    RGB image modality. 
    
    **Type**: `ndarray[uint8]`. **Channels**: `3`.
    """
    NORMALS = auto()
    """
    Normals image. All values are in [-1,1] range.
    
    **Type**: `ndarray[float16]`. **Channels**: 3.
    """
    DEPTH = auto()
    """
    Depth Image. All values are positive floats. Background has depth=0.
    
    **Type**: `ndarray[float16]`. **Channels**: 1.
    """
    ALPHA = auto()
    """
    Alpha Image. 0 - means complete transparency, 255 - solid object.
    
    **Type**: `ndarray[uint8]`. **Channels**: 1.
    """
    SEGMENTS = auto()
    """
    Segmentation map. Semantic of different values is defined by segments mapping.
    **Type**: `ndarray[uint16]`. **Channels**: 1.
    """
    LANDMARKS = auto()
    """
    iBUG-68 landmarks. Each landmark is given by two coordinates (x,y) in pixels.
    
    **Type**: `ndarray[float64]`. **Dimensions**: (68, 2).
    """
    LANDMARKS_3D = auto()
    """
    iBUG-68 landmarks in 3D. Each landmark is given by three coordinates (x,y,z) in camera space.

    **Type**: `ndarray[float64]`. **Dimensions**: (68, 3).
    """
    PUPILS = auto()
    """
    Coordinates of pupils. Each pupil is given by two coordinates (x,y) in pixels.
    
    **Type**: `ndarray[float64]`. **Dimensions**: (2, 2).
    """
    PUPILS_3D = auto()
    """
    Coordinates of pupils in 3D. Each pupil is given by three coordinates (x,y,z) in camera space.
 
    **Type**: `ndarray[float64]`. **Dimensions**: (2, 3).
    """
    IDENTITY = auto()
    """
    Unique ID of the person on the image.
    
    **Type**: `int`.
    """
    IDENTITY_METADATA = auto()
    """
    Additional metadata about the person on the image.
    
    **Format**::
    
        {'gender': 'female'|'male',
         'age': int,
         'weight_kg': int,
         'height_cm': int,
         'id': int,
         'ethnicity': 'arab'|'asian'|'black'|'hisp'|'white'}
    """
    HAIR = auto()
    """
    Hair metadata. If no hair are present `None` is returned.
    
    **Format**::
    
        {'relative_length': float64,
         'relative_density': float64,
         'style': str,
         'color_seed': float64,
         'color': str}
    """
    FACIAL_HAIR = auto()
    """
    Facial hair metadata. If no facial hair are present `None` is returned.

    **Format**::
    
        {'relative_length': float64,
         'relative_density': float64,
         'style': str,
         'color_seed': float64,
         'color': str}
    """
    EXPRESSION = auto()
    """
    Expression and its intensity.
    
    **Format**::
    
        {'intensity': float64, 
        'name': str}
    """
    GAZE = auto()
    """
    Gaze angles in image space.
    
    **Type**: `ndarray[float64]`. **Dimensions**: 2.
    """


class _Extension(str, Enum):
    INFO = "info.json"
    RGB = "rgb.png"
    NORMALS = "normals.tif"
    DEPTH = "depth.tif"
    ALPHA = "alpha.tif"
    SEGMENTS = "segments.png"


def _modality_files(modality: Modality) -> List[_Extension]:
    return {
        Modality.RENDER_ID: [_Extension.INFO],
        Modality.RGB: [_Extension.RGB],
        Modality.NORMALS: [_Extension.NORMALS],
        Modality.DEPTH: [_Extension.DEPTH],
        Modality.ALPHA: [_Extension.ALPHA],
        Modality.SEGMENTS: [_Extension.INFO, _Extension.SEGMENTS],
        Modality.LANDMARKS: [_Extension.INFO],
        Modality.LANDMARKS_3D: [_Extension.INFO],
        Modality.PUPILS: [_Extension.INFO],
        Modality.PUPILS_3D: [_Extension.INFO],
        Modality.IDENTITY: [_Extension.INFO],
        Modality.IDENTITY_METADATA: [_Extension.INFO],
        Modality.HAIR: [_Extension.INFO],
        Modality.FACIAL_HAIR: [_Extension.INFO],
        Modality.EXPRESSION: [_Extension.INFO],
        Modality.GAZE: [_Extension.INFO],
    }[modality]


def _check_import(modality: Modality) -> None:
    if {_Extension.NORMALS, _Extension.DEPTH, _Extension.ALPHA}.intersection(
        _modality_files(modality)
    ):
        if importlib.util.find_spec("tiffile") is None:
            raise ImportError(f"Module tiffile is needed to load {modality}")
        if importlib.util.find_spec("imagecodecs") is None:
            raise ImportError(f"Module imagecodecs is needed to load {modality}")
    if {_Extension.SEGMENTS, _Extension.RGB}.intersection(_modality_files(modality)):
        if importlib.util.find_spec("cv2") is None:
            raise ImportError(f"Module cv2 is needed to load {modality}")


class FaceApiDataset(Sequence):
    """Synthesis AI face dataset.

    This class provides access to all the modalities available in Synthesis AI generated datasets.

    :ivar List[Modality] modalities: list of available modalities.
    :ivar Dict[str,int] segments: mapping from segment names to integral ids, useful for segmentation.
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
    }
    """
    Default segmentation mapping.
    """

    def __init__(
        self,
        root: Union[str, bytes, os.PathLike],
        modalities: Optional[List[Modality]] = None,
        segments: Optional[Dict[str, int]] = None,
        transform: Optional[
            Callable[[Dict[Modality, Any]], Dict[Modality, Any]]
        ] = None,
    ) -> None:
        """
        Initializes FaceApiDataset from the data in :attr:`root` directory, loading listed :attr:`modalities`.
        All dataset files should be located in the root directory::

            root
            ├── metadata.jsonl
                0.exr
                0.rgb.png
                0.info.json
                0.segments.png
                0.alpha.tif
                0.depth.tif
                0.normals.tif
                1.exr
                1.rgb.png
                1.info.json
                1.segments.png
                1.alpha.tif
                1.depth.tif
                1.normals.tif

        No extra files are allowed, but all files which are not needed to load modalities listed may be abcent.

        For instance, if only `RGB` and `SEGMENTS` modalities are needed the following files are enough::

            root
            ├── 0.rgb.png
                0.info.json
                0.segments.png
                1.rgb.png
                1.info.json
                1.segments.png

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

        :param Union[str,bytes,os.PathLike] root: Dataset root. All image files (ex. `0.rgb.png`) should be located directly in this directory.
        :param Optional[List[Modality]] modalities: List of modalities to load. If None all the modalities are loaded.
        :param Optional[Dict[str,int]] segments: Mapping from object names to segmentation id. If `None` :attr:`SEGMENTS` mapping is used.
        :param Optional[Callable[[Dict[Modality,Any]],Dict[Modality,Any]]] transform: Additional transforms to apply to modalities.
        """
        if segments is None:
            segments = self.SEGMENTS
        if modalities is None:
            modalities = list(Modality)
        self.segments = segments
        self.modalities = sorted(modalities, key=lambda x: x.value)
        for modality in self.modalities:
            _check_import(modality)
        self._needs_info = False
        for modality in self.modalities:
            if _Extension.INFO in _modality_files(modality):
                self._needs_info = True
        self.root = Path(root)
        image_numbers = set()
        for file_path in self.root.glob("*"):
            if file_path.name == "metadata.jsonl":
                continue
            number = file_path.name.split(".")[0]
            if not number.isdigit():
                raise ValueError(f"Unexpected file {file_path} in the dataset")
            if number in image_numbers:
                continue
            for modality in modalities:
                for extension in _modality_files(modality):
                    if not (self.root / f"{number}.{extension.value}").exists():
                        raise ValueError(
                            f"Can't find file '{number}.{extension.value}' "
                            f"required for {modality.name} modality"
                        )
            image_numbers.add(number)
        self.image_numbers = sorted(list(image_numbers), key=int)
        self._image_sizes = {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_numbers)

    def __getitem__(self, i: int) -> dict:
        if self.transform is None:
            return self._get(i)
        else:
            return self.transform(self._get(i))

    def _get(self, i: int) -> dict:
        if i > len(self):
            raise ValueError(f"Index {i} is out of bounds")
        number = self.image_numbers[i]
        info = None
        if self._needs_info:
            info_file = self.root / f"{number}.{_Extension.INFO}"
            with info_file.open("r") as f:
                info = json.load(f)

        ret = {}
        for modality in self.modalities:
            ret[modality] = self._open_modality(modality, number, info)
        return ret

    def _open_modality(
        self, modality: Modality, number: str, info: Optional[dict]
    ) -> Any:
        if modality == Modality.RENDER_ID:
            return int(number)

        if modality == Modality.RGB:
            import cv2

            rgb_file = self.root / f"{number}.{_modality_files(Modality.RGB)[0]}"
            img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Error reading {rgb_file}")
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[1::-1]:
                    raise ValueError("Dimensions of different modalities do not match")
            else:
                self._image_sizes[number] = img.shape[1::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        if modality == Modality.SEGMENTS:
            import cv2

            segments_file = (
                self.root / f"{number}.{_modality_files(Modality.SEGMENTS)[1]}"
            )
            img = cv2.imread(str(segments_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Error reading {segments_file}")
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={number}"
                    )
            else:
                self._image_sizes[number] = img.shape[::-1]

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
            return segment_img

        if modality == Modality.NORMALS:
            import tiffile

            normals_file = (
                self.root / f"{number}.{_modality_files(Modality.NORMALS)[0]}"
            )
            img = tiffile.imread(str(normals_file))
            if img is None:
                raise ValueError(f"Error reading {normals_file}")
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[1::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={number}"
                    )
            else:
                self._image_sizes[number] = img.shape[1::-1]
            return img

        if modality == Modality.ALPHA:
            import tiffile

            alpha_file = self.root / f"{number}.{_modality_files(Modality.ALPHA)[0]}"
            img = tiffile.imread(str(alpha_file))
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={number}"
                    )
            else:
                self._image_sizes[number] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {alpha_file}")
            return img

        if modality == Modality.DEPTH:
            import tiffile

            depth_file = self.root / f"{number}.{_modality_files(Modality.DEPTH)[0]}"
            img = tiffile.imread(str(depth_file))
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(
                        f"Dimensions of different image modalities do not match for render_id={number}"
                    )
            else:
                self._image_sizes[number] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {depth_file}")
            return img

        if modality == Modality.LANDMARKS:
            landmarks = []
            if not number in self._image_sizes:
                raise ValueError(
                    "Landmarks can only be loaded with at least one image modality"
                )
            scale = self._image_sizes[number]
            for landmark in info["landmarks"]:
                landmarks.append(landmark["screen_space_pos"])
            return np.array(landmarks, dtype=np.float64) * scale

        if modality == Modality.LANDMARKS_3D:
            landmarks = []
            for landmark in info["landmarks"]:
                landmarks.append(landmark["camera_space_pos"])
            return np.array(landmarks, dtype=np.float64)

        if modality == Modality.PUPILS:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["screen_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["screen_space_pos"],
            ]
            if not number in self._image_sizes:
                raise ValueError(
                    "Pupils can only be loaded with at least one image modality"
                )
            scale = self._image_sizes[number]
            return np.array(pupils, dtype=np.float64) * scale

        if modality == Modality.PUPILS_3D:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["camera_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["camera_space_pos"],
            ]
            return np.array(pupils, dtype=np.float64)

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
            gaze = [
                info["facial_attributes"]["gaze"]["horizontal_angle"],
                info["facial_attributes"]["gaze"]["vertical_angle"],
            ]
            return np.array(gaze, dtype=np.float64)

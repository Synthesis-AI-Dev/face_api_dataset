import importlib
import json
import os
from collections import Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Any, Union
import numpy as np

SEGMENTS = {'default': 0,
            'background': 0,
            'beard': 1,
            'body': 2,
            'brow': 3,
            'cheek_left': 4,
            'cheek_right': 5,
            'chin': 6,
            'clothing': 7,
            'ear_left': 8,
            'ear_right': 9,
            'eye_left': 10,
            'eye_right': 11,
            'eyelashes': 12,
            'eyelid': 13,
            'eyes': 14,
            'forehead': 15,
            'glasses': 16,
            'hair': 17,
            'head': 18,
            'headphones': 19,
            'headwear': 20,
            'jaw': 21,
            'jowl': 22,
            'lip_lower': 23,
            'lip_upper': 24,
            'mask': 25,
            'mouth': 26,
            'mouthbag': 27,
            'mustache': 28,
            'neck': 29,
            'nose': 30,
            'nose_outer': 31,
            'nostrils': 32,
            'shoulders': 33,
            'smile_line': 34,
            'teeth': 35,
            'temples': 36,
            'tongue': 37,
            'undereye': 38}


class Modality(Enum):
    RENDER_ID = auto()
    RGB = auto()
    NORMALS = auto()
    DEPTH = auto()
    ALPHA = auto()
    SEGMENTS = auto()
    LANDMARKS = auto()
    LANDMARKS_3D = auto()
    PUPILS = auto()
    PUPILS_3D = auto()
    IDENTITY = auto()
    IDENTITY_METADATA = auto()
    HAIR = auto()
    FACIAL_HAIR = auto()
    EXPRESSION = auto()
    GAZE = auto()

class Extension(str, Enum):
    INFO = "info.json"
    RGB = "rgb.png"
    NORMALS = "normals.tif"
    DEPTH = "depth.tif"
    ALPHA = "alpha.tif"
    SEGMENTS = "segments.png"


def modality_files(modality: Modality) -> List[Extension]:
    return {
        Modality.RENDER_ID: [Extension.INFO],
        Modality.RGB: [Extension.RGB],
        Modality.NORMALS: [Extension.NORMALS],
        Modality.DEPTH: [Extension.DEPTH],
        Modality.ALPHA: [Extension.ALPHA],
        Modality.SEGMENTS: [Extension.INFO, Extension.SEGMENTS],
        Modality.LANDMARKS: [Extension.INFO],
        Modality.LANDMARKS_3D: [Extension.INFO],
        Modality.PUPILS: [Extension.INFO],
        Modality.PUPILS_3D: [Extension.INFO],
        Modality.IDENTITY: [Extension.INFO],
        Modality.IDENTITY_METADATA: [Extension.INFO],
        Modality.HAIR: [Extension.INFO],
        Modality.FACIAL_HAIR: [Extension.INFO],
        Modality.EXPRESSION: [Extension.INFO],
        Modality.GAZE: [Extension.INFO]
    }[modality]


def check_import(modality: Modality) -> None:
    if {Extension.NORMALS, Extension.DEPTH, Extension.ALPHA}.intersection(modality_files(modality)):
        if importlib.util.find_spec("tiffile") is None:
            raise ImportError(f"Module tiffile is needed to load {modality}")
        if importlib.util.find_spec("imagecodecs") is None:
            raise ImportError(f"Module imagecodecs is needed to load {modality}")
    if {Extension.SEGMENTS, Extension.RGB}.intersection(modality_files(modality)):
        if importlib.util.find_spec("cv2") is None:
            raise ImportError(f"Module cv2 is needed to load {modality}")


class FaceApiDataset(Sequence):
    def __init__(self, root: Union[str, bytes, os.PathLike],
                 modalities: Optional[List[Modality]] = None, segments=None,
                 transform=None) -> None:
        if segments is None:
            segments = SEGMENTS
        if modalities is None:
            modalities = list(Modality)
        self.segments = segments
        self.modalities = sorted(modalities, key=lambda x: x.value)
        for modality in self.modalities:
            check_import(modality)
        self._needs_info = False
        for modality in self.modalities:
            if Extension.INFO in modality_files(modality):
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
                for extension in modality_files(modality):
                    if not (self.root / f"{number}.{extension.value}").exists():
                        raise ValueError(f"Can't find file '{number}.{extension.value}' "
                                         f"required for {modality.name} modality")
            image_numbers.add(number)
        self.image_numbers = sorted(list(image_numbers), key=int)
        self._image_sizes = {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_numbers)

    def __getitem__(self, i: int) -> dict:
        if self.transform is None:
            return self.get(i)
        else:
            return self.transform(self.get(i))

    def get(self, i: int) -> dict:
        if i > len(self):
            raise ValueError(f"Index {i} is out of bounds")
        number = self.image_numbers[i]
        info = None
        if self._needs_info:
            info_file = self.root / f"{number}.{Extension.INFO}"
            with info_file.open("r") as f:
                info = json.load(f)

        ret = {}
        for modality in self.modalities:
            ret[modality] = self._open_modality(modality, number, info)
        return ret

    def _open_modality(self, modality: Modality, number: str, info: Optional[dict]) -> Any:
        if modality == Modality.RENDER_ID:
            return int(number)

        if modality == Modality.RGB:
            import cv2
            rgb_file = self.root / f"{number}.{modality_files(Modality.RGB)[0]}"
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
            segments_file = self.root / f"{number}.{modality_files(Modality.SEGMENTS)[1]}"
            img = cv2.imread(str(segments_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Error reading {segments_file}")
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(f"Dimensions of different image modalities do not match for render_id={number}")
            else:
                self._image_sizes[number] = img.shape[::-1]

            segment_mapping = info["segments_mapping"]
            segment_mapping_int = np.full(np.max(list(segment_mapping.values())) + 1, self.segments["default"],
                                          dtype=np.uint8)
            for key, value in segment_mapping.items():
                if key in self.segments:
                    segment_mapping_int[value] = self.segments[key]

            segment_img = segment_mapping_int[img]
            return segment_img

        if modality == Modality.NORMALS:
            import tiffile
            normals_file = self.root / f"{number}.{modality_files(Modality.NORMALS)[0]}"
            img = tiffile.imread(str(normals_file))
            if img is None:
                raise ValueError(f"Error reading {normals_file}")
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[1::-1]:
                    raise ValueError(f"Dimensions of different image modalities do not match for render_id={number}")
            else:
                self._image_sizes[number] = img.shape[1::-1]
            return img

        if modality == Modality.ALPHA:
            import tiffile
            alpha_file = self.root / f"{number}.{modality_files(Modality.ALPHA)[0]}"
            img = tiffile.imread(str(alpha_file))
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(f"Dimensions of different image modalities do not match for render_id={number}")
            else:
                self._image_sizes[number] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {alpha_file}")
            return img

        if modality == Modality.DEPTH:
            import tiffile
            depth_file = self.root / f"{number}.{modality_files(Modality.DEPTH)[0]}"
            img = tiffile.imread(str(depth_file))
            if number in self._image_sizes:
                if self._image_sizes[number] != img.shape[::-1]:
                    raise ValueError(f"Dimensions of different image modalities do not match for render_id={number}")
            else:
                self._image_sizes[number] = img.shape[::-1]
            if img is None:
                raise ValueError(f"Error reading {depth_file}")
            return img

        if modality == Modality.LANDMARKS:
            landmarks = []
            if not number in self._image_sizes:
                raise ValueError("Landmarks can only be loaded with at least one image modality")
            scale = self._image_sizes[number]
            for landmark in info["landmarks"]:
                landmarks.append(landmark["screen_space_pos"])
            return np.array(landmarks, dtype=np.float16) * scale

        if modality == Modality.LANDMARKS_3D:
            landmarks = []
            for landmark in info["landmarks"]:
                landmarks.append(landmark["camera_space_pos"])
            return np.array(landmarks, dtype=np.float16)

        if modality == Modality.PUPILS:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["screen_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["screen_space_pos"]
            ]
            if not number in self._image_sizes:
                raise ValueError("Pupils can only be loaded with at least one image modality")
            scale = self._image_sizes[number]
            return np.array(pupils, dtype=np.float16) * scale

        if modality == Modality.PUPILS_3D:
            pupils = [
                info["pupil_coordinates"]["pupil_left"]["camera_space_pos"],
                info["pupil_coordinates"]["pupil_right"]["camera_space_pos"]
            ]
            return np.array(pupils, dtype=np.float16)

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
                info["facial_attributes"]["gaze"]["vertical_angle"]
            ]
            return np.array(gaze, dtype=np.float16)

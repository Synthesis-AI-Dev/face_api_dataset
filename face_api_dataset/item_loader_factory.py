from pathlib import Path
from typing import Dict, List, Union
from face_api_dataset.data_types import OutOfFrameLandmarkStrategy
from face_api_dataset.item_loader import _ItemLoader
from face_api_dataset.item_loader_v1 import _ItemLoaderV1
from face_api_dataset.item_loader_v2 import _ItemLoaderV2
from face_api_dataset.modality import Modality


class _ItemLoaderFactory:
    @classmethod
    def _contains_v1_info(cls, root: Path) -> bool:
        for f in root.glob("*"):
            if str(f).endswith("info.json"):
                return True
        return False

    @classmethod
    def _contains_v2_info(cls, root: Path) -> bool:
        for f in root.glob("*"):
            if str(f).endswith("image_metadata.json"):
                return True
        return False

    @classmethod
    def get_item_loader(
        cls,
        root: Path,
        modalities: List[Modality],
        metadata_records: List[Dict[str, Union[int, str]]],
        out_of_frame_landmark_strategy: OutOfFrameLandmarkStrategy,
        body_segmentation_mapping: Dict[str, int],
        face_segmentation_classes: List[str],
        face_bbox_pad: int,
    ) -> _ItemLoader:
        contains_v1_info = cls._contains_v1_info(root)
        contains_v2_info = cls._contains_v2_info(root)

        if not contains_v1_info and not contains_v2_info:
            raise Exception(
                f"Neither *.info.json nor *.image_metadata.json files were found in {root}"
            )

        if contains_v1_info and contains_v2_info:
            raise Exception(
                f"Both *.info.json and *.image_metadata.json files were found in {root}"
            )

        if contains_v1_info:
            return _ItemLoaderV1(
                root,
                modalities,
                metadata_records,
                out_of_frame_landmark_strategy,
                body_segmentation_mapping,
                face_segmentation_classes,
                face_bbox_pad,
            )

        if contains_v2_info:
            return _ItemLoaderV2(
                root,
                modalities,
                metadata_records,
                out_of_frame_landmark_strategy,
                body_segmentation_mapping,
                face_segmentation_classes,
                face_bbox_pad,
            )

        raise Exception("Something went wrong")

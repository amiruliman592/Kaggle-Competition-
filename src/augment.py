"""
augment.py — Training augmentations (run_pipeline.py --aug).

  flip      flips + brightness/contrast at a fixed size (recipe since Phase 2)
  rotscale  flip + 90/180/270 degree rotations + random image size +-20% (Phase 6 S3)
  copypaste flip + filaments pasted from another training image (CopyPasteMapper, copy_paste.py; Phase 6 S4)

The image is resized whole, never cropped, so no filament is cut in two.
"""

import copy
from typing import Dict, List

import numpy as np
import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from config import CFG
from copy_paste import paste_filaments


def train_augmentations(size: int) -> List[T.Augmentation]:
    """Augmentations for images trained at `size` px (cfg.INPUT.MIN_SIZE_TRAIN)."""
    if CFG.train.AUG == "rotscale":
        low, high = round(size * (1 - CFG.train.SCALE_JITTER)), round(size * (1 + CFG.train.SCALE_JITTER))
        resize = T.ResizeShortestEdge(short_edge_length=(low, high), max_size=high, sample_style="range")
        rotate = [T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice")]
    else:
        # Same scale as inference (cfg.INPUT.*_TEST) to avoid a train/test mismatch
        resize = T.ResizeShortestEdge(short_edge_length=(size,), max_size=size, sample_style="choice")
        rotate = []
    return [
        resize,
        T.RandomFlip(prob=0.5, horizontal=True),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        *rotate,
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
    ]


class CopyPasteMapper(DatasetMapper):
    """DatasetMapper that, with prob CFG.train.COPY_PASTE_PROB, first pastes filaments of another image (S4)."""

    def __init__(self, cfg, augmentations: List[T.Augmentation], donors: List[Dict]):
        super().__init__(cfg, is_train=True, augmentations=augmentations)
        self.donors = [d for d in donors if d["annotations"]]

    def __call__(self, dataset_dict: Dict) -> Dict:
        d = copy.deepcopy(dataset_dict)
        image = utils.read_image(d["file_name"], format=self.image_format)
        if np.random.rand() < CFG.train.COPY_PASTE_PROB:
            donor = self.donors[np.random.randint(len(self.donors))]
            if donor["file_name"] != d["file_name"]:
                image, d["annotations"] = paste_filaments(
                    image, d["annotations"], utils.read_image(donor["file_name"], format=self.image_format),
                    donor["annotations"], n_max=CFG.train.COPY_PASTE_MAX)
        # Same steps as DatasetMapper.__call__ (no semantic masks or proposals in this project)
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        d["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        self._transform_annotations(d, transforms, image.shape[:2])
        return d

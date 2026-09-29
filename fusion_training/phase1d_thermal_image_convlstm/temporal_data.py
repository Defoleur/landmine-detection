"""Processed-image sequence Dataset."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD
from .sequence import CONDITIONS


class ImageSequenceDataset(Dataset):
    def __init__(self, sequences, tile_size=256, augment=False):
        self.sequences, self.tile_size, self.augment = (
            sequences,
            int(tile_size),
            augment,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        by_condition = {frame["condition"]: frame for frame in sequence["frames"]}
        images, present = [], []
        for condition in CONDITIONS:
            frame = by_condition.get(condition)
            if frame is None:
                image = np.zeros((3, self.tile_size, self.tile_size), dtype=np.float32)
                present.append(False)
            else:
                with Image.open(frame["image_path"]) as source:
                    image = (
                        np.asarray(
                            source.convert("RGB").resize(
                                (self.tile_size, self.tile_size),
                                Image.Resampling.BILINEAR,
                            ),
                            dtype=np.float32,
                        ).transpose(2, 0, 1)
                        / 255
                    )
                present.append(True)
            images.append(image)
        target = np.asarray(
            Image.open(sequence["target_path"])
            .convert("L")
            .resize((self.tile_size, self.tile_size), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
        target = (target > 0).astype(np.float32)[None]
        images, target = np.stack(images), target
        if self.augment and torch.rand(()) < 0.5:
            images, target = images[:, :, :, ::-1].copy(), target[:, :, ::-1].copy()
        mean = np.asarray(IMAGE_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(IMAGE_STD, dtype=np.float32)[None, :, None, None]
        return {
            "image": torch.from_numpy(((images - mean) / std).copy()),
            "target": torch.from_numpy(target),
            "target_validity": torch.ones_like(torch.from_numpy(target)),
            "condition_present": torch.tensor(present, dtype=torch.bool),
            "metadata": {
                "sequence_id": sequence["sequence_id"],
                "split": sequence["split"],
            },
        }

import json
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Union

from collections import namedtuple

import torch

import torchvision
from torchvision import tv_tensors

import imageio.v3 as imageio


class OpenTextileSegmentation(torchvision.datasets.VisionDataset):
    OpenTextileClass = namedtuple(
        "OpenTextileClass", ["name", "id", "color", "ignore_in_eval"]
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        input_mode: Union[str, list[str]] = ["rgb", "hyper"],
        target_mode: str = "labels",
        target_type: str = "semantic",
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        assert target_type in ["semantic", ""]

        self.input_mode = input_mode
        self.target_mode = target_mode
        self.target_type = target_type

        if not isinstance(input_mode, list):
            self.input_mode = [input_mode]

        meta = json.load(open(Path(root, "metadata.json")))
        grouped = pd.read_csv(Path(root, "ground_truth_grouped.csv"), sep=";")

        label_to_sample = {
            x["label"]: x["sample_id"] for x in meta["sample_label_mapping"]
        }
        sample_to_group = dict(zip(grouped["Label"], grouped["material_group_compact"]))

        label_to_group = {
            label: sample_to_group.get(sample)
            for label, sample in label_to_sample.items()
        }
        groups = sorted(
            {group for group in label_to_group.values() if isinstance(group, str)}
        )
        group_to_id = {group: i + 1 for i, group in enumerate(groups)}

        palette = [
            (0, 0, 0),          # 0 - background
            (244, 67, 54),      # 1 - red
            (33, 150, 243),     # 2 - blue
            (255, 193, 7),      # 3 - yellow
            (76, 175, 80),      # 4 - green
            (156, 39, 176),     # 5 - purple
            (255, 87, 34),      # 6 - deep orange
            (63, 81, 181),      # 7 - indigo
            (0, 188, 212),      # 8 - cyan
            (139, 195, 74),     # 9 - light green
            (255, 152, 0),      # 10 - orange
            (121, 85, 72),      # 11 - brown
            (96, 125, 139),     # 12 - blue grey
            (233, 30, 99),      # 13 - pink
            (0, 150, 136),      # 14 - teal
            (205, 220, 57),     # 15 - lime
            (255, 105, 180),    # 16 - hot pink
            (0, 255, 127),      # 17 - spring green
        ]

        self.classes = [self.OpenTextileClass("background", 0, palette[0], True)]
        self.classes += [
            self.OpenTextileClass(
                group, group_to_id[group], palette[group_to_id[group]], False
            )
            for group in groups
        ]

        self.classes_names = [c.name for c in self.classes]
        self.palette = [c.color for c in self.classes]
        self.num_classes = len(self.classes_names)

        self.label_to_group_id = np.zeros(max(label_to_group) + 1, dtype=np.int64)
        for label, group in label_to_group.items():
            if group is not None:
                self.label_to_group_id[label] = group_to_id[group]

        self.input_dirs = [Path(root, mode, split) for mode in self.input_mode]
        self.target_dir = Path(root, self.target_mode, split)

        self.input_paths = [list(sorted(dir.iterdir())) for dir in self.input_dirs]
        self.target_paths = list(sorted(self.target_dir.glob("*.png")))
        self.pixel_counts = self._compute_pixel_counts(split)

        sample = self[0]
        if isinstance(sample[0], list):
            self.num_channels = [input.shape[0] for input in sample[0]]
        else:
            self.num_channels = sample[0].shape[0]

    def _compute_pixel_counts(self, split: str) -> list[int]:
        if not self.target_paths:
            raise FileNotFoundError(f"No PNG labels found in {self.target_dir}")

        counts = np.zeros(self.num_classes, dtype=np.int64)
        for path in self.target_paths:
            target = imageio.imread(path)
            mapped = self.label_to_group_id[target]
            counts += np.bincount(mapped.reshape(-1), minlength=self.num_classes)

        zero_class_ids = np.flatnonzero(counts == 0)
        if zero_class_ids.size > 0:
            zero_class_names = [self.classes_names[idx] for idx in zero_class_ids]
            raise ValueError(
                f"OpenTextileSegmentation split='{split}' has zero pixel counts "
                f"for classes {zero_class_names}."
            )

        return counts.tolist()
    def __getitem__(self, idx):
        inputs = []
        for i, m in enumerate(self.input_mode):
            path = self.input_paths[i][idx]

            if path.suffix == ".npy":
                input = np.load(path).astype(np.float32)
            elif path.suffix == ".png":
                input = imageio.imread(path)
            elif path.suffix == ".tiff":
                input = imageio.imread(path)
                input = input.transpose(1, 2, 0)
            else:
                raise ValueError

            if issubclass(input.dtype.type, np.integer):
                input = input.astype(np.float32) / np.iinfo(input.dtype).max

            input = tv_tensors.Image(input)
            input = input.permute(2, 0, 1)
            inputs.append(input)
        target = imageio.imread(self.target_paths[idx])
        target = self.label_to_group_id[target]
        target = torch.tensor(target.astype(np.int64))

        if self.target_type == "semantic" or self.target_type == "":
            target = tv_tensors.Mask(target)
        else:
            raise ValueError

        if self.transforms:
            target, *inputs = self.transforms(target, *inputs)

        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs, target

    def __len__(self):
        return len(self.input_paths[0])

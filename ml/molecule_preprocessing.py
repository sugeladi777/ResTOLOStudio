from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class MoleculePreprocessingSpec:
    target_size: tuple[int, int] = (224, 224)
    context_scale: float = 1.0
    padding_mode: str = "reflect"
    normalization: str = "minus_one_to_one"

    def to_dict(self) -> dict:
        return {
            "target_size": list(self.target_size),
            "context_scale": self.context_scale,
            "padding_mode": self.padding_mode,
            "normalization": self.normalization,
        }


DEFAULT_PREPROCESSING = MoleculePreprocessingSpec()


def crop_normalized_box(
    image: Image.Image,
    x: float,
    y: float,
    width: float,
    height: float,
    spec: MoleculePreprocessingSpec = DEFAULT_PREPROCESSING,
) -> Image.Image:
    image = image.convert("RGB")
    pixel_width, pixel_height = image.size
    return crop_pixel_box(
        image,
        center_x=x * pixel_width,
        center_y=y * pixel_height,
        width=width * pixel_width,
        height=height * pixel_height,
        spec=spec,
    )


def crop_xyxy_array(
    image: np.ndarray,
    xyxy,
    spec: MoleculePreprocessingSpec = DEFAULT_PREPROCESSING,
) -> Image.Image:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    return crop_pixel_box(
        pil_image,
        center_x=(x1 + x2) / 2,
        center_y=(y1 + y2) / 2,
        width=max(1.0, x2 - x1),
        height=max(1.0, y2 - y1),
        spec=spec,
    )


def crop_pixel_box(
    image: Image.Image,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    spec: MoleculePreprocessingSpec = DEFAULT_PREPROCESSING,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    image_height, image_width = array.shape[:2]
    side = max(2, int(round(max(width, height) * spec.context_scale)))
    x1 = int(np.floor(center_x - side / 2))
    y1 = int(np.floor(center_y - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image_width)
    pad_bottom = max(0, y2 - image_height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        mode = spec.padding_mode
        if min(image_width, image_height) <= 1 and mode == "reflect":
            mode = "edge"
        array = np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mode,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = Image.fromarray(array[y1:y2, x1:x2])
    return crop.resize(spec.target_size, Image.Resampling.BICUBIC)


def image_to_tensor(
    image: Image.Image,
    spec: MoleculePreprocessingSpec = DEFAULT_PREPROCESSING,
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if spec.normalization == "minus_one_to_one":
        tensor = (tensor - 127.5) / 127.5
    return tensor.float()

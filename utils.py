import os
from contextlib import contextmanager

import numpy as np
from PIL import Image
from torch import Tensor


@contextmanager
def temporary_env_var(key: str, new_value: str | None):
    old_value = os.environ.get(key)
    if new_value:
        os.environ[key] = new_value
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]


def images_to_pillow(images: Tensor | list[Tensor]) -> list[Image.Image]:
    pillow_images = []
    for _bn, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        pillow_images.append(Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)))
    return pillow_images


def pillow_to_tensor(images: list[Image.Image]) -> Tensor:
    return Tensor(np.array([np.array(img) for img in images]).astype(np.float32) / 255.0)
from torchvision import transforms
from PIL import Image
import torch
import torchvision
from config import Config
import numpy as np

cfg = Config()

dtype = torch.cuda.FloatTensor if cfg.use_gpu else torch.FloatTensor


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)

    if max_size is not None:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image.type(dtype)
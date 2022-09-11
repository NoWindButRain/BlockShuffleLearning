from typing import Union, Tuple
from PIL import Image
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class LetterBox(nn.Module):
    """
    Make letter box transform to image and bounding box target.

    Args:
        size (int or tuple of int): the size of the transformed image.
    """

    def __init__(self, size: Union[int, Tuple[int]]):
        super().__init__()
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def forward(self, img: Image.Image):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            tuple: (image, )
        """
        old_width, old_height = img.size
        width, height = self.size

        ratio = min(width / old_width, height / old_height)
        new_width = int(old_width * ratio)
        new_height = int(old_height * ratio)
        img = transforms.functional.resize(img, (new_height, new_width))

        pad_x = (width - new_width) * 0.5
        pad_y = (height - new_height) * 0.5
        left, right = round(pad_x + 0.1), round(pad_x - 0.1)
        top, bottom = round(pad_y + 0.1), round(pad_y - 0.1)
        padding = (left, top, right, bottom)
        img = transforms.functional.pad(img, padding, 255 // 2)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"({self.size})"

transform = transforms.Compose([
    LetterBox(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_transform = transforms.Lambda(lambda y: 1-y)

dataset = ImageFolder('../dataset/ff++', transform=transform, target_transform=target_transform)
dataloader = DataLoader(dataset, batch_size=4)






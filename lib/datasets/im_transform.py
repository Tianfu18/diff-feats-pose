import numpy as np
import torch
import torchvision.transforms as transforms


class DiffusionPreProcess:
    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor):
        return img * 2. - 1.


def im_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        DiffusionPreProcess(),
    ])


def tensor2im(tensor: torch.Tensor):
    tensor = tensor / 2. + .5
    im = (tensor * 255.).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :, :, ::-1]
    return im
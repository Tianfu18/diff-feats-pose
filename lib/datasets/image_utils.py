import numpy as np
from PIL import ImageFile
import torchvision.transforms as transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_pad(img, dim):
    w, h = img.size
    img = transforms.functional.resize(img, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - img.size[0]) / 2))
    right = int(np.floor((dim - img.size[0]) / 2))
    top = int(np.ceil((dim - img.size[1]) / 2))
    bottom = int(np.floor((dim - img.size[1]) / 2))
    img = transforms.functional.pad(img, (left, top, right, bottom))
    return img


def process_mask_image(mask, mask_size):
    mask = mask.resize((mask_size, mask_size))
    mask = (np.asarray(mask) / 255. > 0) * 1
    mask = torch.from_numpy(mask).unsqueeze(0)
    return mask


def check_bbox_in_image(image, bbox):
    """
    Check bounding box is inside image
    """
    img_size = image.size
    check = np.asarray([bbox[0] >= 0, bbox[1] >= 0, bbox[2] <= img_size[0], bbox[3] <= img_size[1]])
    return (check == np.ones(4, dtype=np.bool_)).all()


def crop_image(image, bbox, keep_aspect_ratio):
    if not keep_aspect_ratio:
        return image.crop(bbox)
    else:
        new_bbox = np.array(bbox)
        current_bbox_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        final_size = max(current_bbox_size[0], current_bbox_size[1])
        # Add padding into y axis
        displacement_y = int((final_size - current_bbox_size[1]) / 2)
        new_bbox[1] -= displacement_y
        new_bbox[3] += displacement_y
        # Add padding into x axis
        displacement_x = int((final_size - current_bbox_size[0]) / 2)
        new_bbox[0] -= displacement_x
        new_bbox[2] += displacement_x
        if check_bbox_in_image(image, new_bbox):
            return image.crop(new_bbox)
        else:
            cropped_image = image.crop(bbox)
            return resize_pad(cropped_image, final_size)

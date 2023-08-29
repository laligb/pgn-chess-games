import numpy as np
from PIL import Image
import io


def img_bytes_to_num(img):
    """Transform image in bytes to numpy array.

    Args:
        img (bytes): Image in byte form.

    Returns:
        int: Numpy array.
    """
    result = np.array(Image.open(io.BytesIO(img)))
    return result


def mockup_predict():
    """
    Mockup function to be substituted ince we have the true model.

    Args:
        num_img (int): Numpy array
    """
    return {
        "white": ["d4", "c4", "Nc3", "Nf3", "Bd2", "e3", "Bd3", "O-O", "a3"],
        "black": ["Nf6", "e6", "d5", "Bb4", "O-O", "Nc6", "Bd7", "Qe7", "dxc4"],
    }

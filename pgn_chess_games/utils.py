import numpy as np
from PIL import Image
import io


def img_bytes_to_num(bytes_img: bytes):
    """Transform image in bytes to numpy array.

    Args:
        img (bytes): Image in byte form.

    Returns:
        int: Numpy array.
    """

    img = io.BytesIO(bytes_img)
    type(img)
    pill_img = Image.open(img)
    nparray_img = np.array(pill_img)
    return nparray_img


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

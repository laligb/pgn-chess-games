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


def mockup_predict(num_img):
    """
    Mockup function to be substituted ince we have the true model.

    Args:
        num_img (int): Numpy array
    """

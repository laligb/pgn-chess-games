import numpy as np
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def img_bytes_to_num(bytes_img: bytes):
    """Transform image in bytes to numpy array.

    Args:
        img (bytes): Image in byte form.

    Returns:
        int: 2D Numpy array.
    """

    img = io.BytesIO(bytes_img)
    pill_img = Image.open(img)
    grayscale_img = ImageOps.grayscale(pill_img)
    nparray_img = np.array(grayscale_img)
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


def preproc_image(nparray_img) -> list:
    """
    Preprocess the photo of a scoresheet expressed as a 2D numpy array, cuts it
    into pieces and returns a list of images

    Args:
        nparray_img (int): scoresheet photo as 2D numpy array

    Returns:
        list: List of boxes cut from the main image
    """
    image = nparray_img

    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))

    cleared = clear_border(bw)

    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    regions = regionprops(label_image)

    ALL_BOXES = []
    for i, region in enumerate(regions):
        if region.area >= 1000:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
            ax.add_patch(rect)

            region_image = image[minr:maxr, minc:maxc]
            ALL_BOXES.append(region_image)

    return ALL_BOXES

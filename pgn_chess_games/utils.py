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
import base64


def img_bytes_to_num(img: bytes) -> int:
    """Transform image in bytes to numpy array.

    Args:
        img (bytes): Image in bytes form.

    Returns:
        int: 2D Numpy array.
    """
    # decoded_img = base64.b64decode(base64_img)
    bytes_img = io.BytesIO(img)
    pill_img = Image.open(bytes_img)
    grayscale_img = ImageOps.grayscale(pill_img)
    nparray_img = np.array(grayscale_img)
    return nparray_img


def mockup_predict():
    """
    Mockup function to be substituted once we have the true model.

    Args:
        (str): string of moves in PGN format
    """
    return {
        "white": [
            "e4",
            "Nf3",
            "Bc4",
            "b4",
            "c3",
            "d4",
            "0-0",
            "Qb3",
            "e5",
            "Re1",
            "Ba3",
            "Qxb5",
            "Qa4",
            "Nbd2",
            "Ne4",
            "Bxd3",
            "Nf6+",
            "exf6",
        ],
        "black": [
            "e5",
            "Nc6",
            "Bc5",
            "Bxb4",
            "Ba5",
            "exd4",
            "d3",
            "Qf6",
            "Qg6",
            "Nge7",
            "b5",
            "Rb8",
            "Bb6",
            "Bb7",
            "Qf5",
            "Qh5",
            "gxf6",
            "Rg8",
        ],
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


def json_to_pgn(json_moves: dict) -> str:
    """
    Transforms a json list of moves to a str with the moves in PGN format

    Args:
        json_moves (dict): json with the moves in PGN notation as returned by the predictor:

        {
            "white": [move 1, move 2, move 3,...]
            "black": [move 1, move 2, move 3,...]
        }

    Returns:
        str: string in PGN format without headers:
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.b4 Bxb4 5.c3 Ba5 6.d4 exd4 7.O-O d3 8.Qb3 Qf6 9.e5 Qg6 10.Re1 Nge7 11.Ba3 b5 12.Qxb5 Rb8 13.Qa4 Bb6 14.Nbd2 Bb7 15.Ne4 Qf5 16.Bxd3 Qh5 17.Nf6+ gxf6 18.exf6 Rg8 "

    """
    total_moves = len(json_moves["white"]) + 1
    legal_moves = []
    with open("legal_moves.txt", "r") as txt_moves:
        legal_moves = [txt_move.replace("\n", "") for txt_move in txt_moves]

    legal_moves.append("O-O")
    legal_moves.append("O-0-O")
    print(legal_moves)

    pgn_moves = ""

    for move in range(1, total_moves):
        if json_moves["white"][move - 1] in legal_moves:
            pgn_moves += f"{str(move)}.{json_moves['white'][move-1]}"
        else:
            pgn_moves += f"{str(move)}.missed"

        if move <= len(json_moves["black"]):
            if json_moves["black"][move - 1] in legal_moves:
                pgn_moves += f" {json_moves['black'][move-1]} "
            else:
                pgn_moves += " missed "

    return pgn_moves

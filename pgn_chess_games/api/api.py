from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pgn_chess_games.utils import *

# from pgn-chess-games.model import Model #TODO import the model

app = FastAPI()
# app.state.model = Model() #TODO assign the model

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Generic endpoint
@app.get("/")
def hello():
    """
    Generic root endpoint

    Returns:
        json: Returns a hello world
    """
    return {"response": "Hello world"}


# Endpoint where the images are posted
@app.post("/upload")
async def receive_image(img: UploadFile = File(...)):
    """
    Endpoint to process the images and send them to the model to get a
    prediction.

    Args:
        img (UploadFile, optional): Photo of the scoresheet.
        Defaults to File(...).

    Returns:
        json: dictionary with two lists of movements as strings, one for the
        Whites player and one for the Blacks player.
        {
            "white": ["move 1","move 2","move 3"],
            "black": ["move 1","move 2","move 3"]
        }
    """
    bytes_image = await img.read()

    # Translate img bytes to 1D numpy array
    num_img = img_bytes_to_num(bytes_image)

    # Preprocess image to cut it into boxes
    all_boxes = preproc_image(num_img)

    # Call the model
    # model = app.state.model
    json_moves = mockup_predict()  # TODO Call the model
    pgn_moves = json_to_pgn(json_moves)

    return pgn_moves

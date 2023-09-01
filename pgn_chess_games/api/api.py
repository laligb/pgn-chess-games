from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pgn_chess_games.utils import *
from pgn_chess_games.model.registry import *
from pgn_chess_games.model.main import get_predictions_decoded

app = FastAPI()
# app.state.model = load_model() #There's no need to load the model

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
async def receive_image(img: UploadFile = File(...)) -> str:
    """
    Endpoint to process the images and send them to the model to get a
    prediction.

    Args:
        img (UploadFile, optional): Photo of the scoresheet.
        Defaults to File(...).

    Returns:
        str: string in PGN format without headers:
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.b4 Bxb4 5.c3 Ba5 6.d4 exd4 7.O-O d3 8.Qb3 Qf6 9.e5 Qg6 10.Re1 Nge7 11.Ba3 b5 12.Qxb5 Rb8 13.Qa4 Bb6 14.Nbd2 Bb7 15.Ne4 Qf5 16.Bxd3 Qh5 17.Nf6+ gxf6 18.exf6 Rg8 "
    """
    bytes_image = await img.read()

    # Translate img in base64 to 2D numpy array
    num_img = img_bytes_to_num(bytes_image)

    # Preprocess image to cut it into boxes
    all_boxes = preproc_image(num_img)

    # Call the model
    list_moves = get_predictions_decoded(all_boxes)

    # json_moves = mockup_predict()

    json_moves = {"white": list_moves[0::2], "black": list_moves[1::2]}

    pgn_moves = json_to_pgn(json_moves)

    return pgn_moves

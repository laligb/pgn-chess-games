from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pgn_chess_games.utils import *

# from pgn_chess_games.model.main import predict
from pgn_chess_games.api.box_extraction import box_extraction
import os

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

    # Save img locally
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
    file_path = os.path.join(LOCAL_DATA_PATH, "temp", "scoresheet.png")
    cropped_dir_path = os.path.join(LOCAL_DATA_PATH, "temp", "crop/")

    with open(file_path, "wb") as newfile:
        newfile.write(bytes_image)

    # Extract boxes and save them locally
    box_extraction(file_path, cropped_dir_path)

    # Predict
    # list_moves = predict_batch()

    # Delete temporary images
    print(f"Deleting temporary scoresheet file.")
    if os.path.isfile(file_path):
        os.remove(file_path)

    print(f"Deleting temporary files from {cropped_dir_path} ...")
    files = os.listdir(cropped_dir_path)
    for file in files:
        temp_filepath = os.path.join(cropped_dir_path, file)
        if os.path.isfile(temp_filepath):
            os.remove(temp_filepath)
    print("✅ All temporary files deleted.")

    ####### OLD METHOD #########
    # Translate img in base64 to 2D numpy array
    # num_img = img_bytes_to_num(bytes_image)
    # Preprocess image to cut it into boxes
    # all_boxes = preproc_image(num_img)
    # Call the model
    # list_moves = get_predictions_decoded(all_boxes)
    ####### END OF OLD METHOD #######

    # json_moves = mockup_predict()

    json_moves = {"white": list_moves[0::2], "black": list_moves[1::2]}

    pgn_moves = json_to_pgn(json_moves)

    return pgn_moves

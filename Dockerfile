FROM tensorflow/tensorflow
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
COPY pgn_chess_games pgn_chess_games
COPY setup.py setup.py
COPY Makefile Makefile
RUN make install_package
CMD uvicorn pgn_chess_games.api.api:app --host 0.0.0.0 --port $PORT

FROM python:3.10.6-buster
RUN mkdir gcp
COPY gcp/pgn-chess-games-b15121f9865c.json gcp/pgn-chess-games-b15121f9865c.json
COPY requirements-api.txt requirements.txt
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install \
  'wget'\
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y
COPY pgn_chess_games pgn_chess_games
COPY setup.py setup.py
COPY Makefile Makefile
COPY legal_moves.txt legal_moves.txt
RUN make install_package
RUN make prepare_dirs
CMD uvicorn pgn_chess_games.api.api:app --host 0.0.0.0 --port $PORT

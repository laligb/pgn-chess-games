FROM tensorflow/tensorflow:latest-gpu
RUN mkdir gcp
COPY gcp/pgn-chess-games-b15121f9865c.json gcp/pgn-chess-games-b15121f9865c.json
COPY requirements-train.txt requirements.txt
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
RUN make download_datasets
RUN make install_package
CMD sh

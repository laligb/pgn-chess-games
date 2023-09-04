install_package:
	@pip uninstall -y pgn-chess-games || :
	@pip install -e .

run_api:
	@uvicorn pgn_chess_games.api.api:app --reload

run_docker:
	@docker run -p 8080:8000 chess

run_model_IAM:
	@python -c 'from pgn_chess_games.model.main import main_IAM; main_IAM()'

predict_sample:
	@python -c 'from pgn_chess_games.model.main import predict; predict(s)'

run_model_chess:
	@python -c 'from pgn_chess_games.model.main import train_chess; train_chess()'

run_model_chess:
	@python -c 'from pgn_chess_games.model.main import train_chess; train_chess()'

download_datasets:
	@wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip
	@unzip -qq IAM_Words.zip
	@mkdir ~/.data
	@mkdir ~/.data/models
	@mkdir ~/.data/words
	@mkdir ~/.data/dictionary
	@tar -xf IAM_Words/words.tgz -C ~/.data/words
	@mv IAM_Words/words.txt ~/.data/words

build_docker_train:
	@docker build -t davidrosillo/chess . -f train.Dockerfile

push_docker_train:
	@docker push davidrosillo/chess

build_docker_api:
	@docker build -t eu.gcr.io/pgn-chess-games/chess . -f api.Dockerfile

push_docker_api:
	@docker push eu.gcr.io/pgn-chess-games/chess

run_docker_train:
	@docker run -it --rm --env-file .env --runtime=nvidia davidrosillo/chess

run_docker_api:
	@docker run -e PORT=8000 -p 8080:8000 eu.gcr.io/pgn-chess-games/chess

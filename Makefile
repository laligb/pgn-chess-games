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
	@python -c 'from pgn_chess_games.model.main_chess import run_model_chess; run_model_chess()'

download_datasets:
	@wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip
	@unzip -qq IAM_Words.zip

	@mkdir ~/.data
	@mkdir ~/.data/models
	@mkdir ~/.data/words
	@tar -xf IAM_Words/words.tgz -C ~/.data/words
	@mv IAM_Words/words.txt ~/.data/words

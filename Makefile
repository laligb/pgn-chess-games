install_package:
	@pip uninstall -y pgn-chess-games || :
	@pip install -e .

run_model:
	python -c 'from pgn_chess_games.model.main import main; main()'

predict_sample:
	python -c 'from pgn_chess_games.model.main import predict; predict(s)'

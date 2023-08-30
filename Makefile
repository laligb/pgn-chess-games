install_package:
	@pip uninstall -y pgn-chess-games || :
	@pip install -e .

<<<<<<< HEAD
run_model:
	python -c 'from pgn_chess_games.model.main import main; main()'

predict_sample:
	python -c 'from pgn_chess_games.model.main import predict; predict(s)'
=======
run_api:
	@uvicorn pgn_chess_games.api.api:app --reload

run_docker:
	@docker run -p 8080:8000 chess
>>>>>>> 8999c1f4ac8f82f96091c58337324adea21f425a

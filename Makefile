install_package:
	@pip uninstall -y pgn-chess-games || :
	@pip install -e .

run_api:
	@uvicorn pgn_chess_games.api.api:app --reload

run_docker:
	@docker run -p 8080:8000 chess

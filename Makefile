install_package:
	@pip uninstall -y pgn-chess-games || :
	@pip install -e .

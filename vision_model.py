import os

from gradio_client import Client, handle_file
from stockfish import Stockfish
from dotenv import load_dotenv

load_dotenv()

engine = Stockfish(path=os.getenv("PATH_STOCKFISH_ENGINE"), depth=18, parameters={"Threads": 8, "Minimum Thinking Time": 30, "Hash": 2048})

# retrieve fen string from image with YOLO CV model
client = Client("yamero999/chess-fen-generation-api")
result = client.predict(
		image=handle_file('chessboard1.jpg'),
		api_name="/predict_ui"
)
fen_string = result[0]

# Process this string
side_to_move = "b"
parts = fen_string.split()
ranks = parts[0].split("/")
if side_to_move == "b":
    flipped_ranks = [rank[::-1] for rank in ranks[::-1]]
    parts[0] = "/".join(flipped_ranks)
parts[1] = side_to_move
# remove castling issue
if "KQkq" == parts[2]:
    parts[2] = "-"
# Join everything back into a FEN string
new_fen = " ".join(parts)

# Predict the next move with Stockfish
engine.set_fen_position(new_fen)
print(engine.get_best_move())
print(engine.get_top_moves(3))
import os

from gradio_client import Client, handle_file
from stockfish import Stockfish
from dotenv import load_dotenv

load_dotenv()


def load_engine():
    return Stockfish(path=os.getenv("PATH_STOCKFISH_ENGINE"), depth=18, parameters={"Threads": 8, "Minimum Thinking Time": 30, "Hash": 2048})


def get_fen_string_from_chessboard_image(path_image):
    client = Client("yamero999/chess-fen-generation-api")
    result = client.predict(
            image=handle_file(path_image),
            api_name="/predict_ui"
    )
    return result[0]

# Process this string
fen_string = get_fen_string_from_chessboard_image("chessboard.jpg")
side_to_move = "b"

def process_fen_string(current_string, side_to_move):
    parts = current_string.split()
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

processed_fen_string = process_fen_string(fen_string, side_to_move)

engine = load_engine()

# Predict the next move with Stockfish
engine.set_fen_position(processed_fen_string)
print(engine.get_best_move())
print(engine.get_top_moves(3))
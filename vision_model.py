import os

from gradio_client import Client, handle_file
from stockfish import Stockfish
from dotenv import load_dotenv

load_dotenv()

side_mapping = {
    "white": "w",
    "black": "b"
}

params_engine = {
    "Threads": 8, 
    "Minimum Thinking Time": 30, 
    "Hash": 2048
}

engine = Stockfish(path=os.getenv("PATH_STOCKFISH_ENGINE"), depth=18, parameters=params_engine)


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
    return new_fen

def get_fen_string_from_image_path(path_image, side_to_move):
    client = Client("yamero999/chess-fen-generation-api")
    result = client.predict(
            image=handle_file(path_image),
            api_name="/predict_ui"
    )
    fen_string = result[0]
    if side_to_move in ["white", "black"]:
        side_to_move = side_mapping[side_to_move]
    return process_fen_string(fen_string, side_to_move)

def compute_top_k_best_move(fen_string, k=3):
    engine.set_fen_position(fen_string)
    return engine.get_top_moves(k)


if __name__ == "__main__":
    side_to_move = "b"
    fen_string = get_fen_string_from_chessboard_image("chessboard.jpg", side_to_move)
    next_moves = compute_top_k_best_move(fen_string, engine)    
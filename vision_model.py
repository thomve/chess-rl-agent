import os

from gradio_client import Client, handle_file
from stockfish import Stockfish
from dotenv import load_dotenv
from rich import print

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
    best_moves = engine.get_top_moves(k)
    list_best_fen_moves = compute_fen_strings_from_best_moves(best_moves)
    list_iframes = get_list_iframe(list_best_fen_moves)
    return best_moves, list_iframes[0], list_iframes[1], list_iframes[2]

def compute_fen_strings_from_best_moves(moves):
    current_fen_string = engine.get_fen_position()
    list_fen_string = []
    for move in moves:
        engine.make_moves_from_current_position([move['Move']])
        fen_string_move = engine.get_fen_position()
        engine.set_fen_position(current_fen_string)
        list_fen_string.append(fen_string_move)
    return list_fen_string


iframe_html = """<iframe src="http://localhost:3000/chessboard.html?fen={}"
        width="500"
        height="500"
        frameborder="0"
        style="border: none; overflow: hidden;"></iframe>
"""

def get_list_iframe(list_fen_string):
    list_iframe = []
    for fen_string in list_fen_string:
        list_iframe.append(iframe_html.format(fen_string))
    return list_iframe

if __name__ == "__main__":
    print("[bold green]Computing fen string[/bold green]!")
    side_to_move = "b"
    fen_string = get_fen_string_from_image_path("chessboard.jpg", side_to_move)
    print(fen_string)
    next_moves = compute_top_k_best_move(fen_string)
    print(next_moves)
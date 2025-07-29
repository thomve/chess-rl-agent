import os
import gradio as gr

from vision_model import get_fen_string_from_image_path, compute_top_k_best_move, compute_fen_strings_from_best_moves


def get_fen_string_from_chess_board_image(image, side_to_move):
    try:
        temp_path = "temp.jpg"
        image.save(temp_path)
        fen_string = get_fen_string_from_image_path(temp_path, side_to_move)
        os.unlink(temp_path)
        return fen_string
    except Exception as e:
        return "Error"


with gr.Blocks() as demo:
    gr.Markdown("## Chess vision")
    gr.Markdown("Predict the next move based on a chessboard image")
    gr.Markdown("*This project is for educational purposes only and should not be used in any over the board or online chess games.*")
    with gr.Row():
        with gr.Column():
            # path_chessboard_image = gr.Textbox(label="Path to chessboard image")
            chessboard_image = gr.Image(height=600, type="pil")
        with gr.Column():
            side_to_move = gr.Radio(["white", "black"], label="Side to move")
            fen_button = gr.Button("Compute FEN String")
            output_fen_box = gr.Textbox(label="FEN string")
            gr.Markdown("For more information on FEN strings, please refer to https://www.chess.com/terms/fen-chess")
            next_move_button = gr.Button("Compute best moves")
            output_next_moves = gr.Textbox(label="Best moves")
    with gr.Row():
        with gr.Column():
            chessboard_html_1 = gr.HTML(max_height=600)
        with gr.Column():
            chessboard_html_2 = gr.HTML(max_height=600)
        with gr.Column():
            chessboard_html_3 = gr.HTML(max_height=600)
    fen_button.click(fn=get_fen_string_from_chess_board_image, inputs=[chessboard_image, side_to_move], outputs=[output_fen_box])
    next_move_button.click(fn=compute_top_k_best_move, inputs=[output_fen_box], outputs=[output_next_moves, chessboard_html_1, chessboard_html_2, chessboard_html_3])

demo.launch()
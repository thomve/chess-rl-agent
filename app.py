import gradio as gr

from vision_model import load_engine, get_fen_string_from_chessboard_image, compute_top_k_best_move

with gr.Blocks() as demo:
    gr.Markdown("## Chess vision")
    gr.Markdown("Predict the next move based on a chessboard image")
    # chessboard_image = gr.Image(height=600)
    with gr.Row():
        path_chessboard_image = gr.Textbox(label="Path to chessboard image")
        color_to_move = gr.Radio(["white", "black"], label="Color to move")
    fen_button = gr.Button("Compute FEN String")
    output_fen_box = gr.Textbox(label="FEN string")
    next_move_button = gr.Button("Compute best moves")
    output_next_moves = gr.Textbox(label="Best moves")
    fen_button.click(fn=get_fen_string_from_chessboard_image, inputs=[path_chessboard_image, color_to_move], outputs=output_fen_box)
    next_move_button.click(fn=compute_top_k_best_move, inputs=[output_fen_box], outputs=output_next_moves)

demo.launch()
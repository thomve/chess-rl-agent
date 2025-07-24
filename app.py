import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("## Chess vision")
    gr.Markdown("Predict the next move based on a chessboard image")
    chessboard_image = gr.Image()
    print(chessboard_image)


demo.launch()
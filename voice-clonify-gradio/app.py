import gradio as gr
import time

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Inter')],
    font_mono=[gr.themes.GoogleFont('Fira Mono')],
)


with gr.Blocks(theme=theme, title="Voice Clonify") as demo:
    gr.Markdown(
    """
    # Voice Clonify
    Start typing below to see the output.
    """)
    textbox = gr.Textbox(label="Name", placeholder="Type here")
    slider = gr.Slider(label="Count", minimum=0, maximum=100, step=1)
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")
    output = gr.Textbox(label="Output")

    def repeat(name, count):
        time.sleep(3)
        return name * count

    button.click(repeat, [textbox, slider], output)
demo.launch()

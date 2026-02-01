import logging
import queue
import threading
import time
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

from day5 import opportunities_dataframe
### Internal classes
from deal_agent_framework import DealAgentFramework
from log_utils import reformat

load_dotenv(override=True)


class App:
    def __init__(self):
        ### lazy initialization
        self.agent_framework = None
    ### And assign it here
    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()

        return self.agent_framework

    def run(self):
        with gr.Blocks(title="WorthBrain", fill_width=True) as ui:
            log_data = gr.State([])

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px"><strong>WorthBrain</strong> - Autonomous Agent Framework that hunts for deals</div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">A proprietary fine-tuned LLM deployed on Modal and a RAG pipeline with a frontier model collaborate to send push notifications with great online deals.</div>'
                )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    column_count=5,
                    max_height=400,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value="PLACEHOLDER HERE", show_label=False)

            ui.load(
                ### connect a load event handler
                ### set inputs
                ### set outputs
            )
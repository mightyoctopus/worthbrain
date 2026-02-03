import logging
import queue
import threading
import time
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv
from typing import Tuple

### Internal classes
from deal_agent_framework import DealAgentFramework
from log_utils import reformat

load_dotenv(override=True)

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """

def setup_logging(log_queue):
    """
    Register and configure for logging
    """
    logger = logging.getLogger()

    ### Remove previous QueueHandlers to avoid duplication and UI floods (handler duplication)
    for h in list(logger.handlers):
        if isinstance(h, QueueHandler):
            logger.removeHandler(h)

    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
        with gr.Blocks(
                title="WorthBrain",
                fill_width=True,
                css="footer {visibility: hidden}"
        ) as ui:
            log_data = gr.State([])

            def table_for(opps):
                """
                Format the initial/final_result
                """
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]

            def stream_ui_updates(log_data, log_queue, result_queue):
                """
                A generator loop that streams log messages and opportunities data for the data frame in the UI
                """

                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None

                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_plot():
                """
                Create 3D scatter plot for the result data
                """
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)

                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7),
                        )
                    ]
                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),  # Make x-axis twice as long
                        camera=dict(
                            eye=dict(x=1.6, y=1.6, z=0.8)  # Adjust camera position
                        ),
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2),
                )

                return fig


            def do_run():
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(initial_log_data):
                """
                Load event handler that starts a background worker thread, streaming log updates
                to the UI and yields incremental UI updates.
                :yield:
                    Tuple containing:
                    - log_data (persistent log state)
                    - output (HTML-formatted log message)
                    - final_result (table data for opportunities)
                """
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    """
                    Background task that executes the agent framework,
                    sends final results to result_queue, and emits logs
                    through the logging system.
                    """
                    result = do_run()
                    result_queue.put(result)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                for log_data, html_output, final_result in stream_ui_updates(
                    initial_log_data, log_queue, result_queue
                    ):
                    yield log_data, html_output, final_result

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
                    plot = gr.Plot(value=get_plot(), show_label=False)

            ### Footer
            with gr.Row():
                gr.Markdown(
                    """
                    <div style="
                        text-align: center;
                        font-size: 13px;
                        color: #888;
                        padding: 15px 0;
                    ">
                    © 2026 WorthBrain · Autonomous Deal Hunting Intelligence
                    </div>
                    """
                )

            ui.load(
                ### connect a load event handler
                run_with_logging,
                ### set inputs
                inputs=[log_data],
                ### set outputs
                outputs=[log_data, logs, opportunities_dataframe]
            )

        ui.launch(inbrowser=True)


if __name__ == "__main__":
    App().run()
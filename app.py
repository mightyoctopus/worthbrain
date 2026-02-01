import logging
import queue
import threading
import time
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv
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
        pass
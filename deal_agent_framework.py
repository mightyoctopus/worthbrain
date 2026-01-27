import os
import sys
import logging
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from sklearn.manifold import TSNE
import numpy as np
### Internal classes
from agents.deterministic_planning_agent import DeterministicPlanningAgent
from agents.deals import Opportunity

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(self.DB)
        self.memory = ... ### read_memory() method here later
        self.collection = client.get_or_create_collection("products")
        self.planner = None # Deterministic Agent assigned here later

    def init_agent_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework...")
            self.planner = DeterministicPlanningAgent(self.collection)
            self.log("Agent Framework is ready!")

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)












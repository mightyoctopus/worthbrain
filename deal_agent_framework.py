import os
import sys
import logging
import json
from importlib.metadata import metadata
from typing import List, Optional
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
    'Appliances',
    'Automotive',
    'Cell_Phones_and_Accessories',
    'Electronics','Musical_Instruments',
    'Office_Products',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    "Industrial_and_Scientific",
    "Arts_Crafts_and_Sewing",
    "Handmade_Products",
    "All_Beauty",
    "Gift_Cards"
]

COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan', "purple", "black", "gray", "pink", "olive"]



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
    DB = os.getenv("PRODUCTION_DB", "products_vectorstore")
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(self.DB)
        self.memory: List[Opportunity] = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self.planner = None # lazy initialization

    def init_agent_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework...")
            self.planner = DeterministicPlanningAgent(self.collection)
            self.log("Agent Framework is ready!")

    def read_memory(self) -> List[Opportunity]:
        """
        Read the memory.json file and convert it into Opportunity pydantic model
        :return: A list of Opportunity models, or an empty list if no memory file is found.
        """

        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as f:
                data: List[dict] = json.load(f)
                result = [Opportunity(**opp) for opp in data]
            return result
        return []

    def write_memory(self) -> None:
        """
        Write opportunities into the memory.json file
        """
        if self.memory:
            data: List[dict] = [opp.model_dump() for opp in self.memory]
            with open(self.MEMORY_FILENAME, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        """
        Reset data in the memory.json file back to the default state
        """
        data = []

        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as f:
                data = json.load(f)
        truncated = data[:2]

        with open(cls.MEMORY_FILENAME, "w") as f:
            json.dump(truncated, f, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """
        Initialize and start to run the planner agent that manages the entire agentic workflow

        Process:
        1. Init the planner agent
        2. Fetch result (Opportunity pydantic model -- single Opportunity model with the best deal picked out)
        3. If result is fetched successfully, append it to the memory and write it into memory.json
        4. Return the Opportunity models

        :return: A list of Opportunity models
        """
        self.log("Deal Agent Framework is initializing Planning Agent...")
        if not self.planner:
            self.init_agent_as_needed()

        result: Opportunity = self.planner.plan(memory=self.memory)
        self.log(f"Planning Agent has completed and returned {result}")
        if result:
            self.memory.append(result)
            self.write_memory()

        return self.memory


    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")

        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]

        tsne = TSNE(n_components=3, random_state=42, n_jobs=1)
        reduced_vectors = tsne.fit_transform(vectors)

        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()








############ TEST ############
# agent = DealAgentFramework()
# print(agent.reset_memory())



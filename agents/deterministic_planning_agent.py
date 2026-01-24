from typing import Optional, List
from agents.agents import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class DeterministicPlanningAgent(Agent):

    name = "Deterministic Planning Agent"
    color = Agent.GREEN
    VALUE_GAP_THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        :param collection: Chroma DB collection provided for the frontier model with RAG
        """
        self.log("Planning Agent is initializing...")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messanger = MessagingAgent()
        self.log("Planning Agent is ready!")
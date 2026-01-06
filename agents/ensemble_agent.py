import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from agents.agents import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent...")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.log("Ensemble Agent is ready!")


    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        
        :param description: the description of a product
        :return: an estimate of its price
        """
        
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and neural network agents...")

        desc_into_str = description.prompt.replace(
            "How much does this cost to the nearest dollar?\n\n", ""
        ).split("\n\nPrice is $")[0]
        
        specialist = self.specialist.price(desc_into_str)
        frontier = self.frontier.price(desc_into_str)
        neural_network = self.neural_network.price(desc_into_str)

        ### Only include specialist and frontier model's results to get the average price in a more stable range
        ### as neural network model often makes drastically far off price estimates.

        ### rough_price V1:
        # rough_price = (specialist + frontier) / 2

        ### rough_price V2:
        # rough_price = specialist * 0.2 + frontier * 0.8

        ### Apply a different pricing distribution depending on the estimated price range based on each model's best accuracy by range
        ### Experiment Logs: https://docs.google.com/document/d/1RqaQeTpferlkdPNkXn1aEnrSq9d5uQs7cSWBWDS7As8/edit?tab=t.0
        # if rough_price < 50:
        #     combined = frontier * 0.7 + specialist * 0.3
        # elif rough_price < 100:
        #     combined = frontier * 0.35 + specialist * 0.6 + neural_network * 0.05
        # elif rough_price < 150:
        #     combined = frontier * 0.9 + specialist * 0.08 + neural_network * 0.02
        # elif rough_price < 200:
        #     combined = frontier * 0.9 + specialist * 0.07 + neural_network * 0.03
        # elif rough_price < 250:
        #     combined = frontier * 0.7 + specialist * 0.3
        # elif rough_price < 300:
        #     combined = frontier * 0.6 + specialist * 0.3 + neural_network * 0.1
        # elif rough_price < 350:
        #     combined = frontier * 0.8 + specialist * 0.2
        # elif rough_price < 400:
        #     combined = frontier * 0.9 + specialist * 0.1
        # else:
        #     combined = frontier * 0.9 + specialist * 0.1

        ###### Simplified version of pricing contribution here

        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return round(combined, 2)
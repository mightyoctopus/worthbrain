import math
import pandas as pd
from PIL.ImageStat import Global
from sklearn.linear_model import LinearRegression
import joblib
from agents.agents import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from typing import Optional


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

    def estimate_price_range(
            self,
            frontier: float,
            specialist: float,
            contribution_option: str ="o3"
    ):
        """
        Provide rough_price options that can be selectable to determine which model contributes more to deciding the price range.
        Some rough_price eliminates certain models to remove the volatility and to keep more stable price range estimations
        :param frontier: The outcome of price range estimate by the frontier model(GPT 5-mini)
        :param specialist: The outcome of price range estimate by the fine-tuned LLaMA 3.1 8b model
        :param contribution_option: (Optional) Determines the combination of price models dominance for estimating the rough price range
        """
        if contribution_option == "o1":
            ### rough_price Option 1:
            return (specialist + frontier) / 2
        elif contribution_option == "o2":
            ## rough_price Option 2:
            return specialist * 0.2 + frontier * 0.8
        elif contribution_option == "o3":
            ## rough_price Option 3 ("o3"):
            return frontier
        else:
            raise ValueError(f"Unknown contribution_option: {contribution_option}")


    ### Total Price Range Error for Each:
    frontier_err, special_err, neural_err = (0, 0, 0)


    def price(self, description: str, y_truth: float=None) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price

        :param
            description: the description of a product
            y_truth: the ground truth value(price) of a tested item, used for performance benchmark
        :return: an estimate of its price
        """

        self.log("Running Ensemble Agent - collaborating with specialist, frontier and neural network agents...")

        processed_desc = description.replace(
            "How much does this cost to the nearest dollar?\n\n", ""
        ).split("\n\nPrice is $")[0]

        specialist = self.specialist.price(processed_desc)
        frontier = self.frontier.price(processed_desc)
        neural_network = self.neural_network.price(processed_desc)

        rough_price = self.estimate_price_range(frontier, specialist, "o3")

        ### Apply a different pricing distribution depending on the estimated price range based on each model's best accuracy by range
        ### Experiment Logs: https://docs.google.com/document/d/1RqaQeTpferlkdPNkXn1aEnrSq9d5uQs7cSWBWDS7As8/edit?tab=t.0

        ### Simplified version of allocating model dominance
        if rough_price < 100:
            combined = frontier * 0.7 + specialist * 0.3
        elif rough_price < 200:
            combined = frontier * 0.85 + specialist * 0.1 + neural_network * 0.05
        elif rough_price < 300:
            combined = frontier * 0.7 + specialist * 0.2 + neural_network * 0.1
        else:
            combined = frontier * 0.9 + specialist * 0.1

        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")

        ### This code below was made to check absolute error ranges for each model for testing/experiment purposes.
        ### At inference, this below doesn't affect it.
        if y_truth is not None:
            f_err = abs(frontier - y_truth)
            s_err = abs(specialist - y_truth)
            n_err = abs(neural_network - y_truth)

            EnsembleAgent.frontier_err += f_err
            EnsembleAgent.special_err += s_err
            EnsembleAgent.neural_err += n_err

            self.log(f"Frontier Err: {EnsembleAgent.frontier_err:,.2f}")
            self.log(f"Special Err: {EnsembleAgent.special_err:,.2f}")
            self.log(f"Neural Err: {EnsembleAgent.neural_err:,.2f}")

        return round(combined, 2)
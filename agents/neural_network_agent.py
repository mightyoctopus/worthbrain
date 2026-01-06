import joblib
import torch

from agents.agents import Agent
from models.neural_network import NeuralNetwork


class NeuralNetworkAgent(Agent):
    """
    An agent that runs a neural network model to predict the prices
    """

    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, model=None, input_size=5000):
        """
        Set up this agent by creating an instance of the model class
        """
        self.log("Neural Network Agent is initializing...")
        self.vectorizer = joblib.load("models/vectorizer.joblib")
        self.model = NeuralNetwork(input_size)
        self.model.load_state_dict(
            torch.load("models/neural_network_pricer_model.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()
        self.log("Neural Network Agent is ready!")

    def price(self, description: str) -> float:
        """
        Make a call to return the estimate of the price of a given item description

        Args:
            description (str): Product description provided for price estimation
        """

        with torch.no_grad():
            self.log("Neural Network Agent is processing the price estimation...")
            vector = self.vectorizer.transform([description])
            vector = torch.FloatTensor(vector.toarray())
            prediction = self.model(vector).item()

            result = max(0.0, prediction)
            self.log(f"Neural Network Agent completed -- predicting ${result:.2f}")

        return result
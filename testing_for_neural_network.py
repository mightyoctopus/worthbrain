import math
import matplotlib.pyplot as plt

from testing import Tester

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}


class TesterForNeuralNetwork(Tester):

    def __init__(self, predictor, data, prices, titles, size=250):
        super().__init__(predictor, data, size)
        self.prices = prices
        self.titles = titles

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = self.prices[i]
        
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        
        title = self.titles[i] if len(self.titles[i]) <= 40 else self.titles[i][:40] + "..."
        
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        
        print(
            f"{COLOR_MAP[color]}{i + 1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    @classmethod 
    def test(cls, function, data, prices, titles):
        cls(function, data, prices, titles).run()
from typing import Optional, List, Dict
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolMessageParam

### Internal Classes
from agents.agents import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-5-mini"

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log(f"{self.name} is initializing...")
        self.scanner_agent = ScannerAgent()
        self.ensemble_agent = EnsembleAgent(collection)
        self.messanger_agent = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None
        self.opportunity = None
        self.log(f"{self.name} is ready")


    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool via LLM to scan deals from the internet
        """
        self.log(f"{self.name} is calling Scanner Agent to scan deals online")
        scanned_deals = self.scanner_agent.scan(memory=self.memory)
        return scanned_deals.model_dump_json() if scanned_deals else "No deals found"


    def estimate_true_value(self, description: str) -> str:
        """
        Run the tool via LLM to estimate the actual price value of a product
        """
        self.log(f"{self.name} is calling Ensemble Agent to estimate the true value")
        estimate = self.ensemble_agent.price(description=description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(
            self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> str:
        """
        Run the tool via LLM to notify the user of the best deal
        """
        self.log(f"{self.name} is calling Messanger Agent to notify the best deal")
        self.messanger_agent.notify(
            description,
            deal_price,
            estimated_true_value,
            url
        )
        deal = Deal(
            product_description=description,
            price=deal_price,
            url=url
        )
        discount: float | int = estimated_true_value - deal_price
        self.opportunity = Opportunity(
            deal=deal, estimate=estimated_true_value, discount=discount
        )

        return "Notification has been sent"


    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated for the actual price"
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal scraped from the internet"
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value that this is worth",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this deal as scraped from the internet"
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False
        }
    }


    def get_tools(self):
        """
        Return the JSON for the tools to be used
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function}
        ]


    def handle_tool_call(self, message) -> List[ChatCompletionToolMessageParam]:
        """
        Actually call the tools associated with this message
        """
        results = []

        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal
        }

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments: dict = tool_call.function.arguments
            tool = mapping.get(tool_name)

            result = tool(**arguments) if tool else None
            results.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )

        return results

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


    def plan(self, memory: List[str] = None) -> Optional[Opportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one has been newly surfaced, otherwise None
        """

        self.log(f"{self.name} is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]

        while True:
            response = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.get_tools()
            )

            if response.choices[0].finish_reason == "tool_calls":
                tool_call_msg = response.choices[0].message
                tool_call_result = self.handle_tool_call(tool_call_msg)
                messages.append(tool_call_msg) # Assistant message
                messages.extend(tool_call_result) # Tool call messages
                continue
            else:
                break

        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity















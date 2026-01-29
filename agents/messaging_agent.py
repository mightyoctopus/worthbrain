import os
import requests
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from agents.agents import Agent
from agents.deals import Opportunity

class MessagingAgent(Agent):
    ### PushOver API Post endpoint
    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
    MODEL = "gpt-5-mini"
    name = "Messaging Agent"
    color = Agent.WHITE


    def __init__(self):
        self.log(f"{self.name} is initializing...")
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.openai = OpenAI()
        self.log(f"{self.name} is set!")


    def push(self, text):
        """
        Send a Push Notification using the Pushover API
        :param text: message to send for a notification
        """

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {
        "user": self.pushover_user,
        "token": self.pushover_token,
        "message": text,
        "sound": "magic"
        }

        if not payload["user"] or not payload["token"]:
            self.log("Missing Pushover credentials")
            return

        response = requests.post(self.PUSHOVER_URL, data=payload, headers=headers, timeout=5)
        response.raise_for_status()


    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f}, "
        text += opportunity.deal.product_description[:10] + "..."
        text += opportunity.deal.url
        self.push(text)
        self.log("Messaging Agent has completed sending a push notification!")


    def craft_message(
            self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        """
        Agent feature that crafts a push notification message via LLM, which is sent to the user after the process.
        :param description: product description in a deal
        :param deal_price: the offered price of the product in a deal
        :param estimated_true_value: the actual true value in price of the product, being estimated
        """
        user_prompt = "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
        user_prompt += f"Item Description: {description}\nOffered Price: {deal_price}\nEstimated True Value: {estimated_true_value}"
        user_prompt += "\n\nRespond within only 2-3 sentence in an exiting tone which will be used to alert the user about this deal."

        messages: List[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": user_prompt}
        ]

        response = self.openai.chat.completions.create(
            model=self.MODEL,
            messages=messages
        )

        return response.choices[0].message.content


    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str):
        """
        Make an alert about the specified details
        """
        self.log(f"{self.name} is crafting a push notification message...")
        message_text = self.craft_message(description, deal_price, estimated_true_value)

        self.log(f"{self.name} is sending the notification message...")
        self.push(message_text[:200] + "..." + url)
        self.log(f"{self.name} has completed!")

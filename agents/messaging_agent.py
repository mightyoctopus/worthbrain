import os
import requests
from openai import OpenAI
from agents.agents import Agent

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
        "sound": "cashregister"
        }

        if not payload["user"] or not payload["token"]:
            self.log("Missing Pushover credentials")
            return

        response = requests.post(self.PUSHOVER_URL, data=payload, headers=headers, timeout=5)
        response.raise_for_status()

from typing import Optional, List
from agents.agents import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class DeterministicPlanningAgent(Agent):

    name = "Deterministic Planning Agent"
    color = Agent.GREEN
    DISCOUNT_THRESHOLD = 50

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

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal and convert it into Opportunity object
        :param deal: the deal, summarized from an RSS scrape
        :return: an Opportunity pydantic model, containing the estimated value and discount
        """
        self.log(f"{self.name} is running to find a potential opportunity...")
        estimate: float = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"{self.name} has processed a deal with discount ${discount:,.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)


    def plan(self, memory: List[str] = None) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log(f"{self.name} is starting the workflow...")
        if memory is None:
            memory = []
        selection: DealSelection = self.scanner.scan(memory)
        print("SELECTION:\n\n", selection)

        if selection:
            ### Convert Deal objects into Opportunity objects
            opportunities: List[Opportunity] = [self.run(deal) for deal in selection.deals[:5]]
            ### Sort opportunities by discount to select an Opportunity with the largest discount amount
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best_opp = opportunities[0]
            if best_opp.discount > self.DISCOUNT_THRESHOLD:
                self.messanger.alert(best_opp)
            self.log("Planning Agent has completed a run!")
            return best_opp if best_opp.discount > self.DISCOUNT_THRESHOLD else None

        return None

    ### Where the return value passes forward to?











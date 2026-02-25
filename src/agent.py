from typing import override, Optional

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint
from pydantic import BaseModel
from starlette.applications import Starlette

from entrypoints import search_inaturalist_observations

class INaturalistAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iNaturalist Observation Search",
            description="Searches for observations in iNaturalist (https://inaturalist.org).",
            icon=None,
            entrypoints=[
                search_inaturalist_observations.entrypoint
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        match entrypoint:
            case search_inaturalist_observations.entrypoint.id:
                await search_inaturalist_observations.run(context, request)
            case _:
                raise ValueError()



def create_app() -> Starlette:
    agent = INaturalistAgent()
    app = build_agent_app(agent)
    return app

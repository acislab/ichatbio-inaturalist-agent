import pytest
import pytest_asyncio
from ichatbio.agent_response import ArtifactResponse

from agent import INaturalistAgent


@pytest_asyncio.fixture()
def agent():
    return INaturalistAgent()


@pytest.mark.asyncio
async def test_search_inaturalist_observations(agent, context, messages):
    await agent.run(context, "Find observations of Rattus rattus", "search_inaturalist_observations", None)

    artifact = next((m for m in messages if isinstance(m, ArtifactResponse)), None)
    assert artifact
    assert artifact.mimetype == "application/json"

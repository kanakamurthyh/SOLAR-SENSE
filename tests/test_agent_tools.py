"""Agent tool layer (no API calls)."""
import json

import pytest

from solarsense import agent
from solarsense.config import ATLAS_PATH


def test_find_city():
    out, err = agent.run_tool("find_city", {"name": "bikaner"})
    assert not err and json.loads(out)["state"] == "Rajasthan"


def test_unknown_tool_is_error():
    out, err = agent.run_tool("nope", {})
    assert err and "unknown tool" in json.loads(out)["error"]


def test_bad_input_is_error_not_crash():
    out, err = agent.run_tool("analyse_site", {"latitude": "x", "longitude": 1, "capacity_kwp": 1})
    assert err


@pytest.mark.skipif(not ATLAS_PATH.exists(), reason="atlas not built")
def test_atlas_lookup_filters():
    rows = json.loads(agent.run_tool("atlas_lookup", {"region": "South", "top_n": 2})[0])["rows"]
    assert len(rows) == 2 and all(r["region"] == "South" for r in rows)


def test_tool_schemas_are_closed():
    for t in agent.TOOLS:
        assert t["input_schema"]["additionalProperties"] is False
        assert t["name"] in agent.HANDLERS

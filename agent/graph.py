"""LangGraph graph definition for the Fact-Check agent.

Nodes run sequentially. Flow: detect → image search → claim search → collect → direct response.
"""

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    search_image_evidence,
    search_claim_evidence,
    collect_sources,
    generate_final_response,
)


def _detect_scenario(state: AgentState) -> dict:
    """Detect input scenario: both, image_only, or text_only."""
    has_claim = bool(state.get("claim_text", "").strip())
    has_image = bool(state.get("image_paths", []))

    if has_claim and has_image:
        return {"scenario": "both"}
    elif has_image:
        return {"scenario": "image_only"}
    else:
        return {"scenario": "text_only"}


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph graph."""
    builder = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────
    builder.add_node("detect_scenario", _detect_scenario)
    builder.add_node("search_image_evidence", search_image_evidence)
    builder.add_node("search_claim_evidence", search_claim_evidence)
    builder.add_node("collect_sources", collect_sources)
    builder.add_node("generate_final_response", generate_final_response)

    # ── Sequential edges ────────────────────────────────────────────
    builder.add_edge(START, "detect_scenario")
    builder.add_edge("detect_scenario", "search_image_evidence")
    builder.add_edge("search_image_evidence", "search_claim_evidence")
    builder.add_edge("search_claim_evidence", "collect_sources")
    builder.add_edge("collect_sources", "generate_final_response")
    builder.add_edge("generate_final_response", END)

    return builder.compile()


# Module-level singleton
_graph = None


def get_graph():
    """Lazy-init the graph (needed after model is loaded)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

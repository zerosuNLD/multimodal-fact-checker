"""Agent state definition for the Fact-Check LangGraph pipeline."""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from operator import add


class AgentState(TypedDict):
    """Full state for the fact-check agent graph."""

    # Inputs
    claim_text: str
    image_paths: List[str]
    language: str
    session_id: str
    scenario: str   # "both" | "image_only" | "text_only" — detected early in graph

    # Evidence search results
    image_results: Dict[str, Any]       # filtered image summaries
    claim_results: Dict[str, Any]       # text search results per query

    # Collected sources
    supported_sources: List[str]

    # Final outputs
    final_report_md: str                # final markdown report
    verdict: str

    # Streaming / progress
    current_step: str                   # human-readable step name
    step_output: str                    # partial streaming content for current step
    all_steps: Annotated[List[Dict[str, str]], add]  # accumulated step history

    # Status
    error: Optional[str]
    status: str  # "pending" | "searching" | "synthesizing" | "generating_4w" | "done" | "error"

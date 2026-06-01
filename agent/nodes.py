"""LangGraph node functions for the Fact-Check pipeline.

IMPORTANT: Each node returns a DICT with ONLY the keys it modifies —
not the full AgentState. LangGraph merges partial updates into the state.
This avoids INVALID_CONCURRENT_GRAPH_UPDATE errors when parallel nodes run.
"""

import asyncio
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch

from agent.state import AgentState
from agent.streaming_llm import call_llm, call_llm_stream
from agent.events import emit_event

# ── Existing pipeline imports ──────────────────────────────────────────
from pipeline_extracted__related_inf_for_img import (
    pipeline_extracted_related_information_for_images,
)
from retrieve_related_summary import get_related_summaries_from_all_results
from pipepline_retrieve_related_inf_for_claim import pipepline_retrieve_related_inf_for_claim
from llm.prompt.direct_response_prompt import generate_direct_response_prompt
from setting import scraping_dog_api_key, serp_api_key

# ── Global model references (set externally) ──────────────────────────
_models: Dict[str, Any] = {}
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_models(model, preprocess, device: str = _device):
    """Called once at startup to inject the LongCLIP model."""
    _models["model"] = model
    _models["preprocess"] = preprocess
    _models["device"] = device


# ═══════════════════════════════════════════════════════════════════════
# NODE: search_image_evidence
# ═══════════════════════════════════════════════════════════════════════
async def search_image_evidence(state: AgentState) -> dict:
    """Run the image reverse-search pipeline in a thread pool."""
    await emit_event("step", {
        "node": "search_image_evidence",
        "step": "searching_image",
        "output": "Searching for image sources...",
    })

    update: dict = {
        "current_step": "searching_image",
        "step_output": "Searching for image sources...",
        "status": "searching",
    }

    image_paths = state.get("image_paths", [])
    if not image_paths:
        update["image_results"] = {}
        await emit_event("step", {
            "node": "search_image_evidence",
            "step": "searching_image",
            "output": "No images provided — skipping image search.",
        })
        return update

    await emit_event("step", {
        "node": "search_image_evidence",
        "step": "searching_image",
        "output": f"Processing {len(image_paths)} image(s): uploading, reverse-searching via Google, scraping matched articles, and extracting relevant text...",
    })

    model = _models.get("model")
    preprocess = _models.get("preprocess")
    device = _models.get("device", _device)

    loop = asyncio.get_running_loop()

    def _run():
        raw = pipeline_extracted_related_information_for_images(
            images=image_paths,
            dog_api_key=scraping_dog_api_key,
            serp_api_key=serp_api_key,
            device=device,
            model=model,
            preprocess=preprocess,
            max_urls_per_image=15,
            max_concurrent_images=5,
        )
        return get_related_summaries_from_all_results(raw, max_workers=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        update["image_results"] = await loop.run_in_executor(pool, _run)

    # Emit summary
    total_sources = sum(len(v) for v in (update["image_results"] or {}).values())
    await emit_event("step", {
        "node": "search_image_evidence",
        "step": "searching_image",
        "output": f"✓ Image search complete — found {total_sources} relevant sources across {len(image_paths)} image(s).",
        "done": True,
    })

    return update


# ═══════════════════════════════════════════════════════════════════════
# NODE: search_claim_evidence
# ═══════════════════════════════════════════════════════════════════════
async def search_claim_evidence(state: AgentState) -> dict:
    """Run the text-claim search pipeline in a thread pool."""
    await emit_event("step", {
        "node": "search_claim_evidence",
        "step": "searching_text",
        "output": "Searching for text evidence...",
    })

    update: dict = {
        "current_step": "searching_text",
        "step_output": "Searching for text evidence...",
        "status": "searching",
    }

    claim_text = state.get("claim_text", "").strip()
    if not claim_text:
        update["claim_results"] = {}
        await emit_event("step", {
            "node": "search_claim_evidence",
            "step": "searching_text",
            "output": "No claim text provided — skipping text search.",
        })
        return update

    await emit_event("step", {
        "node": "search_claim_evidence",
        "step": "searching_text",
        "output": f"Generating search queries for claim, then searching Google, scraping top articles, and extracting relevant passages...",
    })

    model = _models.get("model")
    preprocess = _models.get("preprocess")
    device = _models.get("device", _device)

    loop = asyncio.get_running_loop()

    def _run():
        return pipepline_retrieve_related_inf_for_claim(
            claim=claim_text,
            device=device,
            model=model,
            preprocess=preprocess,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        update["claim_results"] = await loop.run_in_executor(pool, _run)

    # Emit summary
    claim_results = update.get("claim_results", {})
    total_articles = sum(len(v) if isinstance(v, list) else 0 for v in (claim_results or {}).values()) if isinstance(claim_results, dict) else len(claim_results) if isinstance(claim_results, list) else 0
    await emit_event("step", {
        "node": "search_claim_evidence",
        "step": "searching_text",
        "output": f"✓ Text search complete — scraped and analyzed {total_articles} articles.",
        "done": True,
    })

    return update


# ═══════════════════════════════════════════════════════════════════════
# NODE: collect_sources
# ═══════════════════════════════════════════════════════════════════════
async def collect_sources(state: AgentState) -> dict:
    """Merge all URLs from image and claim pipelines into supported_sources."""
    sources: set = set()

    for img_path, summaries in state.get("image_results", {}).items():
        for s in summaries:
            if s.get("url"):
                sources.add(s.get("url"))

    claim_results = state.get("claim_results", {})
    if isinstance(claim_results, dict):
        for query, items in claim_results.items():
            for item in items:
                if item.get("url"):
                    sources.add(item.get("url"))
    elif isinstance(claim_results, list):
        for item in claim_results:
            if item.get("url"):
                sources.add(item.get("url"))

    return {
        "current_step": "collecting_sources",
        "step_output": f"Collected {len(list(sources))} unique sources from all evidence.",
        "supported_sources": list(sources),
    }





# ═══════════════════════════════════════════════════════════════════════
# NODE: generate_final_response (direct — skips JSON + 4W + template)
# ═══════════════════════════════════════════════════════════════════════
async def generate_final_response(state: AgentState) -> dict:
    """Single LLM call: evidence → natural-language verdict + explanation + sources.

    Replaces the entire synthesize_claim_summary → synthesize_core →
    generate_4w → build_final_report chain.
    """
    await emit_event("step", {
        "node": "generate_final_response",
        "step": "generating_response",
        "output": "Analyzing evidence and generating response...",
    })

    supported_sources = state.get("supported_sources", [])

    # ── Fallback when no evidence was found ─────────────────────────
    if not supported_sources:
        lang = state.get("language", "en")
        if lang == "vi":
            fallback = (
                "Không thể xác minh được claim này do không tìm thấy nguồn thông tin "
                "liên quan nào. Claim này hiện tại là chưa thể xác minh (unverified).\n\n"
                "**Sources:**\n\nNo external links found."
            )
        else:
            fallback = (
                "This claim cannot be verified because no relevant sources were found. "
                "The claim is currently unverified.\n\n"
                "**Sources:**\n\nNo external links found."
            )
        return {
            "current_step": "generating_response",
            "step_output": fallback,
            "final_report_md": fallback,
            "status": "done",
        }

    system_prompt, user_prompt = generate_direct_response_prompt(
        claim=state.get("claim_text", ""),
        image_results=state.get("image_results", {}),
        claim_results=state.get("claim_results", {}),
        links_list=supported_sources,
        language=state.get("language", "en"),
        scenario=state.get("scenario", "both"),
    )

    full = []
    async for chunk in call_llm_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="deepseek-chat",
        temperature=1.0,
    ):
        full.append(chunk)
        await emit_event("token", {"node": "generate_final_response", "text": chunk})

    llm_output = "".join(full)

    # ── Strip LLM-generated Sources section ──────────────────────────
    clean = re.sub(
        r'\n*Sources?:?\s*\n?(?:\[?\d+\]?\s*https?://[^\s]*\s*)+\s*$',
        '',
        llm_output.strip(),
        flags=re.IGNORECASE,
    )
    clean = re.sub(r'\n*Sources?:?\s*$', '', clean, flags=re.IGNORECASE)
    response = clean.strip()

    # ── Extract verdict and confidence ────────────────────────────────
    scenario = state.get("scenario", "both")
    verdict = _extract_verdict(clean, scenario)
    confidence = _extract_confidence(clean)

    # Save to file (with sources for file persistence)
    session_id = state.get("session_id", uuid.uuid4().hex[:8])
    file_path = f"report/report_{session_id}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        source_lines = "\n".join(
            f"- [{i}] {link}" for i, link in enumerate(supported_sources, start=1)
        )
        f.write(clean.strip() + "\n\n**Sources:**\n\n" + source_lines)

    await emit_event("step", {
        "node": "generate_final_response",
        "step": "generating_response",
        "output": "Response ready.",
        "done": True,
    })

    return {
        "current_step": "generating_response",
        "step_output": response,
        "final_report_md": response,
        "verdict": verdict,
        "confidence": confidence,
        "scenario": scenario,
        "supported_sources": supported_sources,
        "status": "done",
    }


def _extract_verdict(text: str, scenario: str = "both") -> str:
    """Extract verdict from natural-language response.

    Returns one of: REAL, FAKE, MISLEADING, UNVERIFIED, INSUFFICIENT_EVIDENCE
    """
    lowered = text.lower()

    # ── Image-only: default to UNVERIFIED ──────────────────────────
    if scenario == "image_only":
        if any(kw in lowered for kw in (
            "is fake", "is false", "is misleading", "has been debunked",
            "is fabricated", "is manipulated", "is a hoax", "is incorrect",
            "is photoshopped", "is altered",
        )):
            return "FAKE"
        if any(kw in lowered for kw in (
            "is real", "is authentic", "is genuine", "is unaltered",
        )):
            return "REAL"
        return "UNVERIFIED"  # default for image-only

    # ── REAL signals ───────────────────────────────────────────────
    if any(kw in lowered for kw in (
        "is verified", "is confirmed", "is true", "is real",
        "has been verified", "is authentic", "is accurate", "is genuine",
        "claim is correct", "evidence confirms",
    )):
        return "REAL"

    # ── FAKE signals ───────────────────────────────────────────────
    if any(kw in lowered for kw in (
        "is false", "is fake", "has been debunked",
        "is fabricated", "is manufactured", "is a hoax",
        "claim is false", "evidence disproves", "has been falsified",
        "no evidence found", "no credible", "no reliable",
    )):
        return "FAKE"

    # ── MISLEADING signals ─────────────────────────────────────────
    if any(kw in lowered for kw in (
        "is misleading", "taken out of context", "lacks context",
        "is incomplete", "misrepresents",
    )):
        return "MISLEADING"

    # ── INSUFFICIENT_EVIDENCE ──────────────────────────────────────
    if any(kw in lowered for kw in (
        "insufficient evidence", "not enough evidence",
        "cannot be verified", "unable to verify", "cannot be determined",
        "inconclusive", "further investigation needed",
    )):
        return "INSUFFICIENT_EVIDENCE"

    # ── Default: UNVERIFIED ────────────────────────────────────────
    return "UNVERIFIED"


def _extract_confidence(text: str) -> int:
    """Extract a confidence percentage (0-100) from the response text.

    Looks for patterns like: "Confidence: 42%" or "confidence level: 85%"
    """
    import re as _re
    m = _re.search(r'[Cc]onfidence[:\s]+(\d{1,3})\s*%', text)
    if m:
        val = int(m.group(1))
        return max(0, min(100, val))
    # Fallback heuristic: presence of strong evidence words suggests higher confidence
    lowered = text.lower()
    score = 30  # base
    if any(kw in lowered for kw in ("verified", "confirmed", "authentic")):
        score += 30
    if any(kw in lowered for kw in ("evidence", "sources confirm")):
        score += 20
    if any(kw in lowered for kw in ("insufficient", "cannot verify", "unclear", "unknown")):
        score -= 20
    return max(5, min(95, score))

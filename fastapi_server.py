"""FastAPI server for fact-checking agent with SSE streaming.

Endpoints:
  POST /api/analyze   — Submit a fact-check request (multipart: claim text + images)
  GET  /api/stream/{session_id} — SSE stream of step/token events
"""

import asyncio
import json
import os
import uuid
import csv
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import agent.tools as tools_module
import similarity.model.longclip as longclip
from agent.events import register_session, unregister_session, set_current_session, emit_event
from agent.graph import get_graph
from agent.state import AgentState

# ── Globals ────────────────────────────────────────────────────────────
_model_loaded = False


def load_model():
    """Load LongCLIP model once at startup."""
    global _model_loaded
    if _model_loaded:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading LongCLIP on {device.upper()}...")
    model, preprocess = longclip.load(
        "./similarity/checkpoints/longclip-B.pt", device=device
    )
    model.eval()
    tools_module.set_models(model, preprocess, device)
    _model_loaded = True
    print("[+] Model loaded.")


# ── SSE queue helpers ─────────────────────────────────────────────────
_sse_queues: dict[str, asyncio.Queue] = {}


def _get_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _sse_queues:
        _sse_queues[session_id] = asyncio.Queue()
    return _sse_queues[session_id]


def _remove_queue(session_id: str):
    _sse_queues.pop(session_id, None)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    
    # Start Ngrok if configured
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    ngrok_domain = os.environ.get("NGROK_DOMAIN")
    print(f"Debug: token={ngrok_token}, domain={ngrok_domain}")
    
    if ngrok_token and ngrok_domain:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        try:
            public_url = ngrok.connect(8000, domain=ngrok_domain).public_url
            print(f"🚀 Ngrok Tunnel opened at: {public_url}")
        except Exception as e:
            print(f"Ngrok connection failed: {e}")
        
    yield
    
    if ngrok_token and ngrok_domain:
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except:
            pass


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Fact-Check Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# Background pipeline runner
# ═══════════════════════════════════════════════════════════════════════
async def _run_pipeline(session_id: str, claim_text: str, image_paths: List[str], language: str):
    """Execute the ReAct agent and push events into the SSE queue."""
    queue = _get_queue(session_id)
    register_session(session_id, queue)
    set_current_session(session_id)

    try:
        from agent.react_agent import run_react_agent
        
        # Run ReAct loop
        final_report = await run_react_agent(claim_text, image_paths, language)

        # Signal completion
        await queue.put({
            "event": "done",
            "data": {
                "verdict": "", # Removed parsing verdict for simplicity
                "final_report_md": final_report,
                "supported_sources": [], # ReAct agent returns sources natively
                "status": "done",
            },
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        await queue.put({
            "event": "error",
            "data": {"message": str(e), "traceback": tb},
        })
    finally:
        unregister_session(session_id)
        # Keep queue around for a bit so SSE client can drain it
        await asyncio.sleep(2)
        _remove_queue(session_id)


class FeedbackRequest(BaseModel):
    session_id: str
    claim: str
    accuracy: str
    reasoning: str
    sources: str
    image_understanding: str
    comment: str

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Lưu feedback vào file CSV."""
    file_path = "feedback/feedback_logs.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Session ID", "Claim", "Accuracy", "Reasoning", "Sources", "Image Understanding", "Comment"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            feedback.session_id,
            feedback.claim,
            feedback.accuracy,
            feedback.reasoning,
            feedback.sources,
            feedback.image_understanding,
            feedback.comment
        ])
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    claim: str = Form(default=""),
    language: str = Form(default="en"),
    images: List[UploadFile] = File(default=[]),
):
    """Submit a fact-check request.

    Returns a session_id that can be used to connect to the SSE stream.
    """
    session_id = uuid.uuid4().hex[:8]

    # Save uploaded images to temp directory
    image_paths: List[str] = []
    for img in images:
        if img.filename:
            os.makedirs("temp_uploads", exist_ok=True)
            ext = os.path.splitext(img.filename)[1] or ".jpg"
            dest = os.path.join("temp_uploads", f"{session_id}_{uuid.uuid4().hex[:4]}{ext}")
            content = await img.read()
            with open(dest, "wb") as f:
                f.write(content)
            image_paths.append(dest)

    # Launch pipeline in background
    background_tasks.add_task(_run_pipeline, session_id, claim, image_paths, language)

    return {
        "session_id": session_id,
        "stream_url": f"/api/stream/{session_id}",
    }


@app.get("/api/stream/{session_id}")
async def stream_events(session_id: str):
    """SSE endpoint that streams step/token events for a session."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue = _get_queue(session_id)

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                event_type = event.get("event", "message")
                data = event.get("data", {})

                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                if event_type in ("done", "error"):
                    break

            except asyncio.TimeoutError:
                yield f"event: heartbeat\ndata: {json.dumps({})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": _model_loaded}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

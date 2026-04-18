"""
OmniShield – backend/main.py
FastAPI server: REST session management + WebSocket real-time pipeline.

Business logic:
  Each incoming call creates an isolated AgentSession. Audio bytes are piped
  to Deepgram for streaming STT; the resulting transcript lines drive the
  LangGraph state machine in agents.py.  Video frames (Base64 JPEG) are
  forwarded to forensics.py for deepfake + rPPG analysis.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import OmniShieldAgent, AgentState

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("omnishield.main")

# ─── Environment ──────────────────────────────────────────────────────────────

DAILY_API_KEY   = os.getenv("DAILY_API_KEY", "")
DAILY_BASE_URL  = "https://api.daily.co/v1"
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ─── In-memory session registry (swap for Redis in production) ────────────────
# { session_id: AgentState }
_sessions: dict[str, AgentState] = {}


# ─── Lifespan (startup / shutdown hooks) ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OmniShield backend starting up…")
    yield
    logger.info("OmniShield backend shutting down. Active sessions: %d", len(_sessions))
    _sessions.clear()


# ─── App bootstrap ────────────────────────────────────────────────────────────

app = FastAPI(
    title="OmniShield API",
    version="2.0.0",
    description="Agentic AI Video Onboarding for Poonawalla Fincorp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    daily_room_url: str
    daily_room_name: str


class SessionStatusResponse(BaseModel):
    session_id: str
    current_node: str
    applicant_name: str | None
    pan_number: str | None
    income_monthly: float | None
    cibil_score: int | None
    fraud_risk: str | None
    loan_amount: float | None


# ─── REST: session lifecycle ──────────────────────────────────────────────────

@app.post("/api/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session():
    """
    PFL business logic:
    Creates a unique onboarding session.  If a Daily API key is present we
    provision a real ephemeral room (expires in 10 min); otherwise we fall back
    to a static env-var room for local development.
    """
    session_id = str(uuid.uuid4())

    # -- Dynamic Daily.co room provisioning -----------------------------------
    if DAILY_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{DAILY_BASE_URL}/rooms",
                    headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
                    json={
                        "name": f"pfl-{session_id[:8]}",
                        "properties": {
                            "exp": int(asyncio.get_event_loop().time()) + 600,  # 10 min
                            "enable_recording": "cloud",
                            "max_participants": 2,
                        },
                    },
                    timeout=10,
                )
            resp.raise_for_status()
            room = resp.json()
            room_url  = room["url"]
            room_name = room["name"]
        except Exception as exc:
            logger.error("Daily.co room creation failed: %s", exc)
            raise HTTPException(status_code=502, detail="Failed to provision video room")
    else:
        # Local dev fallback
        room_url  = os.getenv("DAILY_ROOM_URL", "https://your-domain.daily.co/dev-room")
        room_name = "dev-room"

    _sessions[session_id] = AgentState(session_id=session_id)
    logger.info("Session created: %s → %s", session_id, room_url)

    return SessionResponse(
        session_id=session_id,
        daily_room_url=room_url,
        daily_room_name=room_name,
    )


@app.get("/api/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(session_id: str):
    """Returns the current state of an onboarding session (audit / dashboard)."""
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStatusResponse(
        session_id=state.session_id,
        current_node=state.current_node,
        applicant_name=state.applicant_name,
        pan_number=state.pan_number,
        income_monthly=state.income_monthly,
        cibil_score=state.cibil_score,
        fraud_risk=state.fraud_risk,
        loan_amount=state.loan_amount,
    )


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_session(session_id: str):
    """Explicitly closes and removes a session (call on hangup)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    logger.info("Session closed: %s", session_id)


@app.get("/healthz")
async def health():
    return JSONResponse({"status": "ok", "sessions": len(_sessions)})


# ─── WebSocket: real-time pipeline ───────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Bidirectional WebSocket channel per session.

    Inbound message types (client → server):
      bytes                  → raw PCM audio, forwarded to Deepgram
      { type: TRANSCRIPT_LINE, text }      → STT result (or browser fallback)
      { type: VIDEO_FRAME,    frame_b64 }  → JPEG for deepfake / rPPG
      { type: PING }                       → keepalive

    Outbound event types (server → client):
      AI_SPEECH              → { text }             Priya's spoken response
      FIELD_COLLECTED        → { field, value }      KYC field extracted
      VERIFICATION_STARTED   → {}
      VERIFICATION_COMPLETE  → { cibil_score, fraud_risk, is_deepfake, heart_rate_bpm }
      OFFER_GENERATED        → { loan_amount, interest_rate, … }
      APPLICATION_DECLINED   → {}
      FRAUD_DETECTED         → {}
      ERROR                  → { message }
      PONG                   → {}
    """
    await websocket.accept()
    logger.info("WS connected: %s", session_id)

    state = _sessions.get(session_id)
    if not state:
        await websocket.send_json({"type": "ERROR", "message": "Session not found"})
        await websocket.close(code=4004)
        return

    agent = OmniShieldAgent(state=state, ws=websocket)

    # Kick off the greeting node immediately
    await agent.start()

    try:
        while True:
            raw: Any = await websocket.receive()

            # ── Binary frame: raw PCM audio ───────────────────────────────
            if "bytes" in raw and raw["bytes"]:
                await agent.handle_audio_chunk(raw["bytes"])

            # ── Text frame: JSON control message ──────────────────────────
            elif "text" in raw and raw["text"]:
                try:
                    msg: dict = json.loads(raw["text"])
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON text frame, ignoring")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "TRANSCRIPT_LINE":
                    text = (msg.get("text") or "").strip()
                    if text:
                        await agent.handle_transcript(text)

                elif msg_type == "VIDEO_FRAME":
                    await agent.handle_video_frame(msg.get("frame_b64", ""))

                elif msg_type == "PING":
                    await websocket.send_json({"type": "PONG"})

                else:
                    logger.debug("Unknown message type: %s", msg_type)

    except WebSocketDisconnect:
        logger.info("WS disconnected: %s (node=%s)", session_id, state.current_node)
    except Exception as exc:
        logger.exception("WS error for session %s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "ERROR", "message": str(exc)})
        except Exception:
            pass

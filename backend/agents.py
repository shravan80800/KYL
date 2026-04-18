"""
OmniShield – backend/agents.py
LangGraph state machine + LLM orchestration + parallel verification.

State flow:
  greeting → data_collection ──(loop until complete)──┐
                                                       ↓
                                              risk_verification
                                            (asyncio.gather PFL Win)
                                                       ↓
                                                  loan_offer → END
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import WebSocket
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from services.forensics import run_deepfake_shield, run_rppg_estimation

load_dotenv()
logger = logging.getLogger("omnishield.agents")

# ─── Bureau data loader ───────────────────────────────────────────────────────

def _load_bureau_db() -> dict[str, dict]:
    """
    Loads mock bureau profiles from mock_bureau.json.
    In production, replace with an async CIBIL/Experian API call.
    """
    import pathlib
    path = pathlib.Path(__file__).parent / "mock_bureau.json"
    with open(path) as f:
        raw = json.load(f)
    # Key by uppercase name for fast lookup
    return {p["name"].upper(): p for p in raw["profiles"]}


_BUREAU_DB: dict[str, dict] = _load_bureau_db()


# ─── AgentState dataclass ─────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Immutable-style session state carried through every LangGraph node.
    One instance lives per WebSocket session.
    """
    session_id: str

    # ── LangGraph position ────────────────────────────────────────────────
    current_node: str = "greeting"

    # ── Conversation history (last N turns fed to LLM) ───────────────────
    transcript_history: list[str] = field(default_factory=list)

    # ── KYC fields extracted by Gemini ───────────────────────────────────
    applicant_name:  Optional[str]   = None
    pan_number:      Optional[str]   = None
    income_monthly:  Optional[float] = None

    # ── Verification outputs ──────────────────────────────────────────────
    cibil_score:   Optional[int]   = None
    fraud_risk:    Optional[str]   = None   # "LOW" | "MEDIUM" | "HIGH"
    is_deepfake:   Optional[bool]  = None
    heart_rate_bpm: Optional[int]  = None

    # ── Loan offer ────────────────────────────────────────────────────────
    loan_amount:     Optional[float] = None
    interest_rate:   Optional[float] = None
    tenure_months:   Optional[int]   = None

    # ── Control flags ─────────────────────────────────────────────────────
    greeted:           bool = False
    data_complete:     bool = False
    verification_done: bool = False
    offer_made:        bool = False


# ─── Mock bureau verification ─────────────────────────────────────────────────

async def verify_bureau(pan_number: str, name: str) -> dict[str, Any]:
    """
    PFL Business Logic:
    Simulates a CIBIL + CRIF bureau lookup with realistic 1-2 s latency.
    Keyed by applicant name (UPPER) from mock_bureau.json.

    In production: replace with:
      async with httpx.AsyncClient() as c:
          resp = await c.post("https://api.cibil.com/v2/score", json={...})
    """
    await asyncio.sleep(1.2)  # simulate network round-trip
    profile = _BUREAU_DB.get(name.upper(), {
        "cibil_score": 680,
        "fraud_risk":  "LOW",
        "eligible":    True,
    })
    logger.info("Bureau result for '%s': CIBIL=%s", name, profile.get("cibil_score"))
    return {
        "cibil_score": profile["cibil_score"],
        "fraud_risk":  profile["fraud_risk"],
        "eligible":    profile.get("eligible", True),
        "pan":         pan_number,
    }


# ─── Loan offer engine ────────────────────────────────────────────────────────

def compute_loan_offer(cibil: int, income_monthly: float) -> dict[str, Any]:
    """
    PFL tiered underwriting logic.
    Tier thresholds are illustrative; real PFL uses a proprietary scorecard.
    """
    annual_income = income_monthly * 12

    if cibil >= 750:
        # Premium tier: 60× monthly income, best rate
        max_foir        = 0.60   # Fixed Obligation to Income Ratio
        rate            = 10.5
        tenure          = 60
        income_multiple = 60
    elif cibil >= 650:
        max_foir        = 0.50
        rate            = 13.0
        tenure          = 48
        income_multiple = 40
    elif cibil >= 550:
        max_foir        = 0.40
        rate            = 16.5
        tenure          = 36
        income_multiple = 20
    else:
        return {"eligible": False}

    # Cap at ₹50L or income multiple
    loan_amount = min(income_monthly * income_multiple, 5_000_000)
    # Round to nearest ₹1,000
    loan_amount = round(loan_amount / 1_000) * 1_000

    # Approximate EMI
    monthly_rate = (rate / 100) / 12
    emi = loan_amount * monthly_rate * (1 + monthly_rate) ** tenure / (
        (1 + monthly_rate) ** tenure - 1
    )
    foir = emi / income_monthly

    # Reject if EMI burden exceeds threshold
    if foir > max_foir:
        loan_amount = int(income_monthly * max_foir / monthly_rate *
                          (1 - (1 + monthly_rate) ** -tenure))
        loan_amount = round(loan_amount / 1_000) * 1_000

    return {
        "eligible":       True,
        "loan_amount":    loan_amount,
        "interest_rate":  rate,
        "tenure_months":  tenure,
        "emi_approx":     round(emi),
    }


# ─── LLM helpers ─────────────────────────────────────────────────────────────

_PRIYA_SYSTEM_PROMPT = """\
You are Priya, a warm, professional, and empathetic AI loan officer at Poonawalla
Fincorp (PFL), one of India's leading NBFCs.

### Persona rules:
- Speak naturally, like a trusted bank relationship manager — never robotic.
- Keep responses to 2-3 sentences maximum.  Be concise.
- Address the applicant by first name once you know it.
- If an applicant seems anxious, offer brief reassurance before moving forward.
- Never ask for more than one piece of information at a time.
- Do NOT reveal internal state names (greeting, data_collection, etc.) to the user.
- Always maintain the warmth of a face-to-face interaction.
- Use English mixed with occasional Hindi words (namaste, bilkul, shukriya) 
  to match PFL's brand voice.

### Data you are collecting (in order):
1. Full legal name
2. PAN card number (format: AAAAA9999A)
3. Monthly take-home income (in INR)

### Never do:
- Ask for Aadhaar number (only collect PAN).
- Make definitive loan promises before verification completes.
- Repeat the same question more than twice.
"""

_EXTRACTION_SYSTEM_PROMPT = """\
You are a JSON field extractor for a financial KYC pipeline.
Your ONLY output must be a single valid JSON object — no markdown fences,
no explanation, no preamble.  Use null for any field not clearly stated.

Extract these fields from the conversation:
  applicant_name  : string | null   (full name as stated)
  pan_number      : string | null   (exactly 10 chars: AAAAA9999A, uppercase)
  income_monthly  : number | null   (monthly take-home in INR, numeric only)
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a Gemini 1.5 Pro model.
    Requires GOOGLE_API_KEY in environment.
    Gemini is chosen for its strong multilingual ability (Hindi/English mix)
    and long context window for conversation history.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.35,
        max_tokens=512,
        convert_system_message_to_human=True,  # required for Gemini via LangChain
    )


async def llm_priya_respond(state: AgentState, user_message: str) -> str:
    """Generates Priya's next conversational response."""
    llm = _get_llm()

    # Feed the last 6 turns (3 exchange pairs) to keep context focused
    history_text = "\n".join(state.transcript_history[-6:]) or "(call just started)"

    messages = [
        SystemMessage(content=_PRIYA_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Conversation so far:
{history_text}

Applicant just said: "{user_message}"

Current collection status:
- Name     : {state.applicant_name or 'NOT YET COLLECTED'}
- PAN      : {state.pan_number or 'NOT YET COLLECTED'}
- Income   : {f'₹{state.income_monthly:,.0f}/month' if state.income_monthly else 'NOT YET COLLECTED'}

Respond as Priya.  If all three fields are collected, say you are running a
quick background check and ask the applicant to stay on the line.
"""),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


async def llm_extract_fields(state: AgentState) -> dict[str, Any]:
    """
    Uses Gemini to extract structured KYC fields from the last 4 transcript lines.
    Returns a dict with keys: applicant_name, pan_number, income_monthly.
    """
    llm = _get_llm()
    recent = "\n".join(state.transcript_history[-4:]) or ""

    messages = [
        SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation excerpt:\n{recent}"),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Extraction JSON parse failed. Raw: %r", raw)
        return {"applicant_name": None, "pan_number": None, "income_monthly": None}


# ─── OmniShieldAgent (orchestrates LangGraph + WebSocket) ────────────────────

class OmniShieldAgent:
    """
    Wraps the LangGraph state machine and exposes simple async entry-points
    that the WebSocket handler calls directly.

    The LangGraph graph is compiled once per agent instance.  Each node
    method mutates `self.state` and sends WS events; the graph itself is
    used mainly to formalise routing logic and make the flow auditable.
    """

    def __init__(self, state: AgentState, ws: WebSocket) -> None:
        self.state  = state
        self.ws     = ws
        self._latest_frame_b64: str = ""
        self._graph = self._build_graph()

    # ── Graph definition ─────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(dict)

        graph.add_node("greeting",          self._node_greeting)
        graph.add_node("data_collection",   self._node_data_collection)
        graph.add_node("risk_verification", self._node_risk_verification)
        graph.add_node("loan_offer",        self._node_loan_offer)

        graph.set_entry_point("greeting")

        graph.add_conditional_edges(
            "greeting",
            lambda _: "data_collection" if self.state.greeted else "greeting",
        )
        graph.add_conditional_edges(
            "data_collection",
            lambda _: "risk_verification" if self.state.data_complete else "data_collection",
        )
        graph.add_edge("risk_verification", "loan_offer")
        graph.add_edge("loan_offer", END)

        return graph.compile()

    # ── Node: greeting ───────────────────────────────────────────────────────

    async def _node_greeting(self, _: dict) -> dict:
        """
        PFL brand-aligned greeting.  Sets the tone for the whole call.
        """
        opening = (
            "Namaste! I'm Priya, your Poonawalla Fincorp loan officer. "
            "This quick video check will take less than two minutes and "
            "everything you share is fully encrypted.  Could you start by "
            "telling me your full name?"
        )
        await self._speak(opening)
        self.state.greeted      = True
        self.state.current_node = "data_collection"
        return {}

    # ── Node: data_collection ────────────────────────────────────────────────

    async def _node_data_collection(self, payload: dict) -> dict:
        """
        Extracts KYC fields from free-form speech using Gemini.
        Loops until name + PAN + income are all confirmed.
        """
        transcript = (payload.get("transcript") or "").strip()
        if not transcript:
            return {}

        self.state.transcript_history.append(f"Applicant: {transcript}")

        # ── Field extraction ─────────────────────────────────────────────────
        extracted = await llm_extract_fields(self.state)
        logger.info("Extracted fields: %s", extracted)

        # Name
        if extracted.get("applicant_name") and not self.state.applicant_name:
            self.state.applicant_name = str(extracted["applicant_name"]).strip().title()
            await self._emit("FIELD_COLLECTED", {
                "field": "name",
                "value": self.state.applicant_name,
            })
            logger.info("Name captured: %s", self.state.applicant_name)

        # PAN — validate format strictly (AAAAA9999A)
        if extracted.get("pan_number") and not self.state.pan_number:
            pan_raw = str(extracted["pan_number"]).upper().replace(" ", "")
            if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan_raw):
                self.state.pan_number = pan_raw
                await self._emit("FIELD_COLLECTED", {
                    "field": "pan",
                    "value": pan_raw,
                })
                logger.info("PAN captured: %s", pan_raw)
            else:
                logger.warning("Invalid PAN format received: %r", pan_raw)

        # Income — must be a positive number
        if extracted.get("income_monthly") and not self.state.income_monthly:
            try:
                income = float(str(extracted["income_monthly"]).replace(",", ""))
                if income > 0:
                    self.state.income_monthly = income
                    await self._emit("FIELD_COLLECTED", {
                        "field": "income",
                        "value": income,
                    })
                    logger.info("Income captured: ₹%.0f/month", income)
            except (ValueError, TypeError):
                pass

        # ── Check completeness ───────────────────────────────────────────────
        if (
            self.state.applicant_name
            and self.state.pan_number
            and self.state.income_monthly
        ):
            self.state.data_complete = True
            self.state.current_node  = "risk_verification"
            reply = (
                f"Bilkul, {self.state.applicant_name.split()[0]}!  "
                "I have everything I need. Give me just a moment — "
                "I'm running a quick background check while we talk."
            )
        else:
            # Ask Priya to naturally probe for the next missing field
            reply = await llm_priya_respond(self.state, transcript)

        self.state.transcript_history.append(f"Priya: {reply}")
        await self._speak(reply)
        return {}

    # ── Node: risk_verification ──────────────────────────────────────────────

    async def _node_risk_verification(self, _: dict) -> dict:
        """
        ★ The PFL Win ★
        Bureau check, deepfake shield, and rPPG heart-rate estimation all run
        SIMULTANEOUSLY using asyncio.gather.  This means zero extra waiting
        time for the applicant — verification is complete by the time Priya
        finishes her transition sentence.
        """
        self.state.current_node = "risk_verification"
        await self._emit("VERIFICATION_STARTED", {})
        logger.info("Starting parallel verification for session %s", self.state.session_id)

        # ── asyncio.gather: the core PFL competitive advantage ───────────────
        bureau_result, deepfake_result, rppg_result = await asyncio.gather(
            verify_bureau(self.state.pan_number or "", self.state.applicant_name or ""),
            run_deepfake_shield(self._latest_frame_b64),
            run_rppg_estimation(self._latest_frame_b64),
        )

        logger.info("Bureau: %s | Deepfake: %s | rPPG: %s",
                    bureau_result, deepfake_result, rppg_result)

        # Write results into shared state
        self.state.cibil_score    = bureau_result["cibil_score"]
        self.state.fraud_risk     = bureau_result["fraud_risk"]
        self.state.is_deepfake    = deepfake_result["is_deepfake"]
        self.state.heart_rate_bpm = rppg_result["heart_rate_bpm"]

        self.state.verification_done = True
        self.state.current_node      = "loan_offer"

        await self._emit("VERIFICATION_COMPLETE", {
            "cibil_score":    self.state.cibil_score,
            "fraud_risk":     self.state.fraud_risk,
            "is_deepfake":    self.state.is_deepfake,
            "heart_rate_bpm": self.state.heart_rate_bpm,
            "deepfake_conf":  deepfake_result.get("confidence", 0.0),
            "rppg_quality":   rppg_result.get("signal_quality", "UNKNOWN"),
        })

        return {}

    # ── Node: loan_offer ─────────────────────────────────────────────────────

    async def _node_loan_offer(self, _: dict) -> dict:
        """
        Generates a personalised loan offer or a graceful decline.
        PFL guardrails: reject if deepfake detected OR CIBIL < 400 OR HIGH fraud risk.
        """
        self.state.current_node = "loan_offer"

        # ── Hard rejection: identity fraud ───────────────────────────────────
        if self.state.is_deepfake:
            await self._speak(
                "I'm sorry, we encountered an issue verifying your identity "
                "through our security system.  Please visit your nearest PFL "
                "branch or call 1800-XXX-XXXX for assistance.  Shukriya."
            )
            await self._emit("FRAUD_DETECTED", {})
            return {}

        # ── Hard rejection: high fraud risk ──────────────────────────────────
        if self.state.fraud_risk == "HIGH" or (self.state.cibil_score or 0) < 400:
            await self._speak(
                f"Thank you for your time, {(self.state.applicant_name or '').split()[0]}. "
                "Based on our current credit policy we are unable to proceed with a "
                "pre-approval at this moment.  We will reach out if this changes."
            )
            await self._emit("APPLICATION_DECLINED", {})
            return {}

        # ── Compute personalised offer ────────────────────────────────────────
        offer = compute_loan_offer(
            cibil=self.state.cibil_score or 600,
            income_monthly=self.state.income_monthly or 30_000,
        )

        if not offer.get("eligible"):
            await self._speak(
                "I'm sorry, we're unable to generate a pre-approved offer based "
                "on the current information.  A PFL relationship manager will "
                "contact you within 24 hours to explore other options."
            )
            await self._emit("APPLICATION_DECLINED", {})
            return {}

        self.state.loan_amount   = offer["loan_amount"]
        self.state.interest_rate = offer["interest_rate"]
        self.state.tenure_months = offer["tenure_months"]
        self.state.offer_made    = True

        await self._emit("OFFER_GENERATED", {
            "applicant_name":  self.state.applicant_name,
            "loan_amount":     self.state.loan_amount,
            "interest_rate":   self.state.interest_rate,
            "tenure_months":   self.state.tenure_months,
            "emi_approx":      offer.get("emi_approx"),
            "cibil_score":     self.state.cibil_score,
        })

        lakhs = round((self.state.loan_amount or 0) / 100_000, 1)
        name_first = (self.state.applicant_name or "").split()[0]
        await self._speak(
            f"Congratulations, {name_first}!  Your personalised offer is ready — "
            f"you're pre-approved for ₹{lakhs} lakh at {self.state.interest_rate}% "
            f"per annum for {self.state.tenure_months} months.  "
            "Please review the offer card on your screen and let me know if you'd "
            "like to proceed!"
        )
        return {}

    # ── Public entry points (called by the WebSocket handler) ────────────────

    async def start(self) -> None:
        """Runs the greeting node immediately on WS connect."""
        await self._graph.ainvoke({})

    async def handle_transcript(self, transcript: str) -> None:
        """Routes a new STT transcript line through the active node."""
        if self.state.current_node == "data_collection":
            await self._node_data_collection({"transcript": transcript})
            # If data just became complete, advance automatically
            if self.state.data_complete and not self.state.verification_done:
                await self._node_risk_verification({})
                await self._node_loan_offer({})

    async def handle_audio_chunk(self, audio_bytes: bytes) -> None:
        """
        Receives raw PCM from the frontend.
        In production, pipe to Deepgram streaming SDK:

            from deepgram import DeepgramClient
            dg  = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
            conn = await dg.listen.asynclive.v("1").start(
                        LiveOptions(model="nova-2", language="en-IN", punctuate=True))
            conn.on(LiveTranscriptionEvents.Transcript, self._on_deepgram_result)
            await conn.send(audio_bytes)
        """
        logger.debug("Audio chunk: %d bytes", len(audio_bytes))

    async def handle_video_frame(self, frame_b64: str) -> None:
        """Caches the latest Base64 JPEG frame for forensic analysis."""
        if frame_b64:
            self._latest_frame_b64 = frame_b64

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """Sends Priya's spoken text to the frontend TTS renderer."""
        await self.ws.send_json({"type": "AI_SPEECH", "text": text})

    async def _emit(self, event_type: str, data: dict) -> None:
        """Sends a structured UI update event to the React frontend."""
        await self.ws.send_json({"type": event_type, "data": data})

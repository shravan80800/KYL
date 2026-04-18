# KYL — Full Setup & Wiring Guide

## Project layout

```
omnishield_v2/
├── backend/
│   ├── main.py                ← FastAPI server + WebSocket pipeline
│   ├── agents.py              ← LangGraph state machine + Gemini orchestration
│   ├── requirements.txt
│   ├── mock_bureau.json       ← Test profiles (Shravan=800, Fraudster=280)
│   └── services/
│       └── forensics.py       ← rPPG + deepfake mock (swap for real models)
├── frontend/
│   ├── VideoCall.tsx          ← Primary Next.js component
│   └── package.json
└── .env.example               ← All required keys documented
```

---

## 1. Backend — local dev

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp ../.env.example .env
# Edit .env: add GOOGLE_API_KEY and DAILY_ROOM_URL at minimum

uvicorn main:app --reload --port 8000
```

Health check: `curl http://localhost:8000/healthz`

---

## 2. Frontend — local dev

```bash
cd frontend
npm install

cp ../.env.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
# NEXT_PUBLIC_WS_URL=ws://localhost:8000

npm run dev
```

Open: http://localhost:3000

---

## 3. Wire: Daily.co video

**Option A — Static room (quickest for hackathon):**
1. Go to https://dashboard.daily.co/ → Create room
2. Set `DAILY_ROOM_URL=https://your-domain.daily.co/room-name` in `backend/.env`
3. Leave `DAILY_API_KEY` empty → backend uses the static URL

**Option B — Dynamic rooms (production):**
1. Set `DAILY_API_KEY` in `backend/.env`
2. Each `POST /api/sessions` call now provisions a fresh ephemeral room (10 min TTL)

---

## 4. Wire: Deepgram speech-to-text

The frontend ships with a Web Speech API fallback. To wire real Deepgram:

```python
# In agents.py → handle_audio_chunk():
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

dg_client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])

async def _start_deepgram(self):
    conn = await dg_client.listen.asynclive.v("1").start(
        LiveOptions(
            model="nova-2",
            language="en-IN",
            punctuate=True,
            smart_format=True,
        )
    )
    async def on_transcript(self, result, **kw):
        sentence = result.channel.alternatives[0].transcript
        if result.is_final and sentence:
            await self.handle_transcript(sentence)
    conn.on(LiveTranscriptionEvents.Transcript, on_transcript)
    self._dg_conn = conn

async def handle_audio_chunk(self, audio_bytes: bytes):
    if not hasattr(self, "_dg_conn"):
        await self._start_deepgram()
    await self._dg_conn.send(audio_bytes)
```

The frontend captures audio via Daily.co's `callFrame.getInputDevices()` +
the browser MediaStream API, then pipes raw PCM chunks over the WebSocket as
`ArrayBuffer` binary frames.

---

## 5. Wire: Gemini intelligence

Set `GOOGLE_API_KEY` in `backend/.env`. No other changes needed — `agents.py`
already calls `ChatGoogleGenerativeAI(model="gemini-1.5-pro")`.

To switch models:
```python
# In agents.py → _get_llm():
model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
return ChatGoogleGenerativeAI(model=model, …)
```

---

## 6. Wire: forensic video frames

The frontend `captureFrame()` function (in VideoCall.tsx) already:
1. Grabs a `<video>` element from the Daily.co iframe
2. Draws it onto an off-screen `<canvas>` at 320×240
3. Encodes it as JPEG (quality 0.55)
4. Sends `{ type: "VIDEO_FRAME", frame_b64: "…" }` every 3 seconds

The backend caches this in `agent._latest_frame_b64` and passes it to both
`run_deepfake_shield()` and `run_rppg_estimation()` during `risk_verification`.

**To wire real models** (replace `services/forensics.py`):
- Deepfake: load a TorchScript binary classifier, decode the base64 to PIL Image, run inference
- rPPG: buffer 30 frames, extract mean RGB per frame, apply CHROM algorithm, FFT peak → HR

---

## 7. LangGraph state flow

```
[WS connect]
      │
  greeting  →  "Namaste! I'm Priya…"
      │
  data_collection  ←── loops until name + PAN + income captured
      │
  risk_verification  ───── asyncio.gather ─────┐
      │                                         ├── verify_bureau()
      │                                         ├── run_deepfake_shield()
      │                                         └── run_rppg_estimation()
      │
  loan_offer  →  OFFER_GENERATED | APPLICATION_DECLINED | FRAUD_DETECTED
```

---

## 8. WebSocket event protocol

| Direction | Event | Payload |
|---|---|---|
| S→C | `AI_SPEECH` | `{ text }` |
| S→C | `FIELD_COLLECTED` | `{ field, value }` |
| S→C | `VERIFICATION_STARTED` | `{}` |
| S→C | `VERIFICATION_COMPLETE` | `{ cibil_score, fraud_risk, is_deepfake, heart_rate_bpm, deepfake_conf, rppg_quality }` |
| S→C | `OFFER_GENERATED` | `{ loan_amount, interest_rate, tenure_months, emi_approx, cibil_score }` |
| S→C | `APPLICATION_DECLINED` | `{}` |
| S→C | `FRAUD_DETECTED` | `{}` |
| S→C | `ERROR` | `{ message }` |
| C→S | `TRANSCRIPT_LINE` | `{ text }` |
| C→S | `VIDEO_FRAME` | `{ frame_b64 }` |
| C→S | `PING` | `{}` |

---

## 9. Mock bureau quick reference

| Name | CIBIL | Risk | Outcome |
|---|---|---|---|
| Shravan | 810 | LOW | Premium offer |
| Priya | 765 | LOW | Gold offer |
| Ananya | 655 | MEDIUM | Standard offer |
| Deepak | 590 | LOW | Sub-prime offer |
| Kiran | 520 | LOW | Minimal offer |
| Fraudster | 280 | HIGH | Hard reject |

---

## 10. Deployment checklist

- [ ] Set all env vars in production (no `.env` file — use secret manager)
- [ ] Replace `ws://` with `wss://` and `http://` with `https://` in frontend `.env`
- [ ] Swap `_sessions` dict for Redis (`redis-py` + `aioredis`)
- [ ] Enable Daily.co cloud recording in room properties
- [ ] Add rate limiting (slowapi) to `/api/sessions`
- [ ] Wire Deepgram SDK in `handle_audio_chunk`
- [ ] Replace forensics mocks with real model inference
- [ ] Configure Sentry for backend error tracking

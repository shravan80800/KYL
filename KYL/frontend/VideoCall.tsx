"use client";

/**
 * OmniShield – frontend/VideoCall.tsx
 *
 * Primary onboarding UI.  Layout:
 *   ┌─────────────────────────────────┬──────────────────┐
 *   │  Daily.co video frame           │  AI Insights     │
 *   │  + Priya speech bubble          │  (AR panel)      │
 *   └─────────────────────────────────┴──────────────────┘
 *
 * WebSocket event flow (server → client):
 *   AI_SPEECH            → speech bubble update
 *   FIELD_COLLECTED      → KYC field card animates in
 *   VERIFICATION_STARTED → progress bars animate
 *   VERIFICATION_COMPLETE→ verification result card
 *   OFFER_GENERATED      → loan offer card springs in
 *   APPLICATION_DECLINED → decline message
 *   FRAUD_DETECTED       → security alert
 */

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";

// ─── Type definitions ─────────────────────────────────────────────────────────

interface LoanOffer {
  applicant_name:  string;
  loan_amount:     number;
  interest_rate:   number;
  tenure_months:   number;
  emi_approx:      number;
  cibil_score:     number;
}

interface VerificationResult {
  cibil_score:    number;
  fraud_risk:     "LOW" | "MEDIUM" | "HIGH";
  is_deepfake:    boolean;
  heart_rate_bpm: number;
  deepfake_conf:  number;
  rppg_quality:   "EXCELLENT" | "GOOD" | "POOR";
}

type KYCField = "name" | "pan" | "income";
type CollectedFields = Partial<Record<KYCField, string | number>>;

type AgentStatus =
  | "init"
  | "connecting"
  | "greeting"
  | "collecting"
  | "verifying"
  | "offer"
  | "declined"
  | "fraud"
  | "error";

// ─── Environment ──────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL  || "http://localhost:8000";
const WS_BASE  = process.env.NEXT_PUBLIC_WS_URL   || "ws://localhost:8000";
const FRAME_INTERVAL_MS = 3_000; // capture one frame every 3 s for forensics

// ─── Formatting helpers ───────────────────────────────────────────────────────

function fmtINR(n: number): string {
  if (n >= 10_00_000) return `₹${(n / 10_00_000).toFixed(1)}Cr`;
  if (n >= 1_00_000)  return `₹${(n / 1_00_000).toFixed(1)}L`;
  return `₹${n.toLocaleString("en-IN")}`;
}

function fmtFieldLabel(f: KYCField): string {
  return { name: "Full name", pan: "PAN number", income: "Monthly income" }[f];
}

function fmtFieldValue(f: KYCField, v: string | number): string {
  if (f === "income") return fmtINR(Number(v)) + "/month";
  return String(v);
}

const STATUS_LABELS: Record<AgentStatus, string> = {
  init:       "Initialising…",
  connecting: "Connecting…",
  greeting:   "Priya is ready",
  collecting: "Gathering details",
  verifying:  "Verifying in parallel…",
  offer:      "Offer ready!",
  declined:   "Application reviewed",
  fraud:      "Identity check failed",
  error:      "Connection error",
};

// ─── Animation variants ───────────────────────────────────────────────────────

const fadeSlide = {
  hidden:  { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35 } },
  exit:    { opacity: 0, y: -8, transition: { duration: 0.2 } },
};

const cardSpring = {
  hidden:  { opacity: 0, scale: 0.88, y: 24 },
  visible: { opacity: 1, scale: 1, y: 0,
             transition: { type: "spring", stiffness: 280, damping: 22 } },
  exit:    { opacity: 0, scale: 0.92 },
};

// ─── Main component ───────────────────────────────────────────────────────────

export default function VideoCall() {
  // Session
  const [sessionId,  setSessionId]  = useState<string | null>(null);
  const [roomUrl,    setRoomUrl]     = useState<string | null>(null);
  const [status,     setStatus]      = useState<AgentStatus>("init");

  // Conversation
  const [speech,     setSpeech]      = useState("Initialising OmniShield…");
  const [transcript, setTranscript]  = useState<string[]>([]);

  // KYC
  const [fields,     setFields]      = useState<CollectedFields>({});

  // Verification
  const [verification, setVerification] = useState<VerificationResult | null>(null);

  // Offer
  const [offer, setOffer] = useState<LoanOffer | null>(null);

  // Refs
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const wsRef             = useRef<WebSocket | null>(null);
  const canvasRef         = useRef<HTMLCanvasElement | null>(null);

  // ── Step 1: create session ──────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    setStatus("connecting");

    (async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/sessions`, { method: "POST" });
        if (!res.ok) throw new Error(`Session API returned ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setSessionId(data.session_id);
        setRoomUrl(data.daily_room_url);
      } catch (err) {
        console.error("Session creation failed:", err);
        if (!cancelled) setStatus("error");
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // ── Step 2: mount Daily.co iframe ───────────────────────────────────────────

  useEffect(() => {
    if (!roomUrl || !videoContainerRef.current) return;

    // Dynamically import daily-js to avoid SSR issues
    import("@daily-co/daily-js").then(({ default: DailyIframe }) => {
      const frame = DailyIframe.createFrame(videoContainerRef.current!, {
        showLeaveButton:      false,
        showFullscreenButton: false,
        iframeStyle: {
          width:        "100%",
          height:       "100%",
          border:       "none",
          borderRadius: "12px",
        },
      });
      frame.join({ url: roomUrl });
      return () => frame.destroy();
    });
  }, [roomUrl]);

  // ── Step 3: open WebSocket ──────────────────────────────────────────────────

  useEffect(() => {
    if (!sessionId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen  = () => setStatus("greeting");
    ws.onclose = () => console.info("WS closed");
    ws.onerror = () => setStatus("error");

    ws.onmessage = (ev: MessageEvent) => {
      try {
        handleServerEvent(JSON.parse(ev.data));
      } catch {
        console.warn("Non-JSON WS message:", ev.data);
      }
    };

    // Keepalive ping every 25 s
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ type: "PING" }));
    }, 25_000);

    return () => { clearInterval(ping); ws.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // ── Step 4: browser speech recognition (Deepgram fallback) ──────────────────

  useEffect(() => {
    const SR = (window as any).SpeechRecognition
            || (window as any).webkitSpeechRecognition;
    if (!SR) {
      console.warn("SpeechRecognition API unavailable — using typed input only");
      return;
    }

    const rec: SpeechRecognition = new SR();
    rec.continuous      = true;
    rec.interimResults  = true;
    rec.lang            = "en-IN";

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      let final = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        if (ev.results[i].isFinal) final += ev.results[i][0].transcript;
      }
      if (!final) return;
      setTranscript(prev => [...prev.slice(-9), final]);  // keep last 10 lines
      sendToAgent({ type: "TRANSCRIPT_LINE", text: final });
    };

    rec.onerror = (e: SpeechRecognitionErrorEvent) =>
      console.error("SpeechRecognition error:", e.error);

    rec.start();
    return () => rec.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // ── Step 5: periodic frame capture for forensics ─────────────────────────────

  const captureFrame = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const video = videoContainerRef.current?.querySelector("video");
    if (!video || video.readyState < 2) return;   // HAVE_CURRENT_DATA

    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;
    canvas.width  = 320;
    canvas.height = 240;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, 320, 240);
    const frame_b64 = canvas.toDataURL("image/jpeg", 0.55).split(",")[1];
    sendToAgent({ type: "VIDEO_FRAME", frame_b64 });
  }, []);

  useEffect(() => {
    const id = setInterval(captureFrame, FRAME_INTERVAL_MS);
    return () => clearInterval(id);
  }, [captureFrame]);

  // ── Event handlers ──────────────────────────────────────────────────────────

  function sendToAgent(payload: object) {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify(payload));
  }

  function handleServerEvent(msg: { type: string; text?: string; data?: any }) {
    switch (msg.type) {

      case "AI_SPEECH":
        setSpeech(msg.text || "");
        break;

      case "FIELD_COLLECTED":
        setStatus("collecting");
        setFields(prev => ({ ...prev, [msg.data.field]: msg.data.value }));
        break;

      case "VERIFICATION_STARTED":
        setStatus("verifying");
        break;

      case "VERIFICATION_COMPLETE":
        setVerification(msg.data);
        break;

      case "OFFER_GENERATED":
        setStatus("offer");
        setOffer(msg.data);
        break;

      case "APPLICATION_DECLINED":
        setStatus("declined");
        break;

      case "FRAUD_DETECTED":
        setStatus("fraud");
        setSpeech("Identity verification failed. Please contact our support team.");
        break;

      case "ERROR":
        setStatus("error");
        break;

      default:
        break;
    }
  }

  // ── Manual transcript input (typed fallback) ─────────────────────────────────

  function handleManualSend(text: string) {
    if (!text.trim()) return;
    setTranscript(prev => [...prev.slice(-9), text]);
    sendToAgent({ type: "TRANSCRIPT_LINE", text });
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#09090b] text-white">

      {/* ════════════════ Left: video + speech ════════════════ */}
      <div className="flex flex-1 flex-col gap-3 p-4">

        {/* Status bar */}
        <div className="flex items-center gap-2 px-1">
          <span className={[
            "h-2.5 w-2.5 rounded-full",
            status === "error" || status === "fraud"
              ? "bg-red-500"
              : "bg-emerald-400 animate-pulse",
          ].join(" ")} />
          <span className="text-xs text-zinc-400 font-medium tracking-wide">
            {STATUS_LABELS[status]}
          </span>
          <span className="ml-auto text-[10px] text-zinc-600">
            {sessionId ? `Session: ${sessionId.slice(0, 8)}…` : ""}
          </span>
        </div>

        {/* Video frame */}
        <div
          ref={videoContainerRef}
          className="flex-1 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900"
        />

        {/* Priya's speech bubble */}
        <AnimatePresence mode="wait">
          <motion.div
            key={speech}
            variants={fadeSlide}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="flex items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900 p-4"
          >
            {/* Avatar */}
            <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center
                            rounded-full bg-gradient-to-br from-violet-600 to-blue-500
                            text-sm font-bold shadow-lg">
              P
            </div>
            <p className="text-sm leading-relaxed text-zinc-200">{speech}</p>
          </motion.div>
        </AnimatePresence>

        {/* Typed input (fallback when mic is unavailable) */}
        <ManualInput onSend={handleManualSend} disabled={status === "offer" || status === "declined"} />
      </div>

      {/* ════════════════ Right: AI Insights panel ════════════════ */}
      <aside className="flex w-80 flex-col gap-4 overflow-y-auto border-l border-zinc-800 p-4">

        <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-500">
          AI Insights
        </h2>

        {/* KYC fields */}
        <KYCCard fields={fields} />

        {/* Verification */}
        <AnimatePresence>
          {status === "verifying" && !verification && (
            <VerificationPending key="pending" />
          )}
          {verification && (
            <VerificationCard key="done" data={verification} />
          )}
        </AnimatePresence>

        {/* Loan offer (the hero moment) */}
        <AnimatePresence>
          {offer && status === "offer" && (
            <LoanOfferCard key="offer" offer={offer} />
          )}
          {status === "declined" && (
            <DeclineCard key="decline" />
          )}
          {status === "fraud" && (
            <FraudAlert key="fraud" />
          )}
        </AnimatePresence>

        {/* Live transcript */}
        {transcript.length > 0 && (
          <TranscriptLog lines={transcript} />
        )}
      </aside>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

// ── KYC fields card ──────────────────────────────────────────────────────────

function KYCCard({ fields }: { fields: CollectedFields }) {
  const allFields: KYCField[] = ["name", "pan", "income"];
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Collected details
      </p>
      <div className="flex flex-col gap-2.5">
        {allFields.map(f => (
          <div key={f} className="flex items-center justify-between">
            <span className="text-xs text-zinc-500">{fmtFieldLabel(f)}</span>
            <AnimatePresence mode="wait">
              {fields[f] != null ? (
                <motion.span
                  key={String(fields[f])}
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-xs font-medium text-emerald-400"
                >
                  {fmtFieldValue(f, fields[f]!)}
                </motion.span>
              ) : (
                <span key="empty" className="text-xs text-zinc-700">—</span>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Verification pending ─────────────────────────────────────────────────────

function VerificationPending() {
  const checks = ["CIBIL bureau lookup", "Deepfake artifact scan", "rPPG heart-rate"];
  return (
    <motion.div
      variants={fadeSlide} initial="hidden" animate="visible" exit="exit"
      className="rounded-xl border border-amber-700/30 bg-zinc-900 p-4"
    >
      <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-amber-400">
        Verifying in parallel
      </p>
      <div className="flex flex-col gap-2.5">
        {checks.map(c => (
          <div key={c} className="flex items-center gap-2">
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-zinc-800">
              <motion.div
                className="h-full rounded-full bg-amber-500"
                initial={{ width: "0%" }}
                animate={{ width: "100%" }}
                transition={{ duration: 2.4, ease: "easeInOut" }}
              />
            </div>
            <span className="w-40 truncate text-[11px] text-zinc-500">{c}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ── Verification result ──────────────────────────────────────────────────────

function VerificationCard({ data }: { data: VerificationResult }) {
  const riskColor = {
    LOW:    "text-emerald-400",
    MEDIUM: "text-amber-400",
    HIGH:   "text-red-400",
  }[data.fraud_risk] ?? "text-zinc-300";

  return (
    <motion.div
      variants={fadeSlide} initial="hidden" animate="visible"
      className="rounded-xl border border-zinc-700 bg-zinc-900 p-4"
    >
      <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Verification results
      </p>
      <div className="flex flex-col gap-2">
        <Row label="CIBIL score"  value={String(data.cibil_score)} />
        <Row label="Fraud risk"   value={data.fraud_risk}          color={riskColor} />
        <Row
          label="Deepfake"
          value={data.is_deepfake ? "Detected ⚠" : "Clear ✓"}
          color={data.is_deepfake ? "text-red-400" : "text-emerald-400"}
        />
        <Row label="Heart rate"   value={`${data.heart_rate_bpm} bpm`} />
        <Row label="Signal quality" value={data.rppg_quality} />
      </div>
    </motion.div>
  );
}

// ── Loan offer ───────────────────────────────────────────────────────────────

function LoanOfferCard({ offer }: { offer: LoanOffer }) {
  return (
    <motion.div
      variants={cardSpring} initial="hidden" animate="visible" exit="exit"
      className="rounded-2xl border border-violet-500/30 bg-gradient-to-br
                 from-violet-900/80 via-indigo-900/80 to-blue-900/80 p-5"
    >
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="h-2 w-2 rounded-full bg-violet-400"
        />
        <span className="text-[10px] font-semibold uppercase tracking-widest text-violet-300">
          Pre-approved offer
        </span>
      </div>

      {/* Loan amount */}
      <p className="text-4xl font-bold tracking-tight text-white">
        {fmtINR(offer.loan_amount)}
      </p>
      <p className="mt-1 text-sm text-violet-300">
        {offer.interest_rate}% p.a. · {offer.tenure_months} months
      </p>
      {offer.emi_approx && (
        <p className="mt-0.5 text-xs text-violet-400">
          EMI ≈ {fmtINR(offer.emi_approx)}/month
        </p>
      )}

      {/* Divider */}
      <div className="my-3 border-t border-violet-700/40" />

      {/* Meta */}
      <div className="flex justify-between text-[11px] text-violet-300">
        <span>CIBIL {offer.cibil_score}</span>
        <span>Zero processing fee*</span>
      </div>

      {/* CTA */}
      <button
        onClick={() => alert("Redirecting to loan acceptance portal…")}
        className="mt-4 w-full rounded-xl bg-white py-2.5 text-sm font-semibold
                   text-indigo-900 transition hover:bg-violet-100 active:scale-95"
      >
        Accept offer
      </button>
      <p className="mt-2 text-center text-[10px] text-violet-500">
        *Subject to documentation. Valid 24 hrs.
      </p>
    </motion.div>
  );
}

// ── Decline card ─────────────────────────────────────────────────────────────

function DeclineCard() {
  return (
    <motion.div
      variants={fadeSlide} initial="hidden" animate="visible"
      className="rounded-xl border border-zinc-700 bg-zinc-900 p-4 text-center"
    >
      <p className="text-sm text-zinc-300">
        We're unable to proceed with a pre-approval at this time.
      </p>
      <p className="mt-1 text-xs text-zinc-500">
        A PFL representative will contact you within 24 hours.
      </p>
    </motion.div>
  );
}

// ── Fraud alert ──────────────────────────────────────────────────────────────

function FraudAlert() {
  return (
    <motion.div
      variants={cardSpring} initial="hidden" animate="visible"
      className="rounded-xl border border-red-700/50 bg-red-950/40 p-4"
    >
      <p className="text-sm font-medium text-red-400">Identity check failed</p>
      <p className="mt-1 text-xs text-red-300/70">
        Please visit your nearest PFL branch or call 1800-XXX-XXXX.
      </p>
    </motion.div>
  );
}

// ── Transcript log ────────────────────────────────────────────────────────────

function TranscriptLog({ lines }: { lines: string[] }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-3">
      <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
        Live transcript
      </p>
      <div className="flex flex-col gap-1 max-h-32 overflow-y-auto">
        {lines.map((line, i) => (
          <p key={i} className="text-[11px] text-zinc-500 leading-relaxed">{line}</p>
        ))}
      </div>
    </div>
  );
}

// ── Shared Row component ──────────────────────────────────────────────────────

function Row({ label, value, color = "text-zinc-300" }: {
  label: string; value: string; color?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-xs font-medium ${color}`}>{value}</span>
    </div>
  );
}

// ── Manual input ──────────────────────────────────────────────────────────────

function ManualInput({ onSend, disabled }: {
  onSend: (t: string) => void; disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  function submit() {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
  }

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={e => e.key === "Enter" && submit()}
        placeholder="Type your response (mic fallback)…"
        disabled={disabled}
        className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2
                   text-sm text-zinc-200 placeholder-zinc-600 outline-none
                   focus:border-violet-600 disabled:opacity-40"
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="rounded-xl bg-violet-700 px-4 py-2 text-sm font-medium
                   text-white transition hover:bg-violet-600 disabled:opacity-40
                   active:scale-95"
      >
        Send
      </button>
    </div>
  );
}

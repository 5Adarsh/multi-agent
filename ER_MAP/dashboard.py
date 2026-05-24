"""
ER_MAP/dashboard.py
===================
Visual Dashboard for ER-MAP multi-agent triage simulation.
Shows God's View of all agent parameters + live conversation flow.

Usage:
    cd d:/Meta_Finals
    python -m ER_MAP.dashboard
    Open http://localhost:5050 in browser
"""

import json
import os
import sys
import time
import asyncio
import io
import threading
from flask import Flask, jsonify, request, Response, send_file

# ---------------------------------------------------------------------------
# Demo API keys (override via real environment variables for production).
# These are loaded into os.environ ONLY if the corresponding var is unset,
# so an externally-set key always takes precedence.
#
# 5-role / 4-key configuration (current demo build):
#     Doctor          -> Key D (8B model — fast, high daily quota)
#     Nurse           -> Key N (70B — realistic clinical communication)
#     Patient         -> Key P (70B — nuanced emotional persona)
#     Empathy Judge   -> Key J (70B — shared with Medical Judge)
#     Medical Judge   -> Key J (70B — shared with Empathy Judge)
#
# Each role has its own Groq client with independent rate-limit budgets,
# and Doctor / Nurse / Patient / Judges can run different models.
# ---------------------------------------------------------------------------
# Default *non-secret* values. API keys are intentionally empty here and
# must come from either (a) the user's shell, or (b) a local `.env` file
# loaded a few lines below. Never hardcode real keys in this dict — they
# would land in git and trip GitHub's push protection.
_DEMO_KEYS = {
    # --- Groq API keys (one per logical role) — populate via .env ---
    "GROQ_DOCTOR_API_KEY":          "",
    "GROQ_NURSE_API_KEY":           "",
    "GROQ_PATIENT_API_KEY":         "",
    "GROQ_EMPATHY_JUDGE_API_KEY":   "",
    "GROQ_MEDICAL_JUDGE_API_KEY":   "",

    # --- Per-role models (traffic-shaping for free-tier budget) ---
    # High-volume agents (Doctor / Nurse / Patient — fire on every env
    # step) run the 8B-instant pool: 14 400 RPD / 500K TPD per account.
    # The two judges fire mostly on terminal events but their grading
    # quality directly shapes the reward, so they stay on 70B-versatile
    # (1 000 RPD / 100K TPD pool — separate budget).
    "ERMAP_DOCTOR_MODEL":           "llama-3.1-8b-instant",
    "ERMAP_NURSE_MODEL":            "llama-3.1-8b-instant",
    "ERMAP_PATIENT_MODEL":          "llama-3.1-8b-instant",
    "ERMAP_EMPATHY_JUDGE_MODEL":    "llama-3.3-70b-versatile",
    "ERMAP_MEDICAL_JUDGE_MODEL":    "llama-3.3-70b-versatile",

    # --- ElevenLabs (single shared instance for UI voice) — populate via .env ---
    "ELEVENLABS_API_KEY":           "",
}


def _load_dotenv_into_environ(*candidates: str) -> str:
    """Tiny zero-dependency .env loader.

    Looks at the given candidate paths in order; the first one that
    exists is parsed for ``KEY=VALUE`` lines (lines starting with ``#``
    or blank are skipped, surrounding quotes on the value are stripped).
    Variables that already exist in ``os.environ`` are NOT overwritten —
    shell exports always win, matching the rest of this module's
    setdefault semantics.

    Returns the path that was loaded, or empty string if none.
    """
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Don't clobber a value the user already exported.
                    if key and not os.environ.get(key):
                        os.environ[key] = val
            return cand
        except Exception:  # noqa: BLE001 — never let .env errors crash startup
            continue
    return ""


# Look for a .env file in the repo root or alongside this module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DOTENV_PATH = _load_dotenv_into_environ(
    os.path.join(_REPO_ROOT, ".env"),
    os.path.join(_HERE, ".env"),
)
# Apply demo defaults. Behaviour depends on ERMAP_USE_DEMO_KEYS:
#   - if "1" / "true" -> force-override even if the shell already exports a value
#                        (use this when stale PowerShell env vars are getting
#                         in the way: $env:GROQ_NURSE_API_KEY left over from
#                         an earlier run can mask the current demo key)
#   - otherwise        -> setdefault (shell exports win, the historical default)
#
# We also collect a list of env vars where the shell value DIFFERS from the
# demo default so the startup banner can warn loudly.
_FORCE_DEMO = os.environ.get("ERMAP_USE_DEMO_KEYS", "").strip().lower() in {"1", "true", "yes"}
SHELL_OVERRIDES = []   # populated with names of env vars set externally
for _k, _v in _DEMO_KEYS.items():
    existing = os.environ.get(_k)
    if existing is None or existing == "":
        os.environ[_k] = _v
    elif existing != _v:
        if _FORCE_DEMO:
            os.environ[_k] = _v
        else:
            SHELL_OVERRIDES.append(_k)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global state
ENV = None
DOCTOR = None
EPISODE_STATE = {
    "active": False,
    "ground_truth": {},
    "conversation": [],
    "metrics": {"total_reward": 0, "step": 0, "outcome": None},
    "obs": "",
    "done": False,
    "reward_components": {},
    "phases_done": [],
    "current_phase": None,
}

# ---------------------------------------------------------------------------
# Clinical Phase Detection
# ---------------------------------------------------------------------------
PHASE_ORDER = [
    "READ_SOAP",
    "PATIENT_CONTACT",
    "VITALS",
    "LABS",
    "ASSESSMENT",
    "DISCHARGE",
]


def _detect_phase_from_action(action: dict) -> str:
    """Map a Doctor action to its corresponding clinical phase, if any."""
    tool = (action or {}).get("tool", "")
    target = (action or {}).get("target", "")
    section = (action or {}).get("section", "")
    if tool == "read_soap":
        return "READ_SOAP"
    if tool == "speak_to" and target == "patient":
        return "PATIENT_CONTACT"
    if tool == "order_lab":
        return "LABS"
    if tool == "update_soap" and "Assessment" in section:
        return "ASSESSMENT"
    if tool == "terminal_discharge":
        return "DISCHARGE"
    return ""


def _detect_phase_from_obs(obs_data: dict) -> str:
    """Detect phase from environment observation (e.g. nurse vitals report)."""
    event = obs_data.get("event", "")
    if event == "nurse_report":
        for ex in obs_data.get("internal_exchanges", []):
            action = (ex.get("nurse_action") or "").lower()
            if "vital" in action or "vitals" in action:
                return "VITALS"
    return ""


# ---------------------------------------------------------------------------
# Terminal-only ground truth banner
# ---------------------------------------------------------------------------

def _print_episode_to_terminal(gt: dict, phase: int, difficulty: str) -> None:
    """
    Print the full episode ground truth to the CMD/PowerShell terminal ONLY.
    The browser UI never receives this information — judges/operators see
    disease, emergency status, correct diagnosis/treatment, lethal treatments,
    and full patient/nurse personas in the terminal where the dashboard runs.
    """
    bar = "=" * 72
    sub = "-" * 72
    disease = (gt or {}).get("disease", {}) or {}
    patient = (gt or {}).get("patient", {}) or {}
    nurse   = (gt or {}).get("nurse", {}) or {}

    print("", flush=True)
    print(bar, flush=True)
    print(f"  ER-MAP NEW EPISODE  |  Phase {phase}  |  Difficulty: {difficulty or 'random'}", flush=True)
    print(bar, flush=True)

    # ----- Disease block -----
    is_emergency = bool(disease.get("is_emergency", False))
    print("", flush=True)
    print(f"  DISEASE        : {disease.get('true_disease', '???')}", flush=True)
    print(f"  EMERGENCY      : {'YES (time-critical)' if is_emergency else 'NO (stable)'}", flush=True)
    print(f"  CORRECT DX     : {disease.get('true_disease', '???')}", flush=True)
    print(f"  CORRECT TX     : {disease.get('correct_treatment', '???')}", flush=True)

    symptoms = disease.get("true_symptoms", []) or []
    if symptoms:
        print("  TRUE SYMPTOMS  :", flush=True)
        for s in symptoms:
            print(f"     - {s}", flush=True)

    lethal = disease.get("lethal_treatments", []) or []
    if lethal:
        print("  LETHAL Tx (avoid):", flush=True)
        for l in lethal:
            print(f"     - {l}", flush=True)

    # ----- Personas -----
    print("", flush=True)
    print(sub, flush=True)
    print("  PATIENT PERSONA", flush=True)
    print(sub, flush=True)
    if patient:
        for k, v in patient.items():
            print(f"     {k:<20s} : {v}", flush=True)
    else:
        print("     (no patient persona)", flush=True)

    print("", flush=True)
    print(sub, flush=True)
    print("  NURSE PERSONA", flush=True)
    print(sub, flush=True)
    if nurse:
        for k, v in nurse.items():
            print(f"     {k:<20s} : {v}", flush=True)
    else:
        print("     (no nurse persona)", flush=True)

    print(bar, flush=True)
    print("", flush=True)


def get_env():
    global ENV
    if ENV is None:
        from ER_MAP.envs.triage_env import TriageEnv
        # Per-role keys (4 distinct Groq accounts in the demo config:
        # Nurse / Patient / Empathy Judge / Medical Judge).
        nurse_key   = ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", ""))
        patient_key = ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", ""))
        empathy_key = ((os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY") or os.environ.get("empathy")) or os.environ.get("empathy", ""))
        medical_key = ((os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY") or os.environ.get("medical")) or os.environ.get("medical", ""))

        # Default / fallback model — used by any role that doesn't
        # specify its own model.
        default_model = os.environ.get("ERMAP_MODEL", "llama-3.3-70b-versatile")
        # Per-role models. Falling back to default_model lets each role
        # be configured independently (or all together via ERMAP_MODEL).
        nurse_model         = os.environ.get("ERMAP_NURSE_MODEL", default_model)
        patient_model       = os.environ.get("ERMAP_PATIENT_MODEL", default_model)
        empathy_judge_model = os.environ.get("ERMAP_EMPATHY_JUDGE_MODEL", default_model)
        medical_judge_model = os.environ.get("ERMAP_MEDICAL_JUDGE_MODEL", default_model)

        ENV = TriageEnv(
            nurse_api_key=nurse_key,
            patient_api_key=patient_key,
            empathy_judge_api_key=empathy_key,
            medical_judge_api_key=medical_key,
            model=default_model,
            nurse_model=nurse_model,
            patient_model=patient_model,
            empathy_judge_model=empathy_judge_model,
            medical_judge_model=medical_judge_model,
        )
    return ENV


def get_doctor():
    global DOCTOR
    if DOCTOR is None:
        # Doctor's primary key = its own dedicated Groq account.
        primary_key = ((os.environ.get("GROQ_DOCTOR_API_KEY") or os.environ.get("doctor")) or os.environ.get("doctor", "")) or ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", ""))
        # Full fallback chain: nurse → patient → empathy_judge → medical_judge.
        # Lets the demo survive 4-of-5 dead keys.
        fallback_keys = [
            ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", "")),
            ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", "")),
            ((os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY") or os.environ.get("empathy")) or os.environ.get("empathy", "")),
            ((os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY") or os.environ.get("medical")) or os.environ.get("medical", "")),
        ]
        # Doctor model defaults to 8B (small/fast tier). Override with
        # ERMAP_DOCTOR_MODEL or legacy ERMAP_MODEL.
        model = (
            os.environ.get("ERMAP_DOCTOR_MODEL")
            or os.environ.get("ERMAP_MODEL")
            or "llama-3.1-8b-instant"
        )
        DOCTOR = DoctorBrain(api_key=primary_key, fallback_api_keys=fallback_keys, model=model)
    return DOCTOR


# ---------------------------------------------------------------------------
# Doctor Brain
# ---------------------------------------------------------------------------
DOCTOR_SYSTEM_PROMPT = """You are an emergency room Doctor in an interactive triage simulation.

## Available Tools (respond with STRICT JSON — exactly one tool per turn)
1. read_soap         : {"thought":"...","tool":"read_soap","section":"Subjective | Objective | ALL"}
2. speak_to          : {"thought":"...","tool":"speak_to","target":"nurse | patient","message":"..."}
3. order_lab         : {"thought":"...","tool":"order_lab","target":"nurse","test_name":"<lab name>"}
4. update_soap       : {"thought":"...","tool":"update_soap","section":"Assessment | Plan","content":"<your text>"}
5. terminal_discharge: {"thought":"...","tool":"terminal_discharge","treatment":"<full treatment plan>"}

## Suggested clinical workflow (free to deviate when justified)
Step 1. read_soap (section=ALL) — review the chart BEFORE talking. Never skip this.
Step 2. speak_to patient — open with empathy + ask their chief complaint and current symptoms.
Step 3. speak_to nurse — ask for current vitals and any recent observations.
Step 4. order_lab — order ONE high-yield test relevant to the differential (e.g. CBC, BMP, troponin, lactate, ECG, CT, ABG).
Step 5. update_soap section=Assessment with your top differential diagnosis.
Step 6. update_soap section=Plan with your proposed plan.
Step 7. terminal_discharge with the full treatment (drugs + dosing + disposition + follow-up).

## Behavioural rules
- Be specific. Always ask a CONCRETE question with details (e.g. "Nurse, please check vitals"). Never use generic filler.
- Show empathy with the patient: acknowledge their concern in your first sentence before asking questions.
- Do not repeat actions you have already done. Look at the conversation so far before choosing the next tool.
- Document Assessment and Plan with update_soap before terminal_discharge — discharging without an Assessment
  is heavily penalized.
- NEVER prescribe a treatment you have not justified with at least one lab or vital sign.
- Stay grounded in the chart and what the nurse/patient actually told you.

RESPOND ONLY WITH A SINGLE VALID JSON OBJECT. No prose, no markdown, no code fences."""


def _smart_fallback_action(history: list) -> dict:
    """
    When the live LLM call fails (auth error, parse error, rate-limit), pick
    a *useful* clinical next action instead of dumping "Update me" every turn.

    Walks the conversation backwards to figure out which workflow stages the
    Doctor has already done, then advances to the next missing stage.
    """
    done_tools = set()
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        try:
            obj = json.loads(msg.get("content", "") or "{}")
            tool = obj.get("tool", "")
            if tool:
                done_tools.add(tool)
        except (json.JSONDecodeError, TypeError):
            continue

    if "read_soap" not in done_tools:
        return {
            "thought": "Local fallback: review the chart before any interaction.",
            "tool": "read_soap",
            "section": "ALL",
        }
    if "speak_to" not in done_tools:
        return {
            "thought": "Local fallback: open with patient and ask the chief complaint.",
            "tool": "speak_to",
            "target": "patient",
            "message": (
                "Hello, I'm the doctor on duty. I'm here to help you. "
                "Can you tell me what brought you to the ER today and what symptoms "
                "you're experiencing right now?"
            ),
        }
    if "order_lab" not in done_tools:
        return {
            "thought": "Local fallback: order a high-yield baseline panel.",
            "tool": "order_lab",
            "target": "nurse",
            "test_name": "CBC and Basic Metabolic Panel",
        }
    if "update_soap" not in done_tools:
        return {
            "thought": "Local fallback: document a working assessment.",
            "tool": "update_soap",
            "section": "Assessment",
            "content": (
                "Working differential pending lab results. Will refine after CBC/BMP "
                "and reassessment of vitals."
            ),
        }
    
    # Check if we also updated the Plan section before discharging
    has_plan_update = False
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        try:
            obj = json.loads(msg.get("content", "") or "{}")
            if obj.get("tool") == "update_soap" and obj.get("section") == "Plan":
                has_plan_update = True
        except:
            pass

    if not has_plan_update:
        return {
            "thought": "Local fallback: document treatment plan before discharge.",
            "tool": "update_soap",
            "section": "Plan",
            "content": "Empirical treatment plan: supportive care, IV fluids, and monitoring.",
        }

    return {
        "thought": "Local fallback: discharge patient to terminate case.",
        "tool": "terminal_discharge",
        "treatment": "Empirical care plan: supportive care, IV fluids, close observation, and follow-up.",
    }


class DoctorBrain:
    """
    Local Groq-backed Doctor with multi-stage automatic key rotation.

    The dashboard configures up to 5 Groq keys (Doctor / Nurse / Patient /
    Empathy Judge / Medical Judge). DoctorBrain holds its own primary key
    and a list of fallback keys (everyone else). If any client returns
    401 / invalid_api_key it is permanently blacklisted for the session
    and the next live client is used — so the Doctor stays online as long
    as ANY one of the 5 keys is still valid.
    """

    def __init__(self, api_key, model="llama-3.3-70b-versatile",
                 fallback_api_keys=None, fallback_api_key=""):
        from groq import Groq
        self.model = model

        # Build deduplicated, ordered key chain: primary first, then any
        # fallbacks (older callers may pass a single fallback_api_key).
        chain_raw = [api_key or ""]
        if fallback_api_keys:
            chain_raw.extend(fallback_api_keys)
        if fallback_api_key:
            chain_raw.append(fallback_api_key)

        seen = set()
        self._chain = []
        for k in chain_raw:
            if k and k not in seen:
                seen.add(k)
                self._chain.append({"key": k, "client": Groq(api_key=k), "dead": False})

        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

    @property
    def client(self):
        """Return the first non-dead Groq client in the chain (or None)."""
        for entry in self._chain:
            if not entry["dead"]:
                return entry["client"]
        return None

    def reset(self):
        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

    @staticmethod
    def _is_auth_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "401" in msg
            or "invalid_api_key" in msg
            or "invalid api key" in msg
            or "unauthorized" in msg
        )

    def _attempt_call(self, client) -> str:
        c = client.chat.completions.create(
            model=self.model, messages=self.history,
            temperature=0.6, max_tokens=320,
            response_format={"type": "json_object"},
        )
        return c.choices[0].message.content or ""

    def decide(self, observation):
        self.history.append({"role": "user", "content": f"Observation:\n{observation}"})
        if len(self.history) > 17:
            self.history = [self.history[0]] + self.history[-16:]

        resp = ""
        # Walk the key chain in order; on auth or rate failure mark dead/bypass and try next.
        for idx, entry in enumerate(self._chain):
            if entry["dead"]:
                continue
            try:
                resp = self._attempt_call(entry["client"])
                if resp:
                    break
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = "429" in msg or "rate" in msg
                if self._is_auth_error(e) or is_rate_limit:
                    entry["dead"] = True
                    masked = entry["key"][:8] + "..." + entry["key"][-4:] if len(entry["key"]) > 14 else "***"
                    err_type = "rate-limit" if is_rate_limit else "auth"
                    print(
                        f"  [DOCTOR] key #{idx + 1} ({masked}) returned {err_type} error; "
                        f"trying next fallback in the chain.",
                        flush=True,
                    )
                    continue
                print(f"  [DOCTOR] live call failed (key #{idx + 1}): {e}", flush=True)
                break  # general network / unknown error → don't burn the rest of the chain

        # If we still have nothing, pick a smart local fallback action
        # (read_soap → speak_to → order_lab → update_soap → status check)
        if not resp:
            fallback = _smart_fallback_action(self.history)
            resp = json.dumps(fallback)

        self.history.append({"role": "assistant", "content": resp})
        return resp


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


# ---------------------------------------------------------------------------
# TTS Engine — uses shared tts_engine module
# ---------------------------------------------------------------------------
from ER_MAP.tts_engine import TTSEngine, get_voice_key, clean_text_for_speech

# Lazy-initialized shared TTS engine for the dashboard
_tts_engine = None

def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine


@app.route("/api/speak", methods=["POST"])
def speak():
    """Generate neural TTS audio for a message."""
    data = request.json or {}
    text = data.get("text", "")
    agent = data.get("agent", "system")

    if not text or len(text.strip()) < 2:
        return jsonify({"error": "no text"}), 400

    print(f"  [TTS] agent={agent} text={text[:120]}", flush=True)

    gt = EPISODE_STATE.get("ground_truth", {})
    tts = _get_tts_engine()

    try:
        # pre_cleaned=True because frontend JS already cleaned the text
        audio_buf = tts.generate(text, agent, gt, pre_cleaned=True)
        if audio_buf is None:
            print(f"  [TTS] generate() returned None for agent={agent}", flush=True)
            return jsonify({"error": "generation failed"}), 500
        buf_size = audio_buf.getbuffer().nbytes
        print(f"  [TTS] Success: agent={agent} buf_size={buf_size}", flush=True)
        return send_file(audio_buf, mimetype="audio/mpeg")
    except Exception as e:
        import traceback
        print(f"  [TTS ERROR] agent={agent}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_episode", methods=["POST"])
def new_episode():
    global EPISODE_STATE
    env = get_env()
    doctor = get_doctor()
    doctor.reset()

    # Accept phase from frontend (default 1)
    req_data = request.json or {}
    phase = req_data.get("phase", 1)
    difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
    options = {"phase": phase, "difficulty": difficulty_map.get(phase, None)}
    print(f"  [ENV] Starting episode: phase={phase}, difficulty={options['difficulty']}", flush=True)

    obs, info = env.reset(options=options)
    gt = env.ground_truth

    # ----- Terminal-only ground-truth banner (operator's view) -----
    # Per demo spec: disease, emergency, correct Dx, and patient/nurse
    # personas appear ONLY on the CMD terminal — never in the browser UI.
    _print_episode_to_terminal(gt, phase, options.get("difficulty") or "")

    EPISODE_STATE = {
        "active": True,
        "ground_truth": gt,
        "conversation": [],
        "metrics": {"total_reward": 0, "step": 0, "outcome": None},
        "obs": obs,
        "done": False,
        "reward_components": {},
        "phases_done": [],
        "current_phase": None,
    }

    # Parse initial obs
    try:
        obs_data = json.loads(obs)
        EPISODE_STATE["conversation"].append({
            "agent": "system",
            "type": "episode_start",
            "message": f"New patient arrived. Nurse experience: {obs_data.get('nurse_experience', '?')}",
            "thought": None,
        })
    except:
        pass

    # NOTE: ground_truth is intentionally NOT returned to the browser —
    # disease, emergency status, and personas live on the CMD terminal only.
    # However, we now surface sanitized behavior traits for the 'God's View'.
    return jsonify({
        "status": "ok",
        "conversation": EPISODE_STATE["conversation"],
        "metrics": EPISODE_STATE["metrics"],
        "reward_components": {},
        "phases_done": [],
        "current_phase": None,
        "phase_order": PHASE_ORDER,
        "difficulty": options.get("difficulty") or "random",
        "phase": phase,
        "persona": {
            "patient": gt.get("patient", {}),
            "nurse": gt.get("nurse", {}),
            "is_emergency": gt.get("disease", {}).get("is_emergency", False),
        }
    })


@app.route("/api/step", methods=["POST"])
def step():
    global EPISODE_STATE
    if not EPISODE_STATE["active"] or EPISODE_STATE["done"]:
        return jsonify({"status": "no_active_episode"})

    env = get_env()
    doctor = get_doctor()

    # Doctor decides
    action_str = doctor.decide(EPISODE_STATE["obs"])
    try:
        action = json.loads(action_str)
    except (json.JSONDecodeError, TypeError):
        # Pick a useful next clinical step instead of dumping "Update me"
        action = _smart_fallback_action(doctor.history)
        action_str = json.dumps(action)

    # Log doctor action
    tool = action.get("tool", "speak_to")
    message = action.get("message", "")
    if tool == "read_soap":
        message = f"Read SOAP note (section: {action.get('section', 'ALL')})"
    elif tool == "update_soap":
        message = f"Update SOAP note (section: {action.get('section', '')}) to:\n{action.get('content', '')}"
    elif tool == "order_lab":
        message = f"Order lab: {action.get('test_name', '')}"
    elif tool == "terminal_discharge":
        message = f"Discharge patient with plan: {action.get('treatment', '')}"
    elif tool == "speak_to":
        message = action.get("message", "")

    EPISODE_STATE["conversation"].append({
        "agent": "doctor",
        "type": tool,
        "target": action.get("target", ""),
        "message": message,
        "thought": action.get("thought", ""),
    })

    # Step environment
    obs, reward, done, truncated, info = env.step(action_str)
    EPISODE_STATE["obs"] = obs
    EPISODE_STATE["metrics"]["total_reward"] = round(EPISODE_STATE["metrics"]["total_reward"] + reward, 2)
    EPISODE_STATE["metrics"]["step"] += 1
    EPISODE_STATE["metrics"]["last_reward"] = round(reward, 2)

    # Surface per-component reward breakdown for the sidebar Live Rewards panel
    components = info.get("reward_components", {}) if isinstance(info, dict) else {}
    if components:
        EPISODE_STATE["reward_components"] = {
            k: round(float(v), 3) for k, v in components.items()
        }

    # Track clinical phases — detect from doctor's action and from env observation
    phase_from_action = _detect_phase_from_action(action)
    if phase_from_action and phase_from_action not in EPISODE_STATE["phases_done"]:
        EPISODE_STATE["phases_done"].append(phase_from_action)
    EPISODE_STATE["current_phase"] = phase_from_action or EPISODE_STATE["current_phase"]

    # Parse observation and log responses
    def extract_spoken_text(raw):
        """Extract just the human-readable message from a raw LLM response."""
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("message", parsed.get("reason", raw))
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    def extract_thought(raw):
        """Extract the thought field from a raw LLM response."""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("thought", None)
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    try:
        obs_data = json.loads(obs)
        event = obs_data.get("event", "")

        # Detect VITALS phase from nurse internal exchanges
        phase_from_obs = _detect_phase_from_obs(obs_data)
        if phase_from_obs and phase_from_obs not in EPISODE_STATE["phases_done"]:
            EPISODE_STATE["phases_done"].append(phase_from_obs)

        if event == "nurse_report":
            # Log internal exchanges
            for ex in obs_data.get("internal_exchanges", []):
                if "nurse_said" in ex:
                    nurse_raw = ex.get("nurse_said", "")
                    patient_raw = ex.get("patient_said", "")
                    EPISODE_STATE["conversation"].append({
                        "agent": "nurse", "type": "speak_to", "target": "patient",
                        "message": extract_spoken_text(nurse_raw),
                        "thought": extract_thought(nurse_raw),
                    })
                    EPISODE_STATE["conversation"].append({
                        "agent": "patient", "type": "speak_to", "target": "nurse",
                        "message": extract_spoken_text(patient_raw),
                        "thought": extract_thought(patient_raw),
                        "status": ex.get("patient_status", ""),
                    })
                elif "nurse_action" in ex:
                    action_type = ex.get("nurse_action", "")
                    # Vitals check is meant for the doctor
                    target = "doctor" if action_type == "check_vitals" else "patient"
                    EPISODE_STATE["conversation"].append({
                        "agent": "nurse", "type": action_type,
                        "target": target,
                        "message": ex.get("result", ex.get("reason", "")),
                        "thought": None,
                    })
            # Nurse report to doctor
            nurse_msg_raw = obs_data.get("nurse_message", "")
            EPISODE_STATE["conversation"].append({
                "agent": "nurse", "type": "report", "target": "doctor",
                "message": extract_spoken_text(nurse_msg_raw),
                "thought": extract_thought(nurse_msg_raw),
            })

        elif event == "patient_response":
            patient_raw = obs_data.get("patient_message", "")
            EPISODE_STATE["conversation"].append({
                "agent": "patient", "type": "speak_to", "target": "doctor",
                "message": extract_spoken_text(patient_raw),
                "thought": extract_thought(patient_raw),
                "status": obs_data.get("patient_status", ""),
            })

        elif event == "lab_result":
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": "lab_result",
                "message": f"[{obs_data.get('test_name','')}] {obs_data.get('result','')}",
                "thought": None, "redundant": obs_data.get("redundant", False),
            })

        elif event == "soap_read":
            content = obs_data.get("content", "")
            if isinstance(content, dict):
                lines = []
                for k, v in content.items():
                    if isinstance(v, dict):
                        sub_lines = []
                        for sk, sv in v.items():
                            sub_lines.append(f"  • {sk}: {sv}")
                        lines.append(f"{k}:\n" + "\n".join(sub_lines))
                    else:
                        lines.append(f"{k}: {v}")
                msg_text = f"Read SOAP EMR (section={obs_data.get('section','')}). Content:\n\n" + "\n\n".join(lines)
            else:
                msg_text = f"Read SOAP EMR (section={obs_data.get('section','')}). Content: {content}"
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": "soap_read",
                "message": msg_text,
                "thought": None,
            })

        elif event == "soap_updated":
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": "soap_updated",
                "message": f"SOAP note updated (section={obs_data.get('section','')}). {obs_data.get('message','')}",
                "thought": None,
            })

        elif "terminal" in event:
            outcome_map = {"terminal_win": "WIN", "terminal_fatal": "FATAL",
                           "terminal_incorrect": "WRONG", "terminal_ama": "AMA"}
            outcome = outcome_map.get(event, event)
            EPISODE_STATE["metrics"]["outcome"] = outcome
            msg = obs_data.get("patient_message", obs_data.get("correct_treatment", ""))
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": event,
                "message": f"GAME OVER: {outcome}. {extract_spoken_text(msg)}",
                "thought": None,
            })

    except:
        pass

    if done or truncated:
        EPISODE_STATE["done"] = True
        if not EPISODE_STATE["metrics"]["outcome"]:
            EPISODE_STATE["metrics"]["outcome"] = "TRUNCATED"

    return jsonify({
        "status": "ok",
        "conversation": EPISODE_STATE["conversation"],
        "metrics": EPISODE_STATE["metrics"],
        "done": EPISODE_STATE["done"],
        "reward": round(reward, 2),
        "reward_components": EPISODE_STATE["reward_components"],
        "phases_done": EPISODE_STATE["phases_done"],
        "current_phase": EPISODE_STATE["current_phase"],
    })


@app.route("/api/state")
def state():
    return jsonify(EPISODE_STATE)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ER-MAP Simulation Canvas</title>
<script src="https://cdn.tailwindcss.com"></script>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  body { font-family: 'Inter', sans-serif; }

  /* Animations from temp.jsx */
  @keyframes audioWave {
    0%   { transform: scaleY(0.5); }
    100% { transform: scaleY(2.2); }
  }
  .animate-wave { animation: audioWave 0.4s ease-in-out infinite alternate; }

  @keyframes glowScale {
    0%   { transform: scale(1);   opacity: 0.8; }
    100% { transform: scale(2.5); opacity: 0;   }
  }
  .animate-glow-scale { animation: glowScale 1.5s ease-out infinite; }

  /* GPT-voice halo (Patient) */
  @keyframes gptPulseFast { 0%,100% { opacity:.8;  transform:scale(1);   } 50% { opacity:.4;  transform:scale(1.15); } }
  @keyframes gptPulseSlow { 0%,100% { opacity:.6;  transform:scale(1);   } 50% { opacity:.25; transform:scale(1.3);  } }
  @keyframes gptPulseHalo { 0%,100% { opacity:.4;  transform:scale(1);   } 50% { opacity:.15; transform:scale(1.5);  } }
  .animate-gpt-fast { animation: gptPulseFast 0.8s ease-in-out infinite; }
  .animate-gpt-slow { animation: gptPulseSlow 1.2s ease-in-out infinite reverse; }
  .animate-gpt-halo { animation: gptPulseHalo 2s   ease-in-out infinite; }

  .custom-scrollbar::-webkit-scrollbar { width: 4px; }
  .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(71, 85, 105, 0.4);
    border-radius: 10px;
  }
</style>
</head>
<body class="bg-slate-950">
  <div id="root"></div>

  <script type="text/babel" data-presets="react">
    const { useState, useEffect, useRef, useCallback } = React;

    // ------------------------------------------------------------------
    //  Inline Lucide icons (Stethoscope, User, Activity, Check)
    // ------------------------------------------------------------------
    const Stethoscope = ({ className }) => (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"
           strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 2v2"/>
        <path d="M5 2v2"/>
        <path d="M5 3a3 3 0 0 0-3 3v4a6 6 0 0 0 6 6h2a6 6 0 0 0 6-6V6a3 3 0 0 0-3-3"/>
        <path d="M8 15a6 6 0 0 0 12 0v-3"/>
        <circle cx="20" cy="10" r="2"/>
      </svg>
    );
    const User = ({ className }) => (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"
           strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    );
    const Activity = ({ className }) => (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"
           strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    );
    const Check = ({ className, strokeWidth }) => (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor"
           strokeWidth={strokeWidth || 3} strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 13l4 4L19 7"/>
      </svg>
    );

    // ------------------------------------------------------------------
    //  Phase order + reward-component label map
    // ------------------------------------------------------------------
    const PHASES = [
      { key: 'READ_SOAP',       label: '1. READ_SOAP'       },
      { key: 'PATIENT_CONTACT', label: '2. PATIENT_CONTACT' },
      { key: 'VITALS',          label: '3. VITALS'          },
      { key: 'LABS',            label: '4. LABS'            },
      { key: 'ASSESSMENT',      label: '5. ASSESSMENT'      },
      { key: 'DISCHARGE',       label: '6. DISCHARGE'       },
    ];

    const COMPONENT_LABELS = {
      process:       'Process',
      milestones:    'Milestones',
      labs:          'Labs',
      empathy:       'Empathy',
      consent:       'Consent',
      diagnosis:     'Diagnosis',
      plan:          'Plan',
      documentation: 'Documentation',
      emergency_id:  'Emergency ID',
      treatment:     'Treatment',
      penalties:     'Penalties',
    };

    // ------------------------------------------------------------------
    //  Animation sub-components
    // ------------------------------------------------------------------
    const GlowingCircleAnimation = ({ colorClass }) => (
      <div className="relative flex items-center justify-center w-full h-full">
        <div className={`absolute w-4 h-4 rounded-full ${colorClass} animate-glow-scale`} style={{ animationDelay: '0s' }} />
        <div className={`absolute w-4 h-4 rounded-full ${colorClass} animate-glow-scale`} style={{ animationDelay: '0.75s' }} />
        <div className={`absolute w-4 h-4 rounded-full ${colorClass} blur-[2px] opacity-80`} />
      </div>
    );

    const VoiceRecordAnimation = ({ colorClass }) => (
      <div className="relative flex items-center justify-center gap-[3px] w-full h-full">
        <div className={`w-[3px] h-3 rounded-full ${colorClass} animate-wave`} style={{ animationDelay: '0.0s' }} />
        <div className={`w-[3px] h-3 rounded-full ${colorClass} animate-wave`} style={{ animationDelay: '0.2s' }} />
        <div className={`w-[3px] h-3 rounded-full ${colorClass} animate-wave`} style={{ animationDelay: '0.4s' }} />
        <div className={`w-[3px] h-3 rounded-full ${colorClass} animate-wave`} style={{ animationDelay: '0.1s' }} />
        <div className={`w-[3px] h-3 rounded-full ${colorClass} animate-wave`} style={{ animationDelay: '0.3s' }} />
      </div>
    );

    const GptVoiceAnimation = ({ colorClass }) => (
      <div className="relative flex items-center justify-center w-full h-full">
        <div className={`absolute w-4  h-4  rounded-full ${colorClass} opacity-80 blur-[2px] animate-gpt-fast`} />
        <div className={`absolute w-6  h-6  rounded-full ${colorClass} opacity-60 blur-[4px] animate-gpt-slow`} />
        <div className={`absolute w-10 h-10 rounded-full ${colorClass} opacity-40 blur-[8px] animate-gpt-halo`} />
      </div>
    );

    // ------------------------------------------------------------------
    //  Strip JSON / control tokens out of LLM raw text before TTS
    // ------------------------------------------------------------------
    function cleanForSpeech(raw) {
      if (!raw) return '';
      try {
        const p = JSON.parse(raw);
        if (p && typeof p === 'object' && p.message) return p.message;
      } catch (e) { /* not JSON, fall through */ }
      let c = raw;
      c = c.replace(/[{}\[\]]/g, '');
      c = c.replace(/"/g, '');
      c = c.replace(/'/g, '');
      c = c.replace(/\b(thought|tool|target|status|test_name|message|speak_to|order_lab|terminal_discharge|check_vitals|administer_treatment|nurse_message|patient_message|event|nurse_report|patient_response|lab_result)\s*:/gi, '');
      c = c.replace(/\b(CONTINUE|ESCALATE|AGREE|LEAVE|null|true|false|undefined)\b/gi, '');
      c = c.replace(/\s*,\s*/g, ' ');
      c = c.replace(/\s+/g, ' ').trim();
      return c;
    }

    // ==================================================================
    //                              ROOT
    // ==================================================================
    function AgentCanvas() {
      const [activeAgent, setActiveAgent] = useState(null);
      const [isSidebarOpen, setIsSidebarOpen] = useState(true); // default open for full God view

      const [phase, setPhase]                 = useState(1);
      const [running, setRunning]             = useState(false);
      const [stepCount, setStepCount]         = useState(0);
      const [totalReward, setTotalReward]     = useState(0);
      const [rewardComponents, setRewardComponents] = useState({});
      const [phasesDone, setPhasesDone]       = useState([]);
      const [currentPhase, setCurrentPhase]   = useState(null);
      const [outcome, setOutcome]             = useState(null);
      const [persona, setPersona]             = useState(null);
      const [conversation, setConversation]   = useState([]);
      const [useVoice, setUseVoice]           = useState(false); // default to false for fast simulation without audio blocks

      const audioQueueRef    = useRef([]);
      const isPlayingRef     = useRef(false);
      const renderedCountRef = useRef(0);
      const stopRef          = useRef(false);
      const chatEndRef       = useRef(null);

      useEffect(() => {
        if (chatEndRef.current) {
          chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }, [conversation]);

      // ---------------- Audio queue ----------------
      const processQueue = useCallback(async () => {
        if (isPlayingRef.current || audioQueueRef.current.length === 0) return;
        isPlayingRef.current = true;
        const item = audioQueueRef.current.shift();
        try {
          const res = await fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: item.text, agent: item.agent }),
          });
          if (!res.ok) {
            isPlayingRef.current = false;
            processQueue();
            return;
          }
          const blob = await res.blob();
          const url  = URL.createObjectURL(blob);
          const audio = new Audio(url);

          let finished = false;
          const finish = () => {
            if (finished) return;
            finished = true;
            URL.revokeObjectURL(url);
            setActiveAgent(prev => (prev === item.agent ? null : prev));
            isPlayingRef.current = false;
            processQueue();
          };

          const watchdog = setTimeout(finish, 60000);
          audio.onplay  = () => setActiveAgent(item.agent);
          audio.onended = () => { clearTimeout(watchdog); finish(); };
          audio.onerror = () => { clearTimeout(watchdog); finish(); };

          const playPromise = audio.play();
          if (playPromise !== undefined) {
            playPromise.catch(() => { clearTimeout(watchdog); finish(); });
          }
        } catch (e) {
          isPlayingRef.current = false;
          processQueue();
        }
      }, []);

      const enqueueSpeech = useCallback((rawText, agent) => {
        if (!useVoice) return;
        const clean = cleanForSpeech(rawText);
        if (!clean || clean.length < 3) return;
        audioQueueRef.current.push({ text: clean, agent });
        processQueue();
      }, [useVoice, processQueue]);

      const waitForAudioDrained = () => new Promise((resolve) => {
        if (!useVoice) {
          resolve();
          return;
        }
        const tick = () => {
          if (!isPlayingRef.current && audioQueueRef.current.length === 0) {
            resolve(); return;
          }
          setTimeout(tick, 250);
        };
        tick();
      });

      // ---------------- Episode flow ----------------
      const startCase = async () => {
        if (running) return;
        stopRef.current = false;
        setRunning(true);
        setOutcome(null);
        setRewardComponents({});
        setTotalReward(0);
        setPhasesDone([]);
        setCurrentPhase(null);
        setStepCount(0);
        renderedCountRef.current = 0;
        audioQueueRef.current    = [];
        isPlayingRef.current     = false;

        try {
          const res = await fetch('/api/new_episode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phase }),
          });
          const data = await res.json();
          setPersona(data.persona);
          setConversation(data.conversation || []);
          renderedCountRef.current = (data.conversation || []).length;
        } catch (e) {
          console.error('new_episode failed', e);
          setRunning(false);
          return;
        }

        autoLoop();
      };

      const stopCase = () => {
        stopRef.current = true;
        audioQueueRef.current = [];
        isPlayingRef.current  = false;
        setRunning(false);
        setActiveAgent(null);
      };

      const autoLoop = async () => {
        while (!stopRef.current) {
          let stepData;
          try {
            const r = await fetch('/api/step', { method: 'POST' });
            stepData = await r.json();
          } catch (e) {
            console.error('step failed', e);
            break;
          }
          if (stepData.status === 'no_active_episode') break;

          // Enqueue any NEW spoken messages
          const all = stepData.conversation || [];
          const newMsgs = all.slice(renderedCountRef.current);
          renderedCountRef.current = all.length;
          for (const m of newMsgs) {
            if (!m || !m.message || !m.agent || m.agent === 'system') continue;
            const SPOKEN = ['speak_to', 'report', 'terminal_discharge'];
            if (SPOKEN.includes(m.type)) enqueueSpeech(m.message, m.agent);
          }

          // Update HUD / sidebar
          if (stepData.metrics) {
            setTotalReward(stepData.metrics.total_reward || 0);
            setStepCount(stepData.metrics.step || 0);
            if (stepData.metrics.outcome) setOutcome(stepData.metrics.outcome);
          }
          if (stepData.reward_components) setRewardComponents(stepData.reward_components);
          if (stepData.phases_done)        setPhasesDone(stepData.phases_done);
          if (stepData.current_phase !== undefined) setCurrentPhase(stepData.current_phase);
          setConversation(stepData.conversation || []);

          // Highlight the currently active speaking agent visually if voice is disabled
          if (!useVoice && all.length > 0) {
            const lastMsg = all[all.length - 1];
            if (lastMsg && lastMsg.agent && lastMsg.agent !== 'system') {
              setActiveAgent(lastMsg.agent);
              await new Promise(r => setTimeout(r, 1200));
              setActiveAgent(null);
            }
          }

          if (stepData.done) {
            await waitForAudioDrained();
            break;
          }

          await waitForAudioDrained();
          await new Promise(r => setTimeout(r, 500));
        }
        setRunning(false);
        setActiveAgent(null);
      };

      return (
        <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 font-sans text-slate-200 selection:bg-indigo-500/30">

          {/* Main Dashboard Panel */}
          <div className="relative w-full max-w-6xl h-[700px] bg-slate-900/60 backdrop-blur-2xl rounded-3xl shadow-[0_20px_60px_-15px_rgba(0,0,0,0.5)] border border-slate-800/80 overflow-hidden flex">

            {/* Subtle grid background */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
                 style={{
                   backgroundImage: 'linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)',
                   backgroundSize: '24px 24px',
                 }}/>

            {/* ---------------- COLLAPSIBLE SIDEBAR ---------------- */}
            <div
              className={`relative border-r border-slate-800/60 bg-slate-900/70 backdrop-blur-md z-20 shadow-[4px_0_24px_rgba(0,0,0,0.2)] transition-all duration-300 ease-in-out overflow-hidden flex shrink-0
                ${isSidebarOpen ? 'w-[320px]' : 'w-14'}`}
            >
              {/* Closed Toggle Button (Vertical Text + Hover) */}
              {!isSidebarOpen && (
                <div
                  onClick={() => setIsSidebarOpen(true)}
                  className="absolute inset-0 flex flex-col items-center justify-between py-6 cursor-pointer hover:bg-slate-800/40 transition-colors"
                >
                  <div className="w-6 h-6 rounded-full bg-slate-800/60 flex items-center justify-center border border-slate-700/50 hover:bg-indigo-650 transition-colors">
                    <svg className="w-3 h-3 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7"/>
                    </svg>
                  </div>
                  <div className="rotate-180 select-none" style={{ writingMode: 'vertical-rl' }}>
                    <span className="text-[10px] font-bold tracking-[0.25em] text-slate-500 uppercase whitespace-nowrap">METRICS & PERSONAS</span>
                  </div>
                  <div className="w-2 h-2 rounded-full bg-indigo-500/50" />
                </div>
              )}

              {/* Open Sidebar State */}
              {isSidebarOpen && (
                <div className="w-full p-5 flex flex-col transition-opacity duration-300 overflow-y-auto custom-scrollbar">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-800/50 shrink-0">
                    <span className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase">GOD'S VIEW</span>
                    <button
                      onClick={() => setIsSidebarOpen(false)}
                      className="w-5 h-5 rounded-full bg-slate-800/50 flex items-center justify-center border border-slate-700/50 hover:bg-slate-700 transition-colors">
                      <svg className="w-2.5 h-2.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7"/>
                      </svg>
                    </button>
                  </div>

                  {/* Section A: Live Rewards */}
                  <div className="mb-6 shrink-0">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-[10px] font-bold text-slate-450 uppercase tracking-wider">Live Rewards</h3>
                      <span className={`text-[11px] font-mono font-bold ${totalReward > 0.01 ? 'text-green-400 drop-shadow-[0_0_4px_rgba(74,222,128,0.3)]' : totalReward < -0.01 ? 'text-red-400 drop-shadow-[0_0_4px_rgba(248,113,113,0.3)]' : 'text-slate-500'}`}>
                        {totalReward >= 0 ? '+' : ''}{totalReward.toFixed(2)} Total
                      </span>
                    </div>
                    <div className="bg-slate-950/45 rounded-xl border border-slate-850 p-3 h-[110px] overflow-y-auto font-mono text-[10px] space-y-2 custom-scrollbar shadow-inner">
                      {Object.keys(COMPONENT_LABELS)
                        .filter(k => rewardComponents[k] !== undefined && Math.abs(rewardComponents[k]) > 0.001)
                        .map(k => {
                          const v = rewardComponents[k];
                          const sign = v >= 0 ? '+' : '';
                          const cls = v > 0 ? 'text-green-400/90' : v < 0 ? 'text-red-400/90' : 'text-slate-500';
                          return (
                            <div key={k} className="flex justify-between items-center border-b border-slate-900/50 pb-1 last:border-0 last:pb-0">
                              <span className="text-slate-405">{COMPONENT_LABELS[k]}</span>
                              <span className={cls}>{sign}{v.toFixed(2)}</span>
                            </div>
                          );
                        })}
                      {Object.keys(rewardComponents).filter(k => Math.abs(rewardComponents[k] || 0) > 0.001).length === 0 && (
                        <div className="text-slate-600 italic py-2 text-center">Reward breakdown will load live.</div>
                      )}
                    </div>
                  </div>

                  {/* Section B: Clinical Phases */}
                  <div className="mb-6 shrink-0 border-t border-slate-800/40 pt-4">
                    <h3 className="text-[10px] font-bold text-slate-450 uppercase tracking-wider mb-3">Clinical Phases</h3>
                    <div className="space-y-2.5">
                      {PHASES.map(({ key, label }, i) => {
                        const done    = phasesDone.includes(key);
                        const current = !done && key === currentPhase;
                        return (
                          <div key={key} className="flex items-center gap-2.5">
                            <div className={`w-3.5 h-3.5 rounded-[4px] border flex items-center justify-center transition-colors shrink-0
                              ${done ? 'bg-indigo-950/40 border-indigo-500/30' : current ? 'border-indigo-500/50 bg-indigo-500/10' : 'border-slate-800 bg-slate-900/20'}`}>
                              {done && <Check className="w-2.5 h-2.5 text-indigo-400" strokeWidth={3.5} />}
                            </div>
                            <span className={`text-[10px] font-semibold tracking-wide ${done ? 'text-slate-500 line-through' : current ? 'text-indigo-300 drop-shadow-[0_0_6px_rgba(129,140,248,0.4)] font-bold' : 'text-slate-650'}`}>
                              {label}
                            </span>
                            {current && (
                              <span className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-450 animate-pulse shadow-[0_0_8px_rgba(129,140,248,0.8)]" />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Section C: Agent Personas */}
                  {persona && (
                    <div className="border-t border-slate-800/50 pt-4 pb-2">
                      <div className="flex items-center justify-between mb-3 shrink-0">
                        <h3 className="text-[10px] font-bold text-slate-450 uppercase tracking-wider">Target Persona</h3>
                        {persona.is_emergency ? (
                          <span className="text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse">🚨 EMERGENT</span>
                        ) : (
                          <span className="text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-emerald-550/20 text-emerald-450 border border-emerald-500/30">STABLE</span>
                        )}
                      </div>
                      <div className="space-y-4 text-[10px]">
                        {/* Patient */}
                        <div>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <User className="w-3 h-3 text-teal-400" />
                            <span className="font-bold text-teal-400/80 uppercase">Patient Traits</span>
                          </div>
                          <div className="grid grid-cols-2 gap-1.5">
                            {Object.entries(persona.patient).map(([k, v]) => (
                              <div key={k} className="bg-slate-950/40 border border-slate-850 rounded-lg p-1.5">
                                <div className="text-[8px] text-slate-500 uppercase leading-none mb-1">{k}</div>
                                <div className="text-[9px] text-slate-350 font-medium truncate">{String(v).replace(/_/g, ' ')}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                        {/* Nurse */}
                        <div>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <Activity className="w-3 h-3 text-blue-400" />
                            <span className="font-bold text-blue-400/80 uppercase">Nurse Traits</span>
                          </div>
                          <div className="grid grid-cols-2 gap-1.5">
                            {Object.entries(persona.nurse).map(([k, v]) => (
                              <div key={k} className="bg-slate-950/40 border border-slate-850 rounded-lg p-1.5">
                                <div className="text-[8px] text-slate-500 uppercase leading-none mb-1">{k}</div>
                                <div className="text-[9px] text-slate-350 font-medium truncate">{String(v).replace(/_/g, ' ')}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ---------------- MAIN DISPLAY AREA ---------------- */}
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              
              {/* Header / Control Pill Bar */}
              <div className="h-16 border-b border-slate-800/80 px-6 flex items-center justify-between bg-slate-900/40 backdrop-blur-sm z-10 shrink-0">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">ER-MAP TRIAGE CORE</span>
                  {running && (
                    <span className="flex h-1.5 w-1.5 relative">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-450 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                    </span>
                  )}
                </div>
                
                {/* Control Panel */}
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={useVoice}
                      onChange={(e) => setUseVoice(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-950 text-indigo-600 focus:ring-indigo-500/30"
                    />
                    <span>Enable Voice</span>
                  </label>
                  <span className="text-slate-800">|</span>
                  {!running ? (
                    <>
                      <select
                        value={phase}
                        onChange={(e) => setPhase(Number(e.target.value))}
                        className="bg-slate-950 border border-slate-700 text-slate-300 text-xs rounded-lg px-2.5 py-1.5 cursor-pointer focus:outline-none focus:border-indigo-500/50">
                        <option value={1}>Phase 1 · Tool Mastery</option>
                        <option value={2}>Phase 2 · Clinical Reasoning</option>
                        <option value={3}>Phase 3 · Empathy + Chaos</option>
                      </select>
                      <button
                        onClick={startCase}
                        className="bg-indigo-650 hover:bg-indigo-600 text-white text-xs font-semibold tracking-wider uppercase px-4 py-1.5 rounded-lg transition flex items-center gap-1.5 shadow-[0_4px_12px_rgba(99,102,241,0.25)]">
                        Start Case
                      </button>
                    </>
                  ) : (
                    <div className="flex items-center gap-3 bg-slate-950 border border-slate-850 px-4 py-1.5 rounded-lg text-xs font-mono">
                      <span className="text-slate-450">Phase <span className="text-indigo-400 font-semibold">{phase}</span></span>
                      <span className="text-slate-700">|</span>
                      <span className="text-slate-450">Step <span className="text-indigo-400 font-semibold">{stepCount}</span></span>
                      <span className="text-slate-700">|</span>
                      <span className={`font-bold ${totalReward > 0.01 ? 'text-emerald-450' : totalReward < -0.01 ? 'text-red-405' : 'text-slate-300'}`}>
                        {totalReward >= 0 ? '+' : ''}{totalReward.toFixed(2)} Rwd
                      </span>
                      {outcome && (
                        <>
                          <span className="text-slate-700">|</span>
                          <span className={`font-bold uppercase tracking-wider px-1.5 py-0.5 rounded text-[9px] ${
                            outcome === 'WIN' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                            outcome === 'AMA' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                            'bg-red-500/20 text-red-400 border border-red-500/30'
                          }`}>{outcome}</span>
                        </>
                      )}
                      <button onClick={stopCase} className="ml-2 text-slate-500 hover:text-red-400 text-sm leading-none transition-colors">×</button>
                    </div>
                  )}
                </div>
              </div>

              {/* Grid Content Area: Prevents overlap completely */}
              <div className="flex-1 grid grid-cols-12 gap-5 p-6 overflow-hidden min-h-0">
                
                {/* Column A (cols 1-4): Doctor & Nurse Cards */}
                <div className="col-span-4 flex flex-col gap-4 justify-center min-h-0">
                  
                  {/* Doctor Card */}
                  <div className={`w-full bg-slate-900/80 backdrop-blur-xl rounded-2xl p-4.5 border shadow-md transition-all duration-500 flex items-center gap-3.5
                    ${activeAgent === 'doctor' ? 'shadow-[0_8px_30px_rgba(99,102,241,0.15)] border-indigo-500/40 scale-[1.015]' : 'border-slate-800 hover:border-slate-750'}`}>
                    <div className="relative flex items-center justify-center w-11 h-11 shrink-0">
                      <div className="z-10 relative bg-slate-850 rounded-full p-2 border border-slate-750 flex items-center justify-center w-full h-full shadow">
                        {activeAgent === 'doctor'
                          ? <GlowingCircleAnimation colorClass="bg-indigo-400" />
                          : <Stethoscope className="w-5 h-5 text-slate-450" />}
                      </div>
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-semibold text-slate-200 text-sm truncate">Doctor</h3>
                      <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mt-0.5">{activeAgent === 'doctor' ? 'Analyzing Case...' : 'Attending Physician'}</p>
                    </div>
                    <div className="ml-auto shrink-0">
                      {activeAgent === 'doctor' ? (
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-indigo-500"></span>
                        </span>
                      ) : (
                        <div className="w-1.5 h-1.5 rounded-full bg-slate-750" />
                      )}
                    </div>
                  </div>

                  {/* Nurse Card */}
                  <div className={`w-full bg-slate-900/80 backdrop-blur-xl rounded-2xl p-4.5 border shadow-md transition-all duration-500 flex items-center gap-3.5
                    ${activeAgent === 'nurse' ? 'shadow-[0_8px_30px_rgba(59,130,246,0.15)] border-blue-500/40 scale-[1.015]' : 'border-slate-800 hover:border-slate-750'}`}>
                    <div className="relative flex items-center justify-center w-11 h-11 shrink-0">
                      <div className="z-10 relative bg-slate-850 rounded-full p-2 border border-slate-750 flex items-center justify-center w-full h-full shadow">
                        {activeAgent === 'nurse'
                          ? <VoiceRecordAnimation colorClass="bg-blue-400" />
                          : <Activity className="w-5 h-5 text-slate-450" />}
                      </div>
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-semibold text-slate-200 text-sm truncate">Nurse</h3>
                      <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mt-0.5">{activeAgent === 'nurse' ? 'Reporting Vitals...' : 'Triage Nurse'}</p>
                    </div>
                    <div className="ml-auto shrink-0">
                      {activeAgent === 'nurse' ? (
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-blue-500"></span>
                        </span>
                      ) : (
                        <div className="w-1.5 h-1.5 rounded-full bg-slate-750" />
                      )}
                    </div>
                  </div>

                </div>

                {/* Column B (cols 5-9): Scrolling Dialogue Transcript Console */}
                <div className="col-span-5 flex flex-col h-full bg-slate-950/45 border border-slate-850/80 rounded-2xl p-4 shadow-inner min-h-0">
                  <div className="text-[10px] font-bold text-slate-550 uppercase tracking-widest mb-3 border-b border-slate-850 pb-2 flex items-center justify-between shrink-0">
                    <span>Dialogue Transcript</span>
                    {running && <span className="text-[8px] font-normal text-indigo-400 animate-pulse tracking-normal">Streaming Live</span>}
                  </div>
                  <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 pr-1 min-h-0">
                    {conversation.map((msg, idx) => {
                      const color = 
                        msg.agent === 'doctor' ? 'text-indigo-400' :
                        msg.agent === 'nurse' ? 'text-blue-400' :
                        msg.agent === 'patient' ? 'text-teal-400' :
                        'text-slate-400';
                      
                      const bgColor = 
                        msg.agent === 'doctor' ? 'bg-indigo-950/20 border-indigo-900/20' :
                        msg.agent === 'nurse' ? 'bg-blue-950/20 border-blue-900/20' :
                        msg.agent === 'patient' ? 'bg-teal-950/20 border-teal-900/20' :
                        'bg-slate-900/20 border-slate-800/20';

                      return (
                        <div key={idx} className={`text-xs border rounded-xl p-3 shadow-sm ${bgColor}`}>
                          <div className="flex items-center justify-between font-mono text-[9px] font-bold mb-1">
                            <div className="flex items-center gap-1.5">
                              <span className={`px-1.5 py-0.5 rounded bg-slate-950 ${color}`}>{msg.agent.toUpperCase()}</span>
                              <span className="text-slate-500 font-normal">→</span>
                              <span className="text-slate-400">{msg.target ? msg.target.toUpperCase() : 'ALL'}</span>
                            </div>
                            {msg.type && <span className="text-[8px] text-slate-400 font-normal tracking-wide">{msg.type.toUpperCase()}</span>}
                          </div>
                          {msg.thought && (
                            <div className="text-[10px] text-slate-500 italic mb-1.5 font-normal leading-relaxed border-l border-slate-800 pl-2">
                              {msg.thought}
                            </div>
                          )}
                          <div className="text-slate-300 leading-relaxed font-normal whitespace-pre-wrap">
                            {msg.message}
                          </div>
                        </div>
                      );
                    })}
                    {conversation.length === 0 && (
                      <div className="h-full flex flex-col items-center justify-center text-slate-600 text-center text-xs italic py-10">
                        <svg className="w-8 h-8 text-slate-700/80 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                        </svg>
                        <span>No conversation events logged yet.</span>
                        <span className="text-[10px] text-slate-750 mt-1">Start a triage case to stream dialogue here.</span>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                </div>

                {/* Column C (cols 10-12): Patient Card */}
                <div className="col-span-3 flex flex-col justify-center min-h-0">
                  
                  {/* Patient Card */}
                  <div className={`w-full bg-slate-900/80 backdrop-blur-xl rounded-2xl p-4.5 border shadow-md transition-all duration-500 flex items-center gap-3.5
                    ${activeAgent === 'patient' ? 'shadow-[0_8px_30px_rgba(20,184,166,0.15)] border-teal-500/40 scale-[1.015]' : 'border-slate-800 hover:border-slate-750'}`}>
                    <div className="relative flex items-center justify-center w-11 h-11 shrink-0">
                      <div className="z-10 relative bg-slate-850 rounded-full p-2 border border-slate-750 flex items-center justify-center w-full h-full shadow">
                        {activeAgent === 'patient'
                          ? <GptVoiceAnimation colorClass="bg-teal-400" />
                          : <User className="w-5 h-5 text-slate-450" />}
                      </div>
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-semibold text-slate-200 text-sm truncate">Patient</h3>
                      <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mt-0.5">{activeAgent === 'patient' ? 'Speaking...' : 'Visiting'}</p>
                    </div>
                    <div className="ml-auto shrink-0">
                      {activeAgent === 'patient' ? (
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-450 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-teal-500"></span>
                        </span>
                      ) : (
                        <div className="w-1.5 h-1.5 rounded-full bg-slate-750" />
                      )}
                    </div>
                  </div>

                </div>

              </div>

            </div>

          </div>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(
      <React.StrictMode>
        <AgentCanvas />
      </React.StrictMode>
    );
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _probe_groq_key(key: str, model: str, timeout_s: float = 6.0):
    """
    Send a tiny 1-token completion to Groq to verify a key is live.

    Returns ``(status, detail)`` where status is one of:
        ``"ALIVE"``     – key responded successfully
        ``"DEAD_AUTH"`` – 401 / invalid_api_key (key revoked or wrong)
        ``"DEAD_RATE"`` – 429 rate-limit (key works, just throttled)
        ``"DEAD_NET"``  – timeout / connection error / other transient issue
        ``"MISSING"``   – key is empty
    """
    if not key:
        return "MISSING", ""
    try:
        from groq import Groq
        c = Groq(api_key=key, timeout=timeout_s)
        c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        return "ALIVE", ""
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "invalid_api_key" in msg or "invalid api key" in msg or "unauthorized" in msg:
            return "DEAD_AUTH", "401 invalid_api_key"
        if "429" in msg or "rate" in msg:
            return "DEAD_RATE", "rate limited"
        return "DEAD_NET", str(e).splitlines()[0][:80]


def _print_role_config_banner() -> None:
    """
    Print the live role -> (key, model, health) wiring to the terminal.

    Each role's key is pinged with a 1-token completion so the operator
    instantly sees ``[ALIVE]`` / ``[DEAD 401]`` / ``[RATE]`` next to the
    five roles before the demo starts. Keys that share the same value
    are only probed once (cached) to save quota.
    """
    def _mask(k: str) -> str:
        if not k:
            return "(missing)"
        if len(k) < 14:
            return "(short:" + k[:4] + "...)"
        return k[:8] + "..." + k[-4:]

    rows = [
        ("Doctor",        ((os.environ.get("GROQ_DOCTOR_API_KEY") or os.environ.get("doctor")) or os.environ.get("doctor", "")),         os.environ.get("ERMAP_DOCTOR_MODEL", "llama-3.1-8b-instant")),
        ("Nurse",         ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", "")),          os.environ.get("ERMAP_NURSE_MODEL", "llama-3.3-70b-versatile")),
        ("Patient",       ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", "")),        os.environ.get("ERMAP_PATIENT_MODEL", "llama-3.3-70b-versatile")),
        ("Empathy Judge", ((os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY") or os.environ.get("empathy")) or os.environ.get("empathy", "")),  os.environ.get("ERMAP_EMPATHY_JUDGE_MODEL", "llama-3.3-70b-versatile")),
        ("Medical Judge", ((os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY") or os.environ.get("medical")) or os.environ.get("medical", "")),  os.environ.get("ERMAP_MEDICAL_JUDGE_MODEL", "llama-3.3-70b-versatile")),
    ]

    print("", flush=True)
    print("=" * 88, flush=True)
    print("  ER-MAP ROLE  -> KEY / MODEL CONFIGURATION  (probing keys against Groq...)", flush=True)
    print("=" * 88, flush=True)

    probe_cache = {}
    statuses = []
    for role, key, model in rows:
        cache_key = (key, model)
        if cache_key in probe_cache:
            status, detail = probe_cache[cache_key]
        else:
            status, detail = _probe_groq_key(key, model)
            probe_cache[cache_key] = (status, detail)
        statuses.append((role, key, model, status, detail))

        tag = {
            "ALIVE":     "[ALIVE]    ",
            "DEAD_AUTH": "[DEAD 401] ",
            "DEAD_RATE": "[RATE-LIM] ",
            "DEAD_NET":  "[NET-ERR]  ",
            "MISSING":   "[MISSING]  ",
        }[status]
        suffix = f"  ({detail})" if detail else ""
        print(f"  {tag} {role:<14s} | {_mask(key):<22s} | {model}{suffix}", flush=True)

    alive_count = sum(1 for _, _, _, s, _ in statuses if s == "ALIVE")
    dead_auth = [r for r, _, _, s, _ in statuses if s == "DEAD_AUTH"]
    print("-" * 88, flush=True)
    print(f"  Live keys: {alive_count}/5", flush=True)
    if dead_auth:
        print(f"  ACTION:    regenerate {', '.join(dead_auth)} at https://console.groq.com/keys", flush=True)
        print(f"             dead roles will auto-fallback to surviving keys at runtime.", flush=True)
    if alive_count == 0:
        print(f"  WARNING:   no live Groq keys! Demo will use mock responses only.", flush=True)
    if SHELL_OVERRIDES:
        print("", flush=True)
        print(f"  >>> SHELL ENV OVERRIDE WARNING <<<", flush=True)
        print(f"  These env vars are set in your shell and OVERRIDE the demo defaults", flush=True)
        print(f"  baked into dashboard.py (so the keys above came from PowerShell, not", flush=True)
        print(f"  from the source file):", flush=True)
        for name in SHELL_OVERRIDES:
            print(f"    - {name}", flush=True)
        print(f"", flush=True)
        print(f"  If a key is dead, EITHER:", flush=True)
        print(f"    (A) Clear the stale exports and rerun:", flush=True)
        for name in SHELL_OVERRIDES:
            print(f"          Remove-Item Env:\\{name}", flush=True)
        print(f"    (B) Or force the demo defaults: $env:ERMAP_USE_DEMO_KEYS=\"1\"", flush=True)
    print("=" * 88, flush=True)
    print("", flush=True)


if __name__ == "__main__":
    _print_role_config_banner()
    print("  ER-MAP Dashboard: http://localhost:5050\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)

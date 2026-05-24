"""
ER_MAP/evaluate_baseline.py
===========================
Per-phase baseline evaluation for the ER-MAP Doctor agent (no RL).

Runs ``--episodes-per-phase`` rollouts (default: 20) of the baseline LLM
Doctor against each curriculum phase (1, 2, 3) and **auto-saves a clean
single-panel episode-vs-reward histogram per phase**, plus a 3-panel
phase-comparison chart for quick visual comparison with the trained
agent's plots from ``ER_MAP/plotting.py``.

Output layout under ``--output-dir`` (default: ``baseline_eval/``)::

    baseline_eval/
    ├── baseline_results.json              # raw per-episode records (all phases)
    ├── baseline_phase1_rewards.png        # Phase 1 episode-vs-reward histogram
    ├── baseline_phase2_rewards.png        # Phase 2 episode-vs-reward histogram
    ├── baseline_phase3_rewards.png        # Phase 3 episode-vs-reward histogram
    └── baseline_phases_comparison.png     # win-rate / avg-reward / fatal-rate per phase

Usage::

    # All 3 phases, 20 episodes each (default — auto-saves all plots):
    python -u -m ER_MAP.evaluate_baseline

    # Single phase only:
    python -u -m ER_MAP.evaluate_baseline --phase 1 --episodes-per-phase 20

Requires: ``GROQ_NURSE_API_KEY`` and ``GROQ_PATIENT_API_KEY`` (and
optionally ``GROQ_DOCTOR_API_KEY`` — falls back to patient key) in env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import mlflow

# Force unbuffered output so terminal logs stream live during long runs
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Demo-key auto-load (mirrors dashboard.py so this script "just runs"
# without forcing the user to set 5 env vars by hand). The `setdefault`
# semantics mean any value the user already exports in their shell wins.
# ---------------------------------------------------------------------------

# Default *non-secret* values. API keys are intentionally empty here and
# must be provided either via the user's shell or a git-ignored `.env`
# file at the repo root (see `.env.example`). Never hardcode real keys
# in this dict — GitHub push protection will reject the commit.
_DEMO_KEYS = {
    "GROQ_DOCTOR_API_KEY":        "",
    "GROQ_NURSE_API_KEY":         "",
    "GROQ_PATIENT_API_KEY":       "",
    "GROQ_EMPATHY_JUDGE_API_KEY": "",
    "GROQ_MEDICAL_JUDGE_API_KEY": "",
    "ERMAP_DOCTOR_MODEL":         "llama-3.1-8b-instant",
    "ERMAP_NURSE_MODEL":          "llama-3.1-8b-instant",
    "ERMAP_PATIENT_MODEL":        "llama-3.1-8b-instant",
}


def _load_dotenv_into_environ(*candidates: str) -> str:
    """Zero-dependency .env loader; identical contract to the one in
    `ER_MAP/dashboard.py`. See that file for full docstring."""
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
                    if key and not os.environ.get(key):
                        os.environ[key] = val
            return cand
        except Exception:  # noqa: BLE001
            continue
    return ""


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_load_dotenv_into_environ(
    os.path.join(_REPO_ROOT, ".env"),
    os.path.join(_HERE, ".env"),
)

_FORCE_DEMO = os.environ.get("ERMAP_USE_DEMO_KEYS", "").strip().lower() in {"1", "true", "yes"}
for _k, _v in _DEMO_KEYS.items():
    if _FORCE_DEMO or not os.environ.get(_k):
        os.environ[_k] = _v


# ---------------------------------------------------------------------------
# Doctor brain — re-uses the simple LLM client from ER_MAP.evaluate
# ---------------------------------------------------------------------------

from ER_MAP.evaluate import (
    DoctorBrain,
    print_doctor_action,
    print_observation,
)


# ---------------------------------------------------------------------------
# Phase-aware episode runner
# ---------------------------------------------------------------------------

def run_episode(env, doctor, episode_num: int, phase: int,
                *, max_steps: int = 30, slow_print: float = 0.6) -> Dict[str, Any]:
    """
    Run a single episode pinned to ``phase`` (1, 2, or 3) and return a
    summary dict suitable for the baseline plotter.
    """
    doctor.reset()
    obs, info = env.reset(options={"phase": phase})
    gt = env.ground_truth
    disease = info.get("ground_truth_disease", "???")
    difficulty = gt.get("difficulty", "random")
    p = gt["patient"]
    n = gt["nurse"]

    print(f"    Phase:      {phase} ({_phase_name(phase)})", flush=True)
    print(f"    Disease:    {disease}", flush=True)
    print(f"    Difficulty: {difficulty}", flush=True)
    print(f"    Patient:    compliance={p['compliance']}, comm={p['communication']}, "
          f"literacy={p['literacy']}", flush=True)
    print(f"    Nurse:      exp={n['experience']}, bandwidth={n['bandwidth']}, "
          f"empathy={n['empathy']}", flush=True)
    print(f"    Correct Tx: {gt['disease']['correct_treatment'][:80]}", flush=True)
    print(f"    {'~' * 60}", flush=True)
    print_observation(obs)

    total_reward = 0.0
    steps = 0
    outcome = "TRUNCATED"

    while True:
        steps += 1
        if slow_print > 0:
            time.sleep(slow_print)

        action_str = doctor.decide(obs)
        print(f"    Step {steps}:", flush=True)
        print_doctor_action(action_str, steps)

        obs, reward, done, truncated, _step_info = env.step(action_str)
        total_reward += reward
        print(f"      REWARD  | {reward:+.2f}  (total: {total_reward:+.2f})", flush=True)
        print_observation(obs)

        if done:
            try:
                obs_data = json.loads(obs)
                event = obs_data.get("event", "")
                if "win" in event:           outcome = "WIN"
                elif "partial" in event:     outcome = "PARTIAL"
                elif "fatal" in event:       outcome = "FATAL"
                elif "ama" in event:         outcome = "AMA"
                elif "incorrect" in event:   outcome = "WRONG"
                else:                        outcome = event or "done"
            except Exception:
                outcome = "done"
            break
        if truncated:
            outcome = "TRUNCATED"
            break
        if steps >= max_steps:
            outcome = "MAX_STEPS"
            break

    return {
        "phase":          phase,
        "phase_name":     _phase_name(phase),
        "episode":        episode_num,
        "disease":        disease,
        "difficulty":     difficulty,
        "compliance":     p["compliance"],
        "communication": p["communication"],
        "outcome":        outcome,
        "total_reward":   round(total_reward, 3),
        "steps":          steps,
    }


def _phase_name(phase: int) -> str:
    return {
        1: "Tool Mastery",
        2: "Clinical Reasoning",
        3: "Empathetic Negotiation",
    }.get(int(phase), f"Phase {phase}")


# ---------------------------------------------------------------------------
# Plot bridge — calls the clean per-phase histogram in ER_MAP.plotting
# ---------------------------------------------------------------------------

def save_phase_plot(results: List[Dict[str, Any]], phase: int, out_dir: str) -> str:
    """Auto-save a clean single-panel episode-vs-reward histogram for one phase."""
    from ER_MAP.plotting import plot_baseline_phase_histogram
    out_path = Path(out_dir) / f"baseline_phase{phase}_rewards.png"
    return plot_baseline_phase_histogram(
        results, phase_id=phase, out_path=str(out_path),
        title_suffix="Baseline Doctor (no RL)",
    )


def save_comparison_plot(results_by_phase: Dict[int, List[Dict[str, Any]]],
                         out_dir: str) -> str:
    from ER_MAP.plotting import plot_baseline_phase_comparison
    out_path = Path(out_dir) / "baseline_phases_comparison.png"
    return plot_baseline_phase_comparison(results_by_phase, str(out_path))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_phase_summary(phase: int, results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    n = len(results)
    wins  = sum(1 for r in results if r["outcome"] == "WIN")
    ama   = sum(1 for r in results if r["outcome"] == "AMA")
    wrong = sum(1 for r in results if r["outcome"] in ("WRONG", "FATAL"))
    avg_r = sum(r["total_reward"] for r in results) / n
    avg_s = sum(r["steps"] for r in results) / n

    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"  PHASE {phase} ({_phase_name(phase)}) — {n} episodes", flush=True)
    print("=" * 70, flush=True)
    print(f"  Win rate:    {wins}/{n} ({100 * wins / n:.0f}%)", flush=True)
    print(f"  AMA rate:    {ama}/{n} ({100 * ama / n:.0f}%)", flush=True)
    print(f"  Wrong/Fatal: {wrong}/{n} ({100 * wrong / n:.0f}%)", flush=True)
    print(f"  Avg reward:  {avg_r:+.2f}", flush=True)
    print(f"  Avg steps:   {avg_s:.1f}", flush=True)
    print("=" * 70, flush=True)

    # MLflow logging
    try:
        # Configure fast timeouts for MLflow connection requests to prevent hanging in CI environments
        if "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ:
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
        if "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ:
            os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"

        resolved_uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"

        # Check if URI is an HTTP/HTTPS endpoint and if it's reachable
        if resolved_uri.startswith("http://") or resolved_uri.startswith("https://"):
            import socket
            from urllib.parse import urlparse
            parsed = urlparse(resolved_uri)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (80 if parsed.scheme == "http" else 443)
            try:
                # Quick socket connection check with a 1-second timeout
                with socket.create_connection((host, port), timeout=1.0):
                    pass
            except (socket.timeout, ConnectionRefusedError, OSError) as conn_err:
                print(
                    f"  Warning: MLflow tracking server at {resolved_uri} is unreachable ({conn_err}). "
                    f"Falling back to local SQLite database (sqlite:///mlflow.db) to ensure logging runs successfully.",
                    flush=True
                )
                resolved_uri = "sqlite:///mlflow.db"

        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment("ER-MAP_Baseline_Evaluation")
        with mlflow.start_run(run_name=f"Phase_{phase}_{_phase_name(phase).replace(' ', '_')}"):
            mlflow.log_metric("win_rate", wins / n)
            mlflow.log_metric("ama_rate", ama / n)
            mlflow.log_metric("fatal_rate", wrong / n)
            mlflow.log_metric("avg_reward", avg_r)
            mlflow.log_metric("avg_steps", avg_s)
        print("  MLflow logging completed successfully.", flush=True)
    except Exception as e:
        print(f"  Warning: MLflow logging failed: {e}. Continuing without logging.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-phase baseline evaluation for the ER-MAP Doctor.")
    parser.add_argument("--episodes-per-phase", type=int, default=20,
                        help="Episodes per phase (default: 20)")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="If 0 (default), runs phases 1, 2, 3. "
                             "Otherwise restricts to a single phase.")
    parser.add_argument("--output-dir", type=str, default="baseline_eval",
                        help="Where to write JSON + PNGs")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max steps per episode")
    parser.add_argument("--slow-print", type=float, default=0.6,
                        help="Seconds to sleep between Doctor turns "
                             "(human-readable terminal). Set 0 for fast.")
    args = parser.parse_args()

    nurse_key   = ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", ""))
    patient_key = ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", ""))
    doctor_key  = ((os.environ.get("GROQ_DOCTOR_API_KEY") or os.environ.get("doctor")) or os.environ.get("doctor", "")) or patient_key
    empathy_key = ((os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY") or os.environ.get("empathy")) or os.environ.get("empathy", "")) or nurse_key
    medical_key = ((os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY") or os.environ.get("medical")) or os.environ.get("medical", "")) or nurse_key

    if not nurse_key or not patient_key:
        print("ERROR: Set GROQ_NURSE_API_KEY and GROQ_PATIENT_API_KEY (and "
              "optionally GROQ_DOCTOR_API_KEY) in your environment.", flush=True)
        return 1

    target_phases = [args.phase] if args.phase else [1, 2, 3]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70, flush=True)
    print(f"  ER-MAP BASELINE EVAL — {len(target_phases)} phase(s) × "
          f"{args.episodes_per_phase} episodes each", flush=True)
    print("=" * 70, flush=True)
    print(f"  Doctor:   Llama (baseline, no RL)   key=...{doctor_key[-4:]}", flush=True)
    print(f"  Nurse:    Llama (LIVE)              key=...{nurse_key[-4:]}", flush=True)
    print(f"  Patient:  Llama (LIVE)              key=...{patient_key[-4:]}", flush=True)
    print(f"  Output:   {out_dir.resolve()}", flush=True)
    print("=" * 70, flush=True)

    from ER_MAP.envs.triage_env import TriageEnv
    env = TriageEnv(
        nurse_api_key=nurse_key,
        patient_api_key=patient_key,
        empathy_judge_api_key=empathy_key,
        medical_judge_api_key=medical_key,
    )

    # Build the Doctor's resilient client chain across every available
    # (key, model) pair. Putting 8B-Instant FIRST means the Doctor
    # uses its own daily TPD pool by default; if that pool is
    # exhausted the chain spills over to the 70B pool (separate quota)
    # on the same key, then to OTHER keys. This effectively gives the
    # Doctor ~5x the usable tokens before it runs out.
    chain = []
    for label, key in [("doctor", doctor_key), ("nurse", nurse_key),
                       ("patient", patient_key), ("empathy", empathy_key),
                       ("medical", medical_key)]:
        if not key:
            continue
        chain.append({"key": key, "model": "llama-3.1-8b-instant", "label": f"{label}/8b"})
        chain.append({"key": key, "model": "llama-3.3-70b-versatile", "label": f"{label}/70b"})
    doctor = DoctorBrain(fallback_chain=chain)
    print(f"  Doctor:   {len(doctor._chain)} (key, model) pairs in fallback chain", flush=True)

    all_results: List[Dict[str, Any]] = []
    results_by_phase: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for phase in target_phases:
        print("\n" + "#" * 70, flush=True)
        print(f"#  PHASE {phase} — {_phase_name(phase)}", flush=True)
        print("#" * 70, flush=True)

        for ep in range(1, args.episodes_per_phase + 1):
            print(f"\n  {'=' * 60}", flush=True)
            print(f"  PHASE {phase} | EPISODE {ep}/{args.episodes_per_phase}", flush=True)
            print(f"  {'=' * 60}", flush=True)
            try:
                rec = run_episode(
                    env, doctor, episode_num=ep, phase=phase,
                    max_steps=args.max_steps, slow_print=args.slow_print,
                )
            except Exception as e:
                print(f"    [ERR] episode failed: {e}", flush=True)
                rec = {
                    "phase": phase, "phase_name": _phase_name(phase),
                    "episode": ep, "disease": "ERROR", "difficulty": "?",
                    "compliance": "?", "communication": "?",
                    "outcome": "ERROR", "total_reward": -2.0, "steps": 0,
                }
            results_by_phase[phase].append(rec)
            all_results.append(rec)
            icon = {"WIN": "[OK]", "AMA": "[!!]", "WRONG": "[XX]",
                    "FATAL": "[XX]"}.get(rec["outcome"], "[--]")
            print(f"    {icon} {rec['outcome']:9s} | reward {rec['total_reward']:+.2f} | "
                  f"steps {rec['steps']}", flush=True)

        # AUTO-SAVE the clean episode-vs-reward histogram for this phase
        # immediately after the phase finishes — we never lose data even
        # if a later phase crashes.
        png_path = save_phase_plot(results_by_phase[phase], phase, str(out_dir))
        print_phase_summary(phase, results_by_phase[phase])
        print(f"\n  >>> Saved: {png_path}", flush=True)

    env.close()

    # Save the full raw JSON record for re-plotting / cross-checks later
    results_path = out_dir / "baseline_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw results: {results_path.resolve()}", flush=True)

    # Cross-phase comparison only makes sense if we ran ≥2 phases
    if len(results_by_phase) >= 2:
        cmp_path = save_comparison_plot(dict(results_by_phase), str(out_dir))
        print(f"  Comparison: {cmp_path}", flush=True)

    print("\n  Baseline evaluation complete.\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

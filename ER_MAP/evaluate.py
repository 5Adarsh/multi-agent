"""
ER_MAP/evaluate.py
==================
Run N episodes with an LLM Doctor brain, show full conversations,
collect metrics, and plot reward curves.

Usage:
    cd d:/Meta_Finals
    python -u -m ER_MAP.evaluate --episodes 30
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, Any, List, Optional

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Doctor LLM Brain
# ---------------------------------------------------------------------------

DOCTOR_SYSTEM_PROMPT = """You are an expert emergency room doctor performing triage. You must diagnose and treat the patient.

## Available Tools (respond with STRICT JSON)

1. speak_to: {"thought":"...","tool":"speak_to","target":"nurse or patient","message":"..."}
2. order_lab: {"thought":"...","tool":"order_lab","target":"nurse","test_name":"lab name"}
3. read_soap: {"thought":"...","tool":"read_soap","section":"Subjective or Objective or ALL"}
4. update_soap: {"thought":"...","tool":"update_soap","section":"Assessment","content":"your diagnosis"}
5. terminal_discharge: {"thought":"...","tool":"terminal_discharge","treatment":"your treatment plan"}

## Strategy
- First: Use read_soap to review the patient's HPI, medical history, allergies, and physical exam
- Ask nurse to assess patient and get vitals
- Order relevant labs based on symptoms (e.g. troponin, D-dimer, BMP, ABG, CBC, ECG, CXR, CSF, tryptase, urine_tox, CT_head, CT_abdomen, CT_angio, CK, peak_flow)
- Update Assessment with your working diagnosis before discharge
- Check Allergies before prescribing medications
- Discharge with treatment when you have enough evidence
- Be concise with patients. Use simple language.

RESPOND ONLY WITH VALID JSON."""


class DoctorBrain:
    """
    Resilient Doctor LLM client.

    - Accepts a single key (legacy) OR a list of (api_key, model) tuples.
    - On 401 (invalid key) or 429 (rate-limited), marks that
      (key, model) pair as dead and silently advances to the next pair
      so the episode keeps progressing instead of looping on stale data.
    - When *every* pair is dead, falls back to a deterministic clinical
      decision tree (`_smart_fallback_action`) that drives the episode
      toward a sensible discharge instead of spamming "Update me on the
      patient" 30 times and burning Nurse/Patient tokens.
    """

    def __init__(self, api_key: str = "", model: str = "llama-3.1-8b-instant",
                 fallback_chain: Optional[List[Dict[str, str]]] = None):
        from groq import Groq
        self._Groq = Groq

        # Build the (key, model) chain. The Doctor's *primary* model is
        # 8B-Instant: it has its own daily TPD pool, separate from the
        # 70B pool used by Nurse/Patient/Judges. Rotating across both
        # pools effectively gives ~5x more headroom than a single key.
        if fallback_chain is None:
            fallback_chain = []
            if api_key:
                fallback_chain.append({"key": api_key, "model": model})
        self._chain: List[Dict[str, Any]] = []
        seen = set()
        for entry in fallback_chain:
            k = (entry["key"], entry["model"])
            if not entry["key"] or k in seen:
                continue
            seen.add(k)
            self._chain.append({
                "key":   entry["key"],
                "model": entry["model"],
                "client": Groq(api_key=entry["key"]),
                "dead":  False,
                "label": entry.get("label", entry["key"][-4:]),
            })
        if not self._chain:
            raise ValueError("DoctorBrain: empty fallback chain")

        # Keep .client / .model for backward compat with any caller
        # that still pokes at them (rare, but safer to expose).
        self.client = self._chain[0]["client"]
        self.model  = self._chain[0]["model"]

        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]
        self._consecutive_failures = 0

    def reset(self):
        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]
        self._consecutive_failures = 0

    def _alive_clients(self) -> List[Dict[str, Any]]:
        return [c for c in self._chain if not c["dead"]]

    @staticmethod
    def _is_dead_error(err: Exception) -> bool:
        """Detect Groq 401 (invalid key) and 429 (rate-limited)."""
        s = str(err)
        return "401" in s or "429" in s or "rate_limit" in s.lower() \
               or "invalid_api_key" in s.lower()

    def _smart_fallback_action(self) -> str:
        """
        Deterministic clinical decision tree used when every Groq client
        in the chain is dead. Drives the episode toward a sensible
        terminal state instead of looping on "Give me an update".
        """
        depth = self._consecutive_failures
        if depth <= 1:
            action = {
                "thought": "Fallback (no LLM available): start by reading the SOAP note",
                "tool": "read_soap", "section": "ALL",
            }
        elif depth == 2:
            action = {
                "thought": "Fallback: ask nurse for vitals and a focused exam",
                "tool": "speak_to", "target": "nurse",
                "message": "Please get full vitals (HR/BP/RR/SpO2/Temp) and report any focal findings.",
            }
        elif depth == 3:
            action = {
                "thought": "Fallback: order a broad initial lab panel",
                "tool": "order_lab", "target": "nurse",
                "test_name": "CBC, BMP, lactate, troponin, ECG",
            }
        elif depth == 4:
            action = {
                "thought": "Fallback: document working assessment before discharge",
                "tool": "update_soap", "section": "Assessment",
                "content": "Working dx pending; treating empirically based on vitals + chief complaint.",
            }
        else:
            # After 5+ consecutive failures, end the episode rather than
            # waste any more Nurse/Patient tokens. Use a safe empirical
            # treatment that covers the most common emergent diagnoses.
            action = {
                "thought": "Fallback: empirical discharge to terminate stuck episode",
                "tool": "terminal_discharge",
                "treatment": "Empirical: O2 + IV fluids + monitor; ICU admit if unstable.",
            }
        return json.dumps(action)

    def decide(self, observation: str) -> str:
        self.history.append({"role": "user", "content": f"Observation:\n{observation}"})
        if len(self.history) > 17:
            self.history = [self.history[0]] + self.history[-16:]

        response = None
        for entry in self._alive_clients():
            try:
                completion = entry["client"].chat.completions.create(
                    model=entry["model"],
                    messages=self.history,
                    temperature=0.6,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                response = completion.choices[0].message.content or ""
                self._consecutive_failures = 0
                break
            except Exception as e:
                if self._is_dead_error(e):
                    print(f"      [Doctor: key=...{entry['label']} "
                          f"model={entry['model']} -> DEAD ({type(e).__name__}); "
                          f"trying next]", flush=True)
                    entry["dead"] = True
                    continue
                # Non-fatal error (network blip, JSON parse, etc.) — give
                # up on this turn but DON'T mark the key dead.
                print(f"      [Doctor API Error: {e}]", flush=True)
                break

        if response is None:
            self._consecutive_failures += 1
            alive = len(self._alive_clients())
            print(f"      [Doctor: all {len(self._chain)} clients dead "
                  f"({alive} alive). Smart fallback depth={self._consecutive_failures}]",
                  flush=True)
            response = self._smart_fallback_action()

        self.history.append({"role": "assistant", "content": response})
        return response


# ---------------------------------------------------------------------------
# Conversation Printer
# ---------------------------------------------------------------------------

def print_doctor_action(action_str: str, step: int):
    try:
        a = json.loads(action_str)
    except json.JSONDecodeError:
        print(f"      DOCTOR: [invalid JSON]", flush=True)
        return
    tool = a.get("tool", "?")
    print(f"      DOCTOR  | {a.get('thought', '')[:80]}", flush=True)
    if tool == "speak_to":
        print(f"              | -> {a.get('target','')}: \"{a.get('message','')}\"", flush=True)
    elif tool == "order_lab":
        print(f"              | -> order_lab: {a.get('test_name','')}", flush=True)
    elif tool == "read_soap":
        print(f"              | -> read_soap: section={a.get('section','')}", flush=True)
    elif tool == "update_soap":
        print(f"              | -> update_soap: section={a.get('section','')} content={a.get('content','')[:80]}", flush=True)
    elif tool == "terminal_discharge":
        print(f"              | -> DISCHARGE: {a.get('treatment','')[:100]}", flush=True)
    else:
        print(f"              | -> {tool}: {action_str[:100]}", flush=True)


def print_observation(obs_str: str, indent="      "):
    try:
        obs = json.loads(obs_str)
    except json.JSONDecodeError:
        print(f"{indent}ENV: {obs_str[:100]}", flush=True)
        return
    event = obs.get("event", "unknown")
    if event == "episode_start":
        print(f"{indent}ENV     | New case. Nurse: {obs.get('nurse_experience')}", flush=True)
    elif event == "nurse_report":
        print(f"{indent}NURSE   | \"{obs.get('nurse_message', '')[:120]}\"", flush=True)
        print(f"{indent}        | nurse_status={obs.get('nurse_status','')} patient_status={obs.get('patient_status','')}", flush=True)
        for ex in obs.get("internal_exchanges", []):
            if "nurse_said" in ex:
                print(f"{indent}  N->P  | \"{ex.get('nurse_said','')[:100]}\"", flush=True)
                print(f"{indent}  P->N  | \"{ex.get('patient_said','')[:100]}\"", flush=True)
            elif "nurse_action" in ex:
                print(f"{indent}  N-act | {ex.get('nurse_action','')} -> {ex.get('result','')[:80]}", flush=True)
    elif event == "patient_response":
        print(f"{indent}PATIENT | \"{obs.get('patient_message', '')[:120]}\"", flush=True)
        print(f"{indent}        | status={obs.get('patient_status','')}", flush=True)
    elif event == "lab_result":
        tag = " (DUP)" if obs.get("redundant") else ""
        print(f"{indent}LAB     | [{obs.get('test_name','')}]{tag}: {obs.get('result','')[:100]}", flush=True)
    elif event == "soap_read":
        print(f"{indent}EMR     | read_soap (section={obs.get('section','')})", flush=True)
        content = obs.get("content", "")
        if isinstance(content, dict):
            snippet = ", ".join(f"{k}: {str(v)[:30]}" for k, v in content.items())
            print(f"{indent}        | {snippet[:120]}", flush=True)
        else:
            print(f"{indent}        | {str(content)[:120]}", flush=True)
    elif event == "soap_updated":
        print(f"{indent}EMR     | SOAP updated (section={obs.get('section','')})", flush=True)
        print(f"{indent}        | message: {obs.get('message','')[:120]}", flush=True)
    elif event == "terminal_win":
        print(f"{indent}RESULT  | >>> WIN! Patient stabilized. <<<", flush=True)
    elif event == "terminal_fatal":
        print(f"{indent}RESULT  | >>> FATAL! Patient died. <<<", flush=True)
    elif event == "terminal_incorrect":
        print(f"{indent}RESULT  | >>> WRONG treatment. Correct: {obs.get('correct_treatment','')[:80]} <<<", flush=True)
    elif event == "terminal_ama":
        print(f"{indent}RESULT  | >>> AMA! Patient left: \"{obs.get('patient_message','')[:80]}\" <<<", flush=True)
    elif event == "system_error":
        print(f"{indent}ERROR   | {obs.get('message','')[:100]}", flush=True)
    else:
        print(f"{indent}EVENT   | {event}: {obs_str[:120]}", flush=True)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_episode(env, doctor, episode_num: int) -> Dict[str, Any]:
    doctor.reset()
    obs, info = env.reset()
    gt = env.ground_truth
    disease = info.get("ground_truth_disease", "???")
    difficulty = gt.get("difficulty", "random")
    p = gt["patient"]
    n = gt["nurse"]

    # Print episode header
    print(f"    Disease:    {disease}", flush=True)
    print(f"    Difficulty: {difficulty}", flush=True)
    print(f"    Patient:    compliance={p['compliance']}, comm={p['communication']}, literacy={p['literacy']}", flush=True)
    print(f"    Nurse:      exp={n['experience']}, bandwidth={n['bandwidth']}, empathy={n['empathy']}", flush=True)
    print(f"    Correct Tx: {gt['disease']['correct_treatment'][:80]}", flush=True)
    print(f"    {'~'*60}", flush=True)

    print_observation(obs)

    total_reward = 0.0
    steps = 0
    outcome = "truncated"

    while True:
        steps += 1
        time.sleep(1.2)

        action_str = doctor.decide(obs)
        print(f"    Step {steps}:", flush=True)
        print_doctor_action(action_str, steps)

        obs, reward, done, truncated, step_info = env.step(action_str)
        total_reward += reward
        print(f"      REWARD  | {reward:+.2f}  (total: {total_reward:+.2f})", flush=True)
        print_observation(obs)

        if done:
            try:
                obs_data = json.loads(obs)
                event = obs_data.get("event", "")
                if "win" in event: outcome = "WIN"
                elif "fatal" in event: outcome = "FATAL"
                elif "ama" in event: outcome = "AMA"
                elif "incorrect" in event: outcome = "WRONG"
                else: outcome = event
            except:
                outcome = "done"
            break
        if truncated:
            outcome = "TRUNCATED"
            break
        if steps >= 30:
            outcome = "MAX_STEPS"
            break

    return {
        "episode": episode_num, "disease": disease,
        "difficulty": difficulty, "compliance": p["compliance"],
        "communication": p["communication"], "outcome": outcome,
        "total_reward": round(total_reward, 2), "steps": steps,
    }


def plot_reward_curve(results: List[Dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed. Skipping plot.", flush=True)
        return

    episodes = [r["episode"] for r in results]
    rewards = [r["total_reward"] for r in results]
    outcomes = [r["outcome"] for r in results]

    window = min(5, len(rewards))
    rolling_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling_avg.append(sum(rewards[start:i+1]) / (i - start + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")

    ax1.set_facecolor("#161b22")
    colors = []
    for o in outcomes:
        if o == "WIN": colors.append("#2ea043")
        elif o == "AMA": colors.append("#f0883e")
        elif o in ("FATAL", "WRONG"): colors.append("#f85149")
        else: colors.append("#8b949e")

    ax1.bar(episodes, rewards, color=colors, alpha=0.6, width=0.8, label="Episode Reward")
    ax1.plot(episodes, rolling_avg, color="#58a6ff", linewidth=2.5, label=f"Rolling Avg (window={window})", zorder=5)
    ax1.axhline(y=0, color="#484f58", linewidth=1, linestyle="--")
    ax1.axhline(y=2.0, color="#2ea043", linewidth=1, linestyle=":", alpha=0.5, label="Win threshold (+2.0)")
    ax1.axhline(y=-1.5, color="#f85149", linewidth=1, linestyle=":", alpha=0.5, label="AMA penalty (-1.5)")
    ax1.set_xlabel("Episode", color="#c9d1d9", fontsize=12)
    ax1.set_ylabel("Total Reward", color="#c9d1d9", fontsize=12)
    ax1.set_title("ER-MAP: LLM Doctor Reward Curve (Baseline - No RL Training)",
                   color="#f0f6fc", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#484f58", labelcolor="#c9d1d9")
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_color("#484f58")

    ax2.set_facecolor("#161b22")
    outcome_types = ["WIN", "AMA", "WRONG", "FATAL", "TRUNCATED", "MAX_STEPS"]
    outcome_colors = ["#2ea043", "#f0883e", "#f85149", "#da3633", "#8b949e", "#6e7681"]
    outcome_counts = [sum(1 for o in outcomes if o == t) for t in outcome_types]
    bars = ax2.barh(outcome_types, outcome_counts, color=outcome_colors, alpha=0.8)
    for bar, count in zip(bars, outcome_counts):
        if count > 0:
            ax2.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                     str(count), va="center", color="#c9d1d9", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Count", color="#c9d1d9", fontsize=11)
    ax2.set_title("Outcome Distribution", color="#c9d1d9", fontsize=12, pad=10)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_color("#484f58")

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Reward curve saved to: {output_path}", flush=True)


def print_summary(results: List[Dict]):
    total = len(results)
    wins = sum(1 for r in results if r["outcome"] == "WIN")
    ama = sum(1 for r in results if r["outcome"] == "AMA")
    wrong = sum(1 for r in results if r["outcome"] in ("WRONG", "FATAL"))
    avg_reward = sum(r["total_reward"] for r in results) / total
    avg_steps = sum(r["steps"] for r in results) / total

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  EVALUATION SUMMARY ({total} episodes)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Win Rate:       {wins}/{total} ({100*wins/total:.0f}%)", flush=True)
    print(f"  AMA Rate:       {ama}/{total} ({100*ama/total:.0f}%)", flush=True)
    print(f"  Wrong/Fatal:    {wrong}/{total} ({100*wrong/total:.0f}%)", flush=True)
    print(f"  Avg Reward:     {avg_reward:+.2f}", flush=True)
    print(f"  Avg Steps:      {avg_steps:.1f}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    diseases = {}
    for r in results:
        d = r["disease"]
        if d not in diseases:
            diseases[d] = {"wins": 0, "total": 0, "reward_sum": 0}
        diseases[d]["total"] += 1
        diseases[d]["reward_sum"] += r["total_reward"]
        if r["outcome"] == "WIN":
            diseases[d]["wins"] += 1

    print("  PER-DISEASE BREAKDOWN:", flush=True)
    print(f"  {'Disease':35s} {'Win':>5s} {'Total':>5s} {'Rate':>6s} {'Avg Rwd':>8s}", flush=True)
    print("  " + "-" * 62, flush=True)
    for d, stats in sorted(diseases.items()):
        rate = f"{100*stats['wins']/stats['total']:.0f}%" if stats["total"] > 0 else "N/A"
        avg = stats["reward_sum"] / stats["total"]
        print(f"  {d:35s} {stats['wins']:>5d} {stats['total']:>5d} {rate:>6s} {avg:>+8.2f}", flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ER-MAP Evaluation Runner")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--output", type=str, default="reward_curve.png", help="Output plot path")
    args = parser.parse_args()

    from ER_MAP.envs.triage_env import TriageEnv

    nurse_key = ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", ""))
    patient_key = ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", ""))
    doctor_key = ((os.environ.get("GROQ_DOCTOR_API_KEY") or os.environ.get("doctor")) or os.environ.get("doctor", "")) or patient_key

    if not nurse_key or not patient_key:
        print("ERROR: Set GROQ_NURSE_API_KEY and GROQ_PATIENT_API_KEY", flush=True)
        return 1

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  ER-MAP EVALUATION: {args.episodes} episodes with LLM Doctor", flush=True)
    print("=" * 70, flush=True)
    print(f"  Doctor:  Llama-3.1-8B (unmodified baseline)", flush=True)
    print(f"  Nurse:   Llama-3.1-8B (LIVE)", flush=True)
    print(f"  Patient: Llama-3.1-8B (LIVE)", flush=True)
    print(f"  Diseases: 15 | Persona combos: 933,120", flush=True)
    print("=" * 70, flush=True)

    env = TriageEnv(nurse_api_key=nurse_key, patient_api_key=patient_key)
    doctor = DoctorBrain(api_key=doctor_key)

    results = []

    for ep in range(1, args.episodes + 1):
        print(flush=True)
        print(f"  {'='*60}", flush=True)
        print(f"  EPISODE {ep}/{args.episodes}", flush=True)
        print(f"  {'='*60}", flush=True)
        try:
            result = run_episode(env, doctor, ep)
            results.append(result)
            icon = {"WIN": "[OK]", "AMA": "[!!]", "WRONG": "[XX]", "FATAL": "[XX]"}.get(result["outcome"], "[--]")
            print(f"    {icon} OUTCOME: {result['outcome']:8s} | Reward: {result['total_reward']:+.2f} | Steps: {result['steps']}", flush=True)
        except Exception as e:
            print(f"    [ERR] Episode {ep} failed: {e}", flush=True)
            results.append({
                "episode": ep, "disease": "ERROR", "difficulty": "?",
                "compliance": "?", "communication": "?",
                "outcome": "ERROR", "total_reward": -2.0, "steps": 0
            })

    env.close()

    # Save results
    out_dir = os.path.dirname(args.output) or "."
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results saved to: {results_path}", flush=True)

    print_summary(results)
    plot_reward_curve(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

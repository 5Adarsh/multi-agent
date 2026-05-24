"""
ER_MAP/autoplay.py
==================
Automated play: An LLM Doctor plays against LLM Nurse & Patient.
All three agents speak aloud with emotion-induced neural TTS.

Usage:
    python -u -m ER_MAP.autoplay
    python -u -m ER_MAP.autoplay --no-voice
    python -u -m ER_MAP.autoplay --model llama-3.3-70b-versatile
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, Any

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Doctor LLM Brain
# ---------------------------------------------------------------------------

DOCTOR_SYSTEM_PROMPT = """You are an expert emergency room doctor performing triage. You must diagnose and treat the patient by interacting with a nurse and the patient.

## Available Tools
You MUST respond with a valid JSON object using one of these tools:

1. speak_to - Talk to nurse or patient
   {"thought": "...", "tool": "speak_to", "target": "nurse or patient", "message": "..."}

2. order_lab - Order a lab test
   {"thought": "...", "tool": "order_lab", "target": "nurse", "test_name": "lab name"}

3. read_soap - Read the patient's SOAP note (medical record)
   {"thought": "...", "tool": "read_soap", "section": "Subjective or Objective or ALL"}

4. update_soap - Update Assessment or Plan in the SOAP note
   {"thought": "...", "tool": "update_soap", "section": "Assessment or Plan", "content": "..."}

5. terminal_discharge - End the case with your treatment plan (ONLY after completing workflow)
   {"thought": "...", "tool": "terminal_discharge", "treatment": "detailed treatment plan"}

## MANDATORY Clinical Workflow (follow this order)
Step 1: Use read_soap to review patient's medical history, HPI, allergies, medications
Step 2: Speak to the PATIENT directly — ask about chief complaint, onset, severity, aggravating/relieving factors
Step 3: Ask nurse to check vitals
Step 4: Order targeted labs based on symptoms and vitals
Step 5: Speak to the PATIENT again — follow-up questions based on results
Step 6: update_soap Assessment with your working diagnosis and clinical reasoning
Step 7: update_soap Plan with your treatment plan
Step 8: ONLY THEN terminal_discharge with detailed treatment

## CRITICAL RULES
- You MUST speak to the patient at least twice before discharging
- You MUST check Allergies in the SOAP note before prescribing medications
- You MUST document Assessment before discharging
- Do NOT rush — gather enough evidence first
- Use simple, empathetic language with patients
- Ask targeted questions: "Where is the pain?", "When did it start?", "Scale of 1-10?"

RESPOND ONLY WITH VALID JSON. No extra text."""


class DoctorBrain:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

    def decide(self, observation: str) -> str:
        self.history.append({"role": "user", "content": f"Observation:\n{observation}"})
        if len(self.history) > 17:
            self.history = [self.history[0]] + self.history[-16:]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.6,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            response = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [Doctor API Error: {e}]", flush=True)
            response = json.dumps({
                "thought": "API error, asking nurse",
                "tool": "speak_to", "target": "nurse",
                "message": "Can you give me an update?"
            })
        self.history.append({"role": "assistant", "content": response})
        return response


# ---------------------------------------------------------------------------
# Pretty Printers
# ---------------------------------------------------------------------------

def divider(char="=", w=70):
    print(char * w, flush=True)

def print_doctor(action_str: str, step: int):
    try:
        a = json.loads(action_str)
    except json.JSONDecodeError:
        print(f"  DOCTOR: [invalid JSON]", flush=True)
        return
    tool = a.get("tool", "?")
    print(f"\n  DOCTOR (Step {step}):", flush=True)
    print(f"    Thinking: {a.get('thought', '')[:100]}", flush=True)
    print(f"    Tool:     {tool}", flush=True)
    if tool == "speak_to":
        print(f"    To:       {a.get('target', '')}", flush=True)
        print(f"    Says:     \"{a.get('message', '')}\"", flush=True)
    elif tool == "order_lab":
        print(f"    Lab:      {a.get('test_name', '')}", flush=True)
    elif tool == "read_soap":
        print(f"    Section:  {a.get('section', '')}", flush=True)
    elif tool == "update_soap":
        print(f"    Section:  {a.get('section', '')}", flush=True)
        print(f"    Content:  {a.get('content', '')[:120]}", flush=True)
    elif tool == "terminal_discharge":
        print(f"    Treatment: {a.get('treatment', '')}", flush=True)
    else:
        print(f"    Action:   {action_str[:100]}", flush=True)


def print_obs(obs_str: str):
    try:
        obs = json.loads(obs_str)
    except json.JSONDecodeError:
        print(f"  ENV: {obs_str[:100]}", flush=True)
        return
    event = obs.get("event", "")
    if event == "episode_start":
        print(f"  ENV: New case. Nurse experience: {obs.get('nurse_experience')}", flush=True)
    elif event == "nurse_report":
        print(f"  NURSE says: \"{obs.get('nurse_message', '')[:150]}\"", flush=True)
        print(f"    Nurse status: {obs.get('nurse_status', '')} | Patient status: {obs.get('patient_status', '')}", flush=True)
        for ex in obs.get("internal_exchanges", []):
            if "nurse_said" in ex:
                print(f"      N->P: \"{ex.get('nurse_said','')[:120]}\"", flush=True)
                print(f"      P->N: \"{ex.get('patient_said','')[:120]}\"", flush=True)
                print(f"        Patient status: {ex.get('patient_status','')}", flush=True)
            elif "nurse_action" in ex:
                print(f"      N-action: {ex.get('nurse_action','')} -> {ex.get('result','')[:100]}", flush=True)
    elif event == "patient_response":
        print(f"  PATIENT says: \"{obs.get('patient_message', '')[:150]}\"", flush=True)
        print(f"    Patient status: {obs.get('patient_status', '')}", flush=True)
    elif event == "lab_result":
        dup = " (DUPLICATE!)" if obs.get("redundant") else ""
        print(f"  LAB [{obs.get('test_name','')}]{dup}: {obs.get('result','')[:120]}", flush=True)
    elif event == "soap_read":
        print(f"  EMR: Read SOAP note (section: {obs.get('section','')})", flush=True)
        content = obs.get("content", "")
        if isinstance(content, dict):
            snippet = ", ".join(f"{k}: {str(v)[:30]}" for k, v in content.items())
            print(f"    Content: {snippet[:120]}", flush=True)
        else:
            print(f"    Content: {str(content)[:120]}", flush=True)
    elif event == "soap_updated":
        print(f"  EMR: SOAP note updated (section: {obs.get('section','')})", flush=True)
        print(f"    Message: {obs.get('message','')[:120]}", flush=True)
    elif event == "terminal_win":
        print(f"  >>> CORRECT DIAGNOSIS! Patient stabilized! <<<", flush=True)
    elif event == "terminal_fatal":
        print(f"  >>> FATAL ERROR! Patient died! <<<", flush=True)
    elif event == "terminal_incorrect":
        print(f"  >>> WRONG treatment! Correct: {obs.get('correct_treatment','')} <<<", flush=True)
    elif event == "terminal_ama":
        print(f"  >>> PATIENT LEFT AMA: \"{obs.get('patient_message','')}\" <<<", flush=True)
    else:
        print(f"  ENV EVENT: {event} -> {obs_str[:120]}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ER-MAP Autoplay")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile",
                        help="Groq model for all agents (default: llama-3.3-70b-versatile)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable TTS voice output")
    args = parser.parse_args()

    from ER_MAP.envs.triage_env import TriageEnv

    nurse_key = ((os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("nurse")) or os.environ.get("nurse", ""))
    patient_key = ((os.environ.get("GROQ_PATIENT_API_KEY") or os.environ.get("patient")) or os.environ.get("patient", ""))
    doctor_key = ((os.environ.get("GROQ_DOCTOR_API_KEY") or os.environ.get("doctor")) or os.environ.get("doctor", "")) or patient_key

    if not nurse_key or not patient_key:
        print("ERROR: Set GROQ_NURSE_API_KEY and GROQ_PATIENT_API_KEY")
        return 1

    # Initialize TTS Engine
    tts = None
    if not args.no_voice:
        try:
            from ER_MAP.tts_engine import TTSEngine
            tts = TTSEngine()
            print(f"  Voice: {'ElevenLabs' if tts.use_elevenlabs else 'Edge-TTS'}", flush=True)
        except Exception as e:
            print(f"  [TTS init failed: {e}] Running without voice.", flush=True)

    print(flush=True)
    divider()
    print(f"  ER-MAP AUTOPLAY: LLM Doctor vs LLM Nurse & Patient", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Voice: {'ON' if tts else 'OFF'}", flush=True)
    divider()

    env = TriageEnv(nurse_api_key=nurse_key, patient_api_key=patient_key, model=args.model)
    obs, info = env.reset()
    doctor = DoctorBrain(api_key=doctor_key, model=args.model)

    gt = env.ground_truth
    print(f"\n  Disease:  {info.get('ground_truth_disease', '???')}", flush=True)
    print(f"  Difficulty: {gt.get('difficulty', 'random')}", flush=True)
    print(flush=True)
    print("  PATIENT PERSONA:", flush=True)
    for k, v in gt["patient"].items():
        print(f"    {k:20s} : {v}", flush=True)
    print(flush=True)
    print("  NURSE PERSONA:", flush=True)
    for k, v in gt["nurse"].items():
        print(f"    {k:20s} : {v}", flush=True)
    print(f"\n  Correct Treatment: {gt['disease']['correct_treatment']}", flush=True)
    divider("-")

    print_obs(obs)
    total_reward = 0.0
    step = 0

    while True:
        step += 1
        time.sleep(1.0)

        # Doctor decides
        action_str = doctor.decide(obs)
        print_doctor(action_str, step)

        # 🔊 Speak Doctor's action
        if tts:
            tts.speak_doctor_action(action_str, gt)

        # Environment step
        obs, reward, done, truncated, info = env.step(action_str)
        total_reward += reward
        print(f"\n  Reward: {reward:+.2f}  |  Total: {total_reward:+.2f}", flush=True)
        divider("-")
        print_obs(obs)

        # 🔊 Speak observation (Nurse/Patient responses)
        if tts:
            tts.speak_observation(obs, gt)

        if done or truncated or step >= 30:
            break

    print(flush=True)
    divider()
    print(f"  GAME OVER  |  Total Reward: {total_reward:+.2f}", flush=True)
    divider()
    print(f"  Disease:           {gt['disease']['true_disease']}", flush=True)
    print(f"  Correct Treatment: {gt['disease']['correct_treatment']}", flush=True)
    print(f"  Steps Taken:       {step}", flush=True)
    divider()

    if tts:
        tts.close()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

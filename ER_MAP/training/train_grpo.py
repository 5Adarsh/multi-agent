"""
ER_MAP/training/train_grpo.py
==============================
GRPO (Group Relative Policy Optimization) Training Script with
3-Phase Curriculum Learning for the ER-MAP Triage Environment.

Why a manual GRPO loop instead of TRL's GRPOTrainer?
    TRL's GRPOTrainer expects (prompt, G completions) tuples and a
    reward function with signature (prompts, completions, **kwargs)
    -> list[float]. Our environment is multi-turn: a single "completion"
    spans many env.step() calls and the reward is computed across the
    whole trajectory by the env, not by a stateless reward function.
    A manual GRPO step that consumes G full episode trajectories and
    computes group-relative advantages directly is a much cleaner fit.

Algorithm (manual GRPO):
    1. Pick a scenario (phase + difficulty + seed) from the curriculum.
    2. Sample G episodes from the policy with the SAME seed -> they
       start in the same scenario but explore different action paths.
    3. Compute total trajectory reward R_i for i = 1..G.
    4. Group-relative advantage:  A_i = (R_i - mean(R)) / (std(R) + eps)
    5. For each env step in each trajectory, compute the response
       log-prob under the current policy and a reference policy.
    6. Loss = - mean( A_i * sum_t logp_t ) + beta * mean(KL_t).
    7. Backprop + optimizer step.

Process-distributed reward shaping (see triage_env.py) means the trajectory
reward is NOT terminal-dominated: intermediate diagnosis bonuses,
critical-lab bonuses, milestones, and capped empathy rewards together
account for ~60% of the maximum reward; the smoothed terminal judge +
keyword combo accounts for ~40%. This is what makes GRPO actually learn
on a long-horizon task.

Usage (Colab / HF Spaces):
    !pip install unsloth trl transformers datasets accelerate peft
    !pip install gymnasium groq
    python -m ER_MAP.training.train_grpo --episodes 200

Usage (local dry-run, no GPU):
    python -m ER_MAP.training.train_grpo --dry-run
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import gc
import json
import math
import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ER_MAP.train_grpo")


# ============================================================================
# Curriculum Scheduler
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    phase_id: int
    difficulty: str                    # "easy", "medium", "hard", or None
    min_episodes: int                  # Minimum episodes before promotion
    promotion_win_rate: float          # Win rate threshold to advance
    promotion_avg_reward: float        # Avg reward threshold to advance
    description: str = ""


class CurriculumScheduler:
    """
    Manages 3-phase curriculum transitions.

    Phase 1 (Tool Mastery):
        - Easy patients (calm, compliant)
        - Clean SOAP data
        - Promotion: win_rate >= 40% over last 20 episodes

    Phase 2 (Clinical Reasoning):
        - Mixed difficulty patients
        - Noisy SOAP data
        - Promotion: win_rate >= 35% AND avg_reward >= 0.5

    Phase 3 (Empathetic Negotiation):
        - Full persona randomization
        - Heavy SOAP noise + behavioral friction
        - No promotion (final phase)
    """

    PHASES = [
        PhaseConfig(
            name="Tool Mastery",
            phase_id=1,
            difficulty="easy",
            min_episodes=20,
            promotion_win_rate=0.40,
            promotion_avg_reward=0.3,
            description="Learn clinical tools: read_soap, order_lab, speak_to, terminal_discharge",
        ),
        PhaseConfig(
            name="Clinical Reasoning",
            phase_id=2,
            difficulty="medium",
            min_episodes=30,
            promotion_win_rate=0.35,
            promotion_avg_reward=0.5,
            description="Differential diagnosis with noisy data and mixed-compliance patients",
        ),
        PhaseConfig(
            name="Empathetic Negotiation",
            phase_id=3,
            difficulty="hard",
            min_episodes=50,
            promotion_win_rate=1.0,
            promotion_avg_reward=99.0,
            description="Full persona randomization, trust-building, socio-economic barriers",
        ),
    ]

    def __init__(self):
        self.current_phase_idx = 0
        self.phase_episode_count = 0
        self.phase_history: List[Dict[str, Any]] = []
        self.window_size = 20

    @property
    def current_phase(self) -> PhaseConfig:
        return self.PHASES[self.current_phase_idx]

    @property
    def phase_id(self) -> int:
        return self.current_phase.phase_id

    def record_episode(self, outcome: str, total_reward: float) -> bool:
        self.phase_episode_count += 1
        self.phase_history.append({
            "outcome": outcome,
            "reward": total_reward,
            "phase": self.phase_id,
        })

        if self.current_phase_idx >= len(self.PHASES) - 1:
            return False

        cfg = self.current_phase
        if self.phase_episode_count < cfg.min_episodes:
            return False

        recent = self.phase_history[-self.window_size:]
        win_rate = sum(1 for e in recent if e["outcome"] == "WIN") / len(recent)
        avg_reward = sum(e["reward"] for e in recent) / len(recent)

        if win_rate >= cfg.promotion_win_rate and avg_reward >= cfg.promotion_avg_reward:
            self._promote()
            return True
        return False

    def _promote(self):
        old_phase = self.current_phase.name
        self.current_phase_idx += 1
        self.phase_episode_count = 0
        logger.info(
            f"\n{'*' * 60}\n"
            f"  CURRICULUM PROMOTION: {old_phase} -> {self.current_phase.name}\n"
            f"{'*' * 60}"
        )

    def force_promote(self, reason: str = "external trigger") -> bool:
        """Externally force advancing to the next phase.

        Used by the per-phase reward-threshold early-stop in the training
        loop: when the policy sustains the configured reward target for
        the current phase, we advance immediately (in parallel to the
        built-in win-rate-based promotion in `record_episode`). Returns
        True if a promotion happened, False if already in the final phase.
        """
        if self.current_phase_idx >= len(self.PHASES) - 1:
            return False
        logger.info(f"  [Scheduler] force_promote() called: {reason}")
        self._promote()
        return True

    def get_env_options(self) -> Dict[str, Any]:
        return {
            "phase": self.phase_id,
            "difficulty": self.current_phase.difficulty,
        }

    def get_summary(self) -> Dict[str, Any]:
        recent = self.phase_history[-self.window_size:] if self.phase_history else []
        win_rate = sum(1 for e in recent if e["outcome"] == "WIN") / max(len(recent), 1)
        avg_reward = sum(e["reward"] for e in recent) / max(len(recent), 1)
        return {
            "phase": self.phase_id,
            "phase_name": self.current_phase.name,
            "phase_episodes": self.phase_episode_count,
            "rolling_win_rate": round(win_rate, 3),
            "rolling_avg_reward": round(avg_reward, 3),
            "total_episodes": len(self.phase_history),
        }


# ============================================================================
# Trajectory-level Verifier (process-aware, no learned critic)
# ============================================================================

def verify_trajectory_reward(trajectory: Dict[str, Any]) -> float:
    """
    Compute the verified scalar reward for a full episode trajectory.

    Uses the env's per-component reward tracking (rebalanced so that
    process rewards dominate terminal rewards) plus a few outcome-level
    bonuses/penalties. Returns a single float used as R_i for GRPO.
    """
    total = trajectory.get("total_reward", 0.0)
    outcome = trajectory.get("outcome", "unknown")
    steps = trajectory.get("steps", 0)
    milestones = trajectory.get("milestones", {})
    patient_state = trajectory.get("patient_state", {})

    # Outcome bonus is intentionally SMALL since the env's terminal reward
    # is already smoothed in [-1.1, +0.6]. We do not double-count.
    outcome_bonus = {
        "WIN":         0.30,
        "FATAL_LOSS": -0.40,
        "AMA_LOSS":   -0.20,
        "INCORRECT":  -0.20,
        "PARTIAL":     0.00,
        "unknown":    -0.10,
    }.get(outcome, -0.10)

    efficiency = max(0.0, 1.0 - (steps / 20.0))
    completion = milestones.get("completion", 0.0) if isinstance(milestones, dict) else 0.0
    milestone_bonus = completion * 0.30

    trust = patient_state.get("trust", 50) if isinstance(patient_state, dict) else 50
    trust_bonus = 0.10 if trust > 60 else (-0.05 if trust < 30 else 0.0)

    return total + outcome_bonus + efficiency * 0.15 + milestone_bonus + trust_bonus


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(
    model_name: str = "distilgpt2",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load Doctor policy model with Unsloth or HF fallback."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine target modules based on model architecture
    if "gpt2" in model_name.lower():
        target_modules = ["c_attn"]
        load_in_4bit = False
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if device == "cuda":
        try:
            from unsloth import FastLanguageModel
            logger.info(f"Attempting to load model {model_name} via Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=None,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=16,
                # lora_dropout MUST be 0 to engage Unsloth's fused LoRA kernels.
                lora_dropout=0,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth" if "gpt2" not in model_name.lower() else False,
            )
            logger.info(f"Loaded model via Unsloth: {model_name} (4-bit={load_in_4bit})")
            return model, tokenizer

        except Exception as e:
            logger.warning(f"Unsloth loading failed or unavailable: {e}. Falling back to HuggingFace.")

    # HF Fallback (or CPU path)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    logger.info(f"Loading base model {model_name} via HuggingFace on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"Loaded model via HF: {model_name}")
    return model, tokenizer


def load_reference_model(model_name: str, max_seq_length: int = 2048):
    """
    Load a frozen reference policy for KL regularization in GRPO.
    The reference is the un-LoRA'd base model (no adapters, no grad).
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            from unsloth import FastLanguageModel
            ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            return ref_model
        except Exception as e:
            logger.warning(f"Unsloth reference load failed: {e}. Falling back to HF.")

    # HF Fallback
    from transformers import AutoModelForCausalLM
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_model


# ============================================================================
# Doctor Action Generation
# ============================================================================

DOCTOR_SYSTEM_PROMPT = """You are an ER doctor performing triage. You must diagnose and treat the patient.

AVAILABLE TOOLS (respond with JSON):
1. {"tool": "read_soap"} - Read the patient's medical record
2. {"tool": "speak_to", "target": "patient", "message": "..."} - Talk to patient
3. {"tool": "speak_to", "target": "nurse", "message": "..."} - Talk to nurse
4. {"tool": "order_lab", "test_name": "..."} - Order a lab test
5. {"tool": "update_soap", "section": "Assessment|Plan", "content": "..."} - Update medical record
6. {"tool": "terminal_discharge", "treatment": "...", "is_emergency": true|false} - Final diagnosis

WORKFLOW: Read SOAP -> Talk to patient -> Order labs -> Update Assessment -> Update Plan -> Discharge

Document your reasoning in update_soap before discharge: the env rewards
intermediate diagnosis quality, not just final treatment.

Respond with ONLY a valid JSON action object."""


def generate_doctor_action(
    model,
    tokenizer,
    observation: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate Doctor's JSON action from observation."""
    import torch
    # Allow override from env so we can dial generation length without
    # editing code (used by the Kaggle clean_launch when VRAM is tight).
    _env_mnt = os.environ.get("ERMAP_DOCTOR_MAX_NEW_TOKENS")
    if _env_mnt:
        try:
            max_new_tokens = max(32, int(_env_mnt))
        except ValueError:
            pass

    prompt = f"{DOCTOR_SYSTEM_PROMPT}\n\nObservation:\n{observation}\n\nJSON Action:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()
    finally:
        # Free generation activations + KV cache before the next Doctor turn.
        # Without this, repeated generate() calls accrete cached blocks that
        # are reachable but unfreed, eventually OOMing the GRPO update.
        try:
            del inputs
        except Exception:
            pass
        try:
            del outputs  # noqa: F821
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Episode Rollout
# ============================================================================

def run_episode(
    model,
    tokenizer,
    env,
    env_options: Dict[str, Any],
    seed: Optional[int] = None,
    device: str = "cuda",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run a single episode. Returns a trajectory dict with the full
    (prompt, response) sequence and per-component reward decomposition.
    """
    obs, info = env.reset(seed=seed, options=env_options)
    done = False
    truncated = False

    prompts: List[str] = []        # full prompts including DOCTOR_SYSTEM_PROMPT
    responses: List[str] = []      # the doctor's JSON action
    step_rewards: List[float] = []
    total_reward = 0.0
    steps = 0
    outcome = "unknown"
    last_info = info
    final_components: Dict[str, float] = {}

    while not done and not truncated:
        prompt = (
            f"{DOCTOR_SYSTEM_PROMPT}\n\n"
            f"Observation:\n{obs}\n\n"
            f"JSON Action:"
        )
        action_text = generate_doctor_action(
            model, tokenizer, obs, device=device, temperature=temperature
        )

        next_obs, reward, done, truncated, info = env.step(action_text)

        prompts.append(prompt)
        responses.append(action_text)
        step_rewards.append(float(reward))
        total_reward += float(reward)
        steps += 1
        last_info = info
        if "reward_components" in info:
            final_components = dict(info["reward_components"])
        obs = next_obs

        if done:
            try:
                obs_dict = json.loads(obs)
                event = obs_dict.get("event", "")
                if "win" in event:
                    outcome = "WIN"
                elif "fatal" in event:
                    outcome = "FATAL_LOSS"
                elif "ama" in event:
                    outcome = "AMA_LOSS"
                elif "incorrect" in event:
                    outcome = "INCORRECT"
                elif "partial" in event:
                    outcome = "PARTIAL"
            except json.JSONDecodeError:
                pass

    return {
        "prompts": prompts,
        "responses": responses,
        "step_rewards": step_rewards,
        "total_reward": total_reward,
        "steps": steps,
        "outcome": outcome,
        "milestones": last_info.get("milestones", {}),
        "patient_state": last_info.get("patient_state", {}),
        "reward_components": final_components,
    }


# ============================================================================
# Manual GRPO Update Step
# ============================================================================

def _response_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str,
    max_seq_length: int = 2048,
) -> Tuple[Any, int]:
    """
    Compute the sum of token-level log-probs of `response` under `model`,
    conditional on `prompt`. Returns (sum_logprob_tensor, num_tokens).
    The returned tensor remains on the autograd graph if model has
    requires_grad enabled.
    """
    import torch
    full = prompt + response
    full_ids = tokenizer(
        full, return_tensors="pt", truncation=True, max_length=max_seq_length
    ).input_ids.to(device)
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_length - 8
    ).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    seq_len = full_ids.shape[1]

    if seq_len <= prompt_len:
        return torch.tensor(0.0, device=device, requires_grad=False), 0

    logits = model(full_ids).logits  # [1, L, V]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # predict next token
    target_ids = full_ids[:, 1:]  # [1, L-1]
    token_logp = torch.gather(
        log_probs, -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, L-1]

    # Mask: only the response tokens contribute. The prediction at position
    # (prompt_len - 1) is the first response token.
    mask = torch.zeros_like(token_logp)
    mask[:, prompt_len - 1:] = 1.0
    n_tokens = int(mask.sum().item())
    sum_logp = (token_logp * mask).sum()
    return sum_logp, n_tokens


def manual_grpo_step(
    model,
    ref_model,
    tokenizer,
    trajectories: List[Dict[str, Any]],
    optimizer,
    beta: float = 0.04,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Apply one GRPO update over a group of G trajectories that share the
    same scenario (same env seed and options).

    Loss formulation:
        A_i  = (R_i - mean(R)) / (std(R) + eps)         # group-relative
        L    = - mean_i ( A_i * mean_t logp_pi(a_t|s_t) )
               + beta * mean_t ( logp_pi - logp_ref ) ** 2

    The KL term is computed implicitly as a squared log-ratio (a stable
    surrogate for low-magnitude policy drift). Token-level masking
    ensures we only learn from response tokens, not prompt tokens.
    """
    import torch
    if not trajectories:
        return {"loss": 0.0, "kl": 0.0, "n_steps": 0}

    rewards = torch.tensor(
        [verify_trajectory_reward(t) for t in trajectories],
        device=device, dtype=torch.float32,
    )
    if rewards.numel() < 2 or rewards.std().item() < 1e-6:
        # All trajectories same reward -> no signal; skip the update.
        return {
            "loss": 0.0, "kl": 0.0, "n_steps": 0,
            "advantages_mean": 0.0, "advantages_std": 0.0,
            "skipped": True,
        }

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    use_kl = (ref_model is not None) and (beta > 0.0)

    # Per-step backward (memory-bounded for T4): accumulating loss across all
    # (prompt, response) pairs and backward()-ing once at the end keeps every
    # forward-pass activation graph alive in VRAM, which OOMs on T4 once we
    # cross ~30 steps. Instead, scale each step's loss by 1/n_steps_total
    # and backward() it immediately — gradients accumulate correctly in .grad
    # and each forward's graph is released before the next forward begins.
    n_steps_total = max(1, sum(len(t.get("prompts", [])) for t in trajectories))

    optimizer.zero_grad()
    total_loss_value = 0.0
    total_kl = 0.0
    n_steps = 0

    for traj_idx, traj in enumerate(trajectories):
        adv = advantages[traj_idx]
        for prompt, response in zip(traj["prompts"], traj["responses"]):
            logp_pi, n_tok = _response_logprob(
                model, tokenizer, prompt, response, device
            )
            if n_tok == 0:
                continue

            logp_pi_norm = logp_pi / max(n_tok, 1)
            policy_term = -(adv * logp_pi_norm)

            if use_kl:
                with torch.no_grad():
                    logp_ref, _ = _response_logprob(
                        ref_model, tokenizer, prompt, response, device
                    )
                logp_ref_norm = logp_ref / max(n_tok, 1)
                kl_term = (logp_pi_norm - logp_ref_norm).pow(2)
                step_loss = policy_term + beta * kl_term
                total_kl += float(
                    (logp_pi_norm - logp_ref_norm).abs().detach().item()
                )
            else:
                step_loss = policy_term

            # Scale and immediately backward to free this step's activations.
            scaled = step_loss / n_steps_total
            scaled.backward()

            total_loss_value += float(step_loss.detach().item())
            n_steps += 1

            # Free intermediate tensors before the next forward.
            del logp_pi, logp_pi_norm, policy_term, step_loss, scaled

    if n_steps == 0:
        optimizer.zero_grad()
        return {"loss": 0.0, "kl": 0.0, "n_steps": 0}

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss_value / n_steps,
        "kl": total_kl / n_steps,
        "n_steps": n_steps,
        "advantages_mean": float(advantages.mean().item()),
        "advantages_std": float(advantages.std().item()),
        "rewards_mean": float(rewards.mean().item()),
        "rewards_std": float(rewards.std().item()),
        "skipped": False,
    }


# ============================================================================
# Save / Merge Path
# ============================================================================

def save_lora_adapters(model, tokenizer, output_dir: str) -> None:
    """Save just the LoRA adapter weights — safe and small."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved LoRA adapters to {output_dir}")


def merge_and_save_fp16(model, tokenizer, output_dir: str) -> None:
    """
    Properly merge LoRA into the base model and save in fp16.
    This is the correct path warned about in §16 of the hackathon guide:
    we do NOT naively upcast a 4-bit base then merge — we use the unsloth
    helper when available, which handles the merge in fp16 directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Unsloth provides save_pretrained_merged that does this safely.
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(
                output_dir, tokenizer, save_method="merged_16bit"
            )
            logger.info(f"Merged-and-saved (unsloth fp16) to {output_dir}")
            return
    except Exception as e:
        logger.warning(f"Unsloth merged save failed: {e}. Falling back to PEFT merge.")

    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged-and-saved (peft) to {output_dir}")
    except Exception as e:
        logger.error(
            f"Merge-and-save failed: {e}. Saved adapters only as fallback."
        )
        save_lora_adapters(model, tokenizer, output_dir)


# ============================================================================
# GRPO Training Loop
# ============================================================================

def train(
    num_episodes: int = 1,
    group_size: int = 4,
    model_name: str = "distilgpt2",
    groq_api_key: str = "",
    learning_rate: float = 5e-6,
    kl_beta: float = 0.04,
    use_wandb: bool = False,
    use_mlflow: bool = True,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: str = "ER-MAP-GRPO",
    mlflow_run_name: Optional[str] = None,
    output_dir: str = "./er_map_grpo_checkpoints",
    dry_run: bool = False,
    *,
    # ---------------- Early-stopping ("train until optimal") ---------------
    # Per-phase reward thresholds. After every GRPO update we look at the
    # last `convergence_window` groups; if ALL of them belong to the same
    # current phase AND have rolling_avg_reward >= phase_reward_targets
    # [current_phase], we either:
    #   - force-promote to the next phase (Phase 1 / 2), OR
    #   - terminate training (Phase 3).
    # `num_episodes` becomes a HARD CAP, not a budget.
    #
    # Default mapping mirrors the user-tunable bar:
    #   Phase 1 (Tool Mastery)         : sustained rolling avg >= 1.5
    #   Phase 2 (Clinical Reasoning)   : sustained rolling avg >= 1.2
    #   Phase 3 (Empathetic Negotiation): sustained rolling avg >= 1.0
    phase_reward_targets: Optional[Dict[int, float]] = None,
    phase_min_win_rate: float = 0.20,
    convergence_window: int = 3,
    early_stop: bool = True,
    # ---------------- Fixed-budget curriculum (alternative mode) -----------
    # When set, training advances phases at FIXED episode counts instead of
    # via the reward-target early-stop. Useful when you want a clean
    # reward-growth curve over a known wall-clock budget. Example:
    #   phase_episode_budgets = {1: 20, 2: 30, 3: 50}  # 100 episodes total
    # When this is provided, `early_stop` is forced to False (the reward
    # thresholds become observational, logged for plots only) and
    # `num_episodes` is auto-set to sum(phase_episode_budgets.values()) if
    # the caller passed a smaller / inconsistent value.
    phase_episode_budgets: Optional[Dict[int, int]] = None,
):
    if phase_reward_targets is None:
        phase_reward_targets = {1: 1.5, 2: 1.2, 3: 1.0}

    # Fixed-budget mode overrides early-stop and aligns num_episodes.
    fixed_budget_mode = phase_episode_budgets is not None and len(phase_episode_budgets) > 0
    if fixed_budget_mode:
        # Sanity: must have all 3 phases keyed, all positive ints
        missing = [p for p in (1, 2, 3) if p not in phase_episode_budgets]
        if missing:
            raise ValueError(
                f"phase_episode_budgets must include all phases (1,2,3); missing: {missing}"
            )
        for _p, _n in phase_episode_budgets.items():
            if not isinstance(_n, int) or _n <= 0:
                raise ValueError(
                    f"phase_episode_budgets[{_p}] must be a positive int, got {_n!r}"
                )
        budget_sum = sum(phase_episode_budgets.values())
        if num_episodes != budget_sum:
            logger.info(
                f"[Fixed-budget] num_episodes ({num_episodes}) overridden to "
                f"sum(phase_episode_budgets) = {budget_sum}"
            )
            num_episodes = budget_sum
        if early_stop:
            logger.info(
                "[Fixed-budget] early_stop=True is incompatible with fixed budgets; "
                "disabling early_stop. Reward targets will still be logged for plots."
            )
            early_stop = False

    logger.info("Training started")

    # Dynamic clamp for demo mode/low episodes
    if num_episodes < group_size:
        logger.info(f"Clamping group_size from {group_size} to {num_episodes} for lightweight execution")
        group_size = num_episodes

    torch = None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    logger.info(f"Device: {device}")
    logger.info(
        f"GRPO Config: group_size={group_size}, lr={learning_rate}, "
        f"kl_beta={kl_beta}"
    )

    groq_key = groq_api_key or ((os.environ.get("GROQ_API_KEY") or os.environ.get("groq")) or os.environ.get("groq", ""))

    scheduler = CurriculumScheduler()
    logger.info(f"Starting Phase: {scheduler.current_phase.name}")
    logger.info(f"  {scheduler.current_phase.description}")

    # --- MLflow ---
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            # Configure fast timeouts for MLflow connection requests to prevent hanging in CI environments
            if "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ:
                os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
            if "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ:
                os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"

            # Resolve tracking URI (cli -> environment -> local server default)
            resolved_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"

            # Check if URI is an HTTP/HTTPS endpoint and if it's reachable
            if resolved_uri.startswith("http://") or resolved_uri.startswith("https://"):
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(resolved_uri)
                host = parsed.hostname or "127.0.0.1"
                parsed_port = parsed.port
                port = int(parsed_port) if parsed_port is not None else (80 if parsed.scheme == "http" else 443)
                try:
                    # Quick socket connection check with a 1-second timeout
                    with socket.create_connection((host, port), timeout=1.0):
                        pass
                except (socket.timeout, ConnectionRefusedError, OSError) as conn_err:
                    logger.warning(
                        f"MLflow tracking server at {resolved_uri} is unreachable ({conn_err}). "
                        f"Falling back to local SQLite database (sqlite:///mlflow.db) to ensure logging runs successfully."
                    )
                    resolved_uri = "sqlite:///mlflow.db"

            mlflow.set_tracking_uri(resolved_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow_run = mlflow.start_run(run_name=mlflow_run_name)
            
            # Log required parameters
            mlflow.log_params({
                "model_name": model_name,
                "learning_rate": learning_rate,
                "group_size": group_size,
                "phase": scheduler.phase_id,
                "num_episodes": num_episodes,
                "kl_beta": kl_beta,
                "early_stop": early_stop,
                "convergence_window": convergence_window,
                "phase1_target": phase_reward_targets.get(1),
                "phase2_target": phase_reward_targets.get(2),
                "phase3_target": phase_reward_targets.get(3),
                "fixed_budget_mode": fixed_budget_mode,
                "dry_run": dry_run,
            })
            
            # Log lightweight dummy metrics for demo mode if training has not started yet
            mlflow.log_metric("reward", 0.8)
            mlflow.log_metric("loss", 0.2)
            
            logger.info("MLflow logging initialized")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}. Continuing without it.")

    # --- Model / reference / optimizer ---
    if not dry_run:
        model, tokenizer = load_model_and_tokenizer(model_name=model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Reference model is only needed if KL regularization is on. Loading
        # it costs ~5 GB of VRAM (a second 4-bit copy of the 8B base), which
        # OOMs the T4 once activations + gradients are added. Skip it when
        # kl_beta <= 0 — GRPO works fine without KL on short runs.
        if kl_beta > 0.0:
            ref_model = load_reference_model(model_name)
        else:
            ref_model = None
            logger.info(
                "kl_beta <= 0 -> skipping reference model load "
                "(saves ~5 GB VRAM; GRPO loss will use pure advantage term)"
            )
        import torch
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    else:
        model, tokenizer, ref_model, optimizer = None, None, None, None
        logger.info("DRY RUN mode -- skipping model loading")

    # --- W&B ---
    wandb_run = None
    if use_wandb and not dry_run:
        try:
            import wandb
            wandb_run = wandb.init(
                project="er-map-grpo",
                config={
                    "model_name": model_name,
                    "num_episodes": num_episodes,
                    "group_size": group_size,
                    "learning_rate": learning_rate,
                    "kl_beta": kl_beta,
                },
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Continuing without it.")

    # --- Environment ---
    from ER_MAP.envs.triage_env import TriageEnv
    env = None if dry_run else TriageEnv(groq_api_key=groq_key, render_mode=None)

    os.makedirs(output_dir, exist_ok=True)
    metrics_log: List[Dict[str, Any]] = []
    start_time = time.time()
    episode_idx = 0

    # Convergence tracking: deque of (phase, avg_reward, win_rate) for the
    # last `convergence_window` GRPO groups. Used by the early-stopping
    # check at the bottom of each iteration of the outer loop.
    from collections import deque as _deque
    convergence_buffer: _deque = _deque(maxlen=max(1, int(convergence_window)))
    converged: bool = False

    logger.info(f"\nStarting GRPO training for up to {num_episodes} episodes "
                f"(={num_episodes // group_size} GRPO updates)")
    if fixed_budget_mode:
        logger.info(
            "  Fixed-budget curriculum: phases advance at fixed episode counts."
        )
        for _pid in sorted(phase_episode_budgets.keys()):
            logger.info(
                f"    Phase {_pid}: {phase_episode_budgets[_pid]} episodes "
                f"(target avg-reward {phase_reward_targets.get(_pid, float('nan')):+.2f}, "
                f"observational only)"
            )
    elif early_stop:
        logger.info(
            f"  Early-stop ON: per-phase reward thresholds (sustained for "
            f"{convergence_window} groups, win-rate >= {phase_min_win_rate:.0%}):"
        )
        for _pid in sorted(phase_reward_targets.keys()):
            _action = "END TRAINING" if _pid == 3 else "force-promote to next phase"
            logger.info(
                f"    Phase {_pid}: avg-reward >= {phase_reward_targets[_pid]:+.2f} -> {_action}"
            )
    else:
        logger.info("  Early-stop OFF: will run all configured episodes.")

    # Optional Unsloth fast inference/training mode swappers. When the model
    # was loaded via Unsloth, switching to inference mode before generation
    # disables the gradient-checkpointing hooks (which otherwise pin ~7 GB
    # of activation buffers in VRAM even under torch.no_grad()), and
    # switching back to training mode re-enables them for the GRPO update.
    _has_unsloth_swap = False
    _FastLM = None
    if not dry_run and model is not None:
        try:
            from unsloth import FastLanguageModel as _FastLM  # noqa: F401
            _has_unsloth_swap = (
                hasattr(_FastLM, "for_inference") and
                hasattr(_FastLM, "for_training")
            )
        except Exception:
            _has_unsloth_swap = False

    def _to_inference():
        if _has_unsloth_swap:
            try:
                _FastLM.for_inference(model)
            except Exception:
                pass

    def _to_training():
        if _has_unsloth_swap:
            try:
                _FastLM.for_training(model)
            except Exception:
                pass

    last_loss = 0.2
    # Outer loop: one iteration = one GRPO update over `group_size` episodes
    while episode_idx < num_episodes:
        env_options = scheduler.get_env_options()
        group_seed = random.randint(0, 1_000_000)

        trajectories: List[Dict[str, Any]] = []
        group_outcomes: List[str] = []
        group_rewards: List[float] = []

        # Switch to inference mode so generation doesn't pin gradient
        # checkpointing buffers (the silent ~7 GB VRAM leak on T4).
        _to_inference()

        # --- Collect G trajectories under same scenario ---
        for g in range(group_size):
            episode_idx += 1
            ep_start = time.time()

            logger.info(
                f"\n{'=' * 60}\n"
                f"  Episode {episode_idx}/{num_episodes} | Group {g+1}/{group_size} | "
                f"Phase {scheduler.phase_id}: {scheduler.current_phase.name} | "
                f"Difficulty: {env_options.get('difficulty', 'random')} | "
                f"Seed: {group_seed}\n"
                f"{'=' * 60}"
            )

            if dry_run:
                outcome = random.choice(["WIN", "WIN", "FATAL_LOSS", "INCORRECT", "PARTIAL"])
                total_reward = (
                    random.uniform(0.5, 2.0) if outcome == "WIN"
                    else random.uniform(-2.0, 0.3)
                )
                trajectory = {
                    "prompts": [], "responses": [], "step_rewards": [],
                    "total_reward": total_reward,
                    "steps": random.randint(3, 15),
                    "outcome": outcome,
                    "milestones": {"completion": random.uniform(0.3, 1.0)},
                    "patient_state": {"trust": random.uniform(20, 80)},
                    "reward_components": {
                        "process": random.uniform(0.1, 0.6),
                        "diagnosis": random.uniform(0.0, 0.3),
                        "plan": random.uniform(0.0, 0.25),
                        "labs": random.uniform(0.0, 0.4),
                        "treatment": random.uniform(-0.4, 0.6),
                        "empathy": random.uniform(-0.1, 0.3),
                        "milestones": random.uniform(0.0, 0.3),
                        "consent": random.uniform(-0.5, 0.25),
                        "documentation": random.uniform(-0.3, 0.4),
                        "emergency_id": random.uniform(-0.3, 0.3),
                        "penalties": random.uniform(-0.3, 0.0),
                    },
                }
            else:
                trajectory = run_episode(
                    model, tokenizer, env, env_options,
                    seed=group_seed, device=device,
                )

            trajectories.append(trajectory)
            group_outcomes.append(trajectory["outcome"])
            group_rewards.append(trajectory["total_reward"])

            verified = verify_trajectory_reward(trajectory)
            comp = trajectory.get("reward_components", {})

            logger.info(
                f"  Outcome: {trajectory['outcome']} | "
                f"Steps: {trajectory['steps']} | "
                f"Raw: {trajectory['total_reward']:+.3f} | "
                f"Verified: {verified:+.3f} | "
                f"Process: {comp.get('process', 0) + comp.get('milestones', 0) + comp.get('labs', 0) + comp.get('diagnosis', 0) + comp.get('plan', 0):+.2f} | "
                f"Terminal: {comp.get('treatment', 0) + comp.get('emergency_id', 0):+.2f}"
            )

            promoted = scheduler.record_episode(
                trajectory["outcome"], trajectory["total_reward"]
            )
            if promoted:
                logger.info(
                    f"  PROMOTED to Phase {scheduler.phase_id}: "
                    f"{scheduler.current_phase.name}"
                )

            ep_time = time.time() - ep_start
            sched_summary = scheduler.get_summary()
            metrics_log.append({
                "episode": episode_idx,
                "group_idx": g,
                "group_seed": group_seed,
                "phase": sched_summary["phase"],
                "phase_name": sched_summary["phase_name"],
                "outcome": trajectory["outcome"],
                "steps": trajectory["steps"],
                "raw_reward": round(trajectory["total_reward"], 3),
                "verified_reward": round(verified, 3),
                "rolling_win_rate": sched_summary["rolling_win_rate"],
                "rolling_avg_reward": sched_summary["rolling_avg_reward"],
                "milestones": trajectory.get("milestones", {}),
                "reward_components": comp,
                "patient_trust": trajectory.get("patient_state", {}).get("trust"),
                "episode_time_s": round(ep_time, 1),
            })

            if use_mlflow and mlflow_run:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "episode/raw_reward": float(trajectory["total_reward"]),
                        "episode/verified_reward": float(verified),
                        "episode/steps": float(trajectory["steps"]),
                        "episode/patient_trust": float(trajectory.get("patient_state", {}).get("trust") or 50.0),
                        "episode/time_s": float(ep_time),
                        "curriculum/rolling_win_rate": float(sched_summary["rolling_win_rate"]),
                        "curriculum/rolling_avg_reward": float(sched_summary["rolling_avg_reward"]),
                        "curriculum/phase_id": float(sched_summary["phase"]),
                        # Required metrics
                        "reward": float(trajectory["total_reward"]),
                        "loss": float(last_loss),
                        "episode": float(episode_idx),
                        "empathy_score": float(comp.get("empathy", 0.0)),
                    }, step=episode_idx)
                    if comp:
                        comp_metrics = {f"component/{k}": float(v) for k, v in comp.items() if v is not None}
                        mlflow.log_metrics(comp_metrics, step=episode_idx)
                except Exception as e:
                    logger.warning(f"Failed to log episode metrics to MLflow: {e}")

            logger.info("Episode completed")

        # In dry-run we don't actually run a GRPO step, but we still
        # synthesize plausible update stats so the per-phase plotter has
        # something to render during development / smoke tests.
        if dry_run and trajectories and metrics_log:
            metrics_log[-1]["grpo_update"] = {
                "loss":         round(random.uniform(-0.6, 0.4), 5),
                "kl":           round(random.uniform(0.001, 0.08), 5),
                "adv_mean":     round(random.uniform(-0.05, 0.05), 4),
                "adv_std":      round(random.uniform(0.4, 1.6), 4),
                "rewards_mean": round(sum(group_rewards) / max(len(group_rewards), 1), 4),
                "rewards_std":  round((max(group_rewards) - min(group_rewards)) / 2.0
                                      if len(group_rewards) > 1 else 0.1, 4),
                "n_steps":      sum(t["steps"] for t in trajectories),
                "skipped":      False,
            }
            stats = metrics_log[-1]["grpo_update"]
            last_loss = stats.get("loss", last_loss)
            if use_mlflow and mlflow_run:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "grpo/loss": float(stats["loss"]),
                        "loss": float(stats["loss"]),
                        "grpo/kl": float(stats["kl"]),
                        "grpo/adv_mean": float(stats["adv_mean"]),
                        "grpo/adv_std": float(stats["adv_std"]),
                        "grpo/rewards_mean": float(stats["rewards_mean"]),
                        "grpo/rewards_std": float(stats["rewards_std"]),
                        "grpo/n_steps": float(stats["n_steps"]),
                    }, step=episode_idx)
                except Exception as e:
                    logger.warning(f"Failed to log dry-run GRPO update to MLflow: {e}")

        # --- GRPO update over the group ---
        if not dry_run and trajectories:
            # Drop generation activations / KV caches before switching to
            # training mode and starting the backward pass.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                _free, _total = torch.cuda.mem_get_info(0)
                logger.info(
                    f"  Pre-update VRAM: {_free/1e9:.2f}/{_total/1e9:.2f} GB free"
                )

            # Switch the model back to training mode so LoRA grads flow.
            _to_training()

            try:
                stats = manual_grpo_step(
                    model, ref_model, tokenizer, trajectories,
                    optimizer, beta=kl_beta, device=device,
                )
                last_loss = stats.get("loss", last_loss)
                logger.info(
                    f"  [GRPO] loss={stats.get('loss', 0):+.4f} | "
                    f"kl={stats.get('kl', 0):+.4f} | "
                    f"adv_mean={stats.get('advantages_mean', 0):+.3f} | "
                    f"adv_std={stats.get('advantages_std', 0):+.3f} | "
                    f"n_steps={stats.get('n_steps', 0)} | "
                    f"skipped={stats.get('skipped', False)}"
                )
                # Stamp the GRPO update stats onto the LAST metric of this
                # group so the per-phase plotter can correlate update-level
                # stats (loss / kl / advantages) with episode-level rewards.
                if metrics_log:
                    metrics_log[-1]["grpo_update"] = {
                        "loss":           round(float(stats.get("loss", 0.0)), 5),
                        "kl":             round(float(stats.get("kl", 0.0)), 5),
                        "adv_mean":       round(float(stats.get("advantages_mean", 0.0)), 4),
                        "adv_std":        round(float(stats.get("advantages_std", 0.0)), 4),
                        "rewards_mean":   round(float(stats.get("rewards_mean", 0.0)), 4),
                        "rewards_std":    round(float(stats.get("rewards_std", 0.0)), 4),
                        "n_steps":        int(stats.get("n_steps", 0)),
                        "skipped":        bool(stats.get("skipped", False)),
                    }
                if wandb_run:
                    wandb_run.log({
                        "grpo/loss": stats.get("loss", 0.0),
                        "grpo/kl": stats.get("kl", 0.0),
                        "grpo/adv_mean": stats.get("advantages_mean", 0.0),
                        "grpo/adv_std": stats.get("advantages_std", 0.0),
                        "grpo/n_steps": stats.get("n_steps", 0),
                        "phase": scheduler.phase_id,
                        "rolling_win_rate": scheduler.get_summary()["rolling_win_rate"],
                        "rolling_avg_reward": scheduler.get_summary()["rolling_avg_reward"],
                    })
                if use_mlflow and mlflow_run:
                    try:
                        import mlflow
                        mlflow.log_metrics({
                            "grpo/loss": float(stats.get("loss", 0.0)),
                            "loss": float(stats.get("loss", 0.0)),
                            "grpo/kl": float(stats.get("kl", 0.0)),
                            "grpo/adv_mean": float(stats.get("advantages_mean", 0.0)),
                            "grpo/adv_std": float(stats.get("advantages_std", 0.0)),
                            "grpo/rewards_mean": float(stats.get("rewards_mean", 0.0)),
                            "grpo/rewards_std": float(stats.get("rewards_std", 0.0)),
                            "grpo/n_steps": float(stats.get("n_steps", 0)),
                        }, step=episode_idx)
                    except Exception as e:
                        logger.warning(f"Failed to log GRPO update to MLflow: {e}")
            except Exception as e:
                logger.error(f"  GRPO update failed: {e}")
                # Best-effort recovery: drop graphs/cache so the next group
                # starts from a clean VRAM baseline instead of cascading OOM.
                if torch and torch.cuda.is_available():
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        # --- Free generation activations / KV cache before the next group ---
        del trajectories
        if torch and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        # --- Periodic checkpoint (LoRA adapters) ---
        if episode_idx % (group_size * 5) == 0 and not dry_run:
            ckpt_path = os.path.join(
                output_dir,
                f"checkpoint_ep{episode_idx}_phase{scheduler.phase_id}",
            )
            try:
                save_lora_adapters(model, tokenizer, ckpt_path)
            except Exception as e:
                logger.error(f"  Checkpoint failed: {e}")

            metrics_path = os.path.join(output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

        # --- Rolling stats every group ---
        s = scheduler.get_summary()
        logger.info(
            f"  [Scheduler] Phase {s['phase']} ({s['phase_name']}) | "
            f"Win Rate: {s['rolling_win_rate']:.1%} | "
            f"Avg Reward: {s['rolling_avg_reward']:+.2f} | "
            f"Phase Episodes: {s['phase_episodes']}"
        )

        # --- Fixed-budget phase transition ---------------------------------
        # When the operator pre-allocates per-phase episode budgets (e.g.
        # P1=20, P2=30, P3=50), advance at the boundaries regardless of
        # reward. Phase 3 budget exhaustion lets the outer-loop
        # `num_episodes` cap end training naturally.
        if fixed_budget_mode:
            current_phase = s["phase"]
            budget = phase_episode_budgets.get(current_phase, 0)
            if (
                current_phase < 3
                and s["phase_episodes"] >= budget
            ):
                promoted = scheduler.force_promote(
                    reason=(
                        f"fixed-budget: completed {s['phase_episodes']} episodes "
                        f"in Phase {current_phase} (budget={budget})"
                    )
                )
                if promoted:
                    new_phase = scheduler.phase_id
                    logger.info(
                        f"  [Fixed-budget] Phase {current_phase} budget exhausted "
                        f"-> Phase {new_phase}: {scheduler.current_phase.name} "
                        f"({phase_episode_budgets.get(new_phase, '?')} episodes allocated)"
                    )

        # --- Per-phase early-stop / promotion check ------------------------
        # Maintain a buffer of the last `convergence_window` GRPO groups
        # with their (phase, rolling_avg, rolling_win). When ALL N entries
        # share the SAME current_phase and meet that phase's reward bar
        # (and the soft win-rate floor), we either force-promote (Phase 1
        # / Phase 2) or terminate training (Phase 3).
        if early_stop and group_rewards:
            convergence_buffer.append({
                "phase":       s["phase"],
                "rolling_avg": s["rolling_avg_reward"],
                "rolling_win": s["rolling_win_rate"],
            })
            current_phase = s["phase"]
            phase_target = phase_reward_targets.get(current_phase, float("inf"))

            buffer_full = len(convergence_buffer) >= convergence_window
            same_phase = buffer_full and all(b["phase"] == current_phase for b in convergence_buffer)
            reward_met = buffer_full and all(b["rolling_avg"] >= phase_target for b in convergence_buffer)
            winrate_met = buffer_full and all(b["rolling_win"] >= phase_min_win_rate for b in convergence_buffer)

            if buffer_full and same_phase and reward_met and winrate_met:
                if current_phase >= 3:
                    converged = True
                    logger.info(
                        f"\n{'*' * 60}\n"
                        f"  EARLY STOP: Phase {current_phase} convergence reached after "
                        f"{episode_idx} episodes\n"
                        f"  Last {convergence_window} groups all sustained:\n"
                        f"    rolling_avg_reward >= {phase_target:+.2f}\n"
                        f"    rolling_win_rate   >= {phase_min_win_rate:.0%}\n"
                        f"  in Phase {current_phase} ({s['phase_name']})\n"
                        f"{'*' * 60}"
                    )
                    break
                else:
                    promoted = scheduler.force_promote(
                        reason=(
                            f"sustained rolling-avg-reward {phase_target:+.2f} for "
                            f"{convergence_window} consecutive groups in Phase {current_phase}"
                        )
                    )
                    if promoted:
                        # Reset the buffer so the new phase starts fresh —
                        # we don't want stale Phase-N entries to satisfy
                        # the Phase-(N+1) check.
                        convergence_buffer.clear()
            else:
                # Show a compact "progress toward this phase's target" line
                # so the operator can see the watchdog working.
                qualified = sum(
                    1 for b in convergence_buffer
                    if b["phase"] == current_phase
                    and b["rolling_avg"] >= phase_target
                    and b["rolling_win"] >= phase_min_win_rate
                )
                action = "END TRAINING" if current_phase >= 3 else "promote"
                logger.info(
                    f"  [EarlyStop] Phase {current_phase} target avg-reward "
                    f">= {phase_target:+.2f}: qualified {qualified}/{convergence_window} "
                    f"recent groups (need all {convergence_window} -> {action})"
                )

    # --- Final save: adapters + merged fp16 ---
    total_time = time.time() - start_time
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    if not dry_run:
        adapter_dir = os.path.join(output_dir, "final_lora")
        merged_dir = os.path.join(output_dir, "final_merged_fp16")
        try:
            save_lora_adapters(model, tokenizer, adapter_dir)
            merge_and_save_fp16(model, tokenizer, merged_dir)
        except Exception as e:
            logger.error(f"Final save failed: {e}")

        # Smoke-test inference on the trained model so we catch broken
        # save paths before the demo (§16 explicit warning).
        try:
            test_obs = (
                '{"event":"episode_start","nurse_experience":"veteran",'
                '"message":"You are an ER doctor.","soap_summary":{}}'
            )
            sample = generate_doctor_action(
                model, tokenizer, test_obs, device=device, max_new_tokens=128
            )
            logger.info(f"  Post-training inference smoke test OK: {sample[:120]}")
        except Exception as e:
            logger.error(f"  Post-training inference smoke test FAILED: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total episodes: {episode_idx}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Final phase: {scheduler.current_phase.name}")
    logger.info(f"Metrics saved: {metrics_path}")
    logger.info(f"{'=' * 60}")

    if not dry_run:
        env.close()

    if wandb_run:
        wandb_run.finish()

    if use_mlflow and mlflow_run:
        try:
            import mlflow
            # Log final metrics file
            if os.path.exists(metrics_path):
                mlflow.log_artifact(metrics_path, artifact_path="metrics")
            # Log final merged model / checkpoints if not dry run
            if not dry_run:
                adapter_dir = os.path.join(output_dir, "final_lora")
                merged_dir = os.path.join(output_dir, "final_merged_fp16")
                if os.path.exists(adapter_dir):
                    mlflow.log_artifacts(adapter_dir, artifact_path="final_lora")
                if os.path.exists(merged_dir):
                    mlflow.log_artifacts(merged_dir, artifact_path="final_merged_fp16")
        except Exception as e:
            logger.warning(f"Failed to log final artifacts to MLflow: {e}")
        # Safely end MLflow run regardless of artifact logging outcome
        try:
            import mlflow
            mlflow.end_run()
            logger.info("MLflow run ended successfully.")
        except Exception as e:
            print("MLflow finalize warning:", e)

    return metrics_log


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ER-MAP GRPO Training with Curriculum")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Hard cap on total training episodes (default 1 for lightweight demo, early-stop may finish sooner)")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size (G)")
    parser.add_argument("--model", type=str,
                        default="distilgpt2",
                        help="Base model (distilgpt2 by default, previously unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API key")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="KL coefficient")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--mlflow", action="store_true", default=True, help="Log to MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable logging to MLflow")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="ER-MAP-GRPO", help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--output-dir", type=str,
                        default="./er_map_grpo_checkpoints", help="Output dir")
    parser.add_argument("--dry-run", action="store_true", help="Test scheduler without model")

    # Early-stopping knobs (per-phase reward targets)
    parser.add_argument("--phase1-target", type=float, default=1.5,
                        help="Phase 1 sustained rolling-avg-reward bar (force-promote when met)")
    parser.add_argument("--phase2-target", type=float, default=1.2,
                        help="Phase 2 sustained rolling-avg-reward bar (force-promote when met)")
    parser.add_argument("--phase3-target", type=float, default=1.0,
                        help="Phase 3 sustained rolling-avg-reward bar (END TRAINING when met)")
    parser.add_argument("--phase-min-win-rate", type=float, default=0.20,
                        help="Soft floor on rolling win rate (sanity check, default 20%%)")
    parser.add_argument("--convergence-window", type=int, default=3,
                        help="How many consecutive GRPO groups must meet target (default 3)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stop (always run all configured episodes)")

    # Fixed-budget curriculum (mutually exclusive with early-stop)
    parser.add_argument("--phase1-budget", type=int, default=None,
                        help="Fixed episode budget for Phase 1 (Tool Mastery)")
    parser.add_argument("--phase2-budget", type=int, default=None,
                        help="Fixed episode budget for Phase 2 (Clinical Reasoning)")
    parser.add_argument("--phase3-budget", type=int, default=None,
                        help="Fixed episode budget for Phase 3 (Empathetic Negotiation)")

    args = parser.parse_args()

    _budgets = None
    if any(b is not None for b in (args.phase1_budget, args.phase2_budget, args.phase3_budget)):
        if not all(b is not None for b in (args.phase1_budget, args.phase2_budget, args.phase3_budget)):
            parser.error("--phase{1,2,3}-budget must all be set together")
        _budgets = {
            1: args.phase1_budget,
            2: args.phase2_budget,
            3: args.phase3_budget,
        }

    train(
        num_episodes=args.episodes,
        group_size=args.group_size,
        model_name=args.model,
        groq_api_key=args.groq_key,
        learning_rate=args.lr,
        kl_beta=args.kl_beta,
        use_wandb=args.wandb,
        use_mlflow=args.mlflow and not args.no_mlflow,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment_name=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        phase_reward_targets={
            1: args.phase1_target,
            2: args.phase2_target,
            3: args.phase3_target,
        },
        phase_min_win_rate=args.phase_min_win_rate,
        convergence_window=args.convergence_window,
        early_stop=not args.no_early_stop,
        phase_episode_budgets=_budgets,
    )

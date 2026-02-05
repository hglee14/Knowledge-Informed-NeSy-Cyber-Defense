"""RLlib PPO training entrypoint for CybORG (CAGE-4) with NeSy ablations.

This is a lightly-extended version of the paper's training script:
  - Adds --ablation {base,state,reward,full}
  - Sets env vars so the environment wrapper (cyborg_env_maker_paper_ablation.py)
    can switch behavior without touching CybORG internals.

Design intent:
  - Keep *everything else* identical across ablations (network, PPO config,
    rollout length, seeds, etc.), so improvements are attributable to NeSy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress DeprecationWarning (gymnasium rng.randint)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import numpy as np
import random
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

# PyTorch seed setup (if available)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local wrapper-only environment (Gymnasium-compatible)
import cyborg_env_maker_paper_ablation as cyborg_env_maker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Core experiment control
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stop-iters", type=int, default=100)

    # Ablation switch
    p.add_argument(
        "--ablation",
        type=str,
        default="base",
        choices=["base", "state", "reward", "full", "graph", "frame_stack", "adaptive_scale", "logic_guided", "rule_pruning", "full_logic", "full_rule", "full_all", "ontology", "full_ontology"],
        help="NeSy ablation mode: base | state | reward | full | graph | frame_stack | adaptive_scale | logic_guided | rule_pruning | full_logic | full_rule | full_all | ontology",
    )
    p.add_argument(
        "--nesy-lam",
        type=float,
        default=1.0,
        help="Reward shaping coefficient λ (used in reward/full).",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=800,
        help="TimeLimit for episodes (Gymnasium wrapper). Same as 1st_success: 800.",
    )

    # Resources
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-gpus", type=int, default=0)
    p.add_argument("--cpus-per-worker", type=int, default=1)
    p.add_argument("--rollout-fragment-length", type=int, default=200)
    p.add_argument("--train-batch-size", type=int, default=4000)
    
    # Logging
    p.add_argument("--logdir", type=str, default="ray_results/paper_ablation")

    return p.parse_args()


def _set_ablation_env_vars(mode: str, lam: float) -> None:
    """Environment wrapper reads these; RLlib workers inherit them."""

    os.environ["NESY_MODE"] = mode
    os.environ["NESY_LAM"] = str(lam)


class CybUptimeCallbacks(DefaultCallbacks):
    """Same callback as 1st_success — collect uptime metrics."""
    def on_postprocess_trajectory(
        self,
        *,
        worker=None,
        episode=None,
        agent_id="",
        policy_id="",
        policies=None,
        postprocessed_batch=None,
        original_batches=None,
        **kwargs,
    ) -> None:
        infos = postprocessed_batch.get("infos") if postprocessed_batch is not None else None
        arr: List[Dict[str, Any]] = []
        if isinstance(infos, (list, tuple)):
            arr = [d for d in infos if isinstance(d, dict)]
        else:
            try:
                arr = [d for d in list(infos) if isinstance(d, dict)]
            except Exception:
                try:
                    arr = [d for d in infos.tolist() if isinstance(d, dict)]
                except Exception:
                    arr = []
        if not arr:
            return

        ups = [d.get("uptime_value") for d in arr if d.get("uptime_value") is not None]
        if ups:
            episode.user_data.setdefault("uptime_values", []).extend(ups)
        
        # Separate raw reward logging (required for paper defense metrics)
        raw_rewards = [d.get("raw_reward") for d in arr if d.get("raw_reward") is not None]
        if raw_rewards:
            episode.user_data.setdefault("raw_rewards", []).extend(raw_rewards)
        
        shaping_bonuses = [d.get("shaping_bonus") for d in arr if d.get("shaping_bonus") is not None]
        if shaping_bonuses:
            episode.user_data.setdefault("shaping_bonuses", []).extend(shaping_bonuses)
        
        shaped_returns = [d.get("shaped_return") for d in arr if d.get("shaped_return") is not None]
        if shaped_returns:
            episode.user_data.setdefault("shaped_returns", []).extend(shaped_returns)
        
        # Observation dimension logging (to show Full Ontology vs Ontology difference)
        obs_dims = [d.get("obs_dim") for d in arr if d.get("obs_dim") is not None]
        if obs_dims:
            episode.user_data.setdefault("obs_dims", []).extend(obs_dims)

    def on_episode_end(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index=0,
        **kwargs,
    ) -> None:
        ups = episode.user_data.get("uptime_values", [])
        if ups:
            episode.custom_metrics["uptime_rate_mean"] = float(np.mean(ups))
        
        # Separate raw reward logging (required for paper defense metrics)
        raw_rewards = episode.user_data.get("raw_rewards", [])
        if raw_rewards:
            episode.custom_metrics["raw_reward_mean"] = float(np.mean(raw_rewards))
        
        shaping_bonuses = episode.user_data.get("shaping_bonuses", [])
        if shaping_bonuses:
            episode.custom_metrics["shaping_bonus_mean"] = float(np.mean(shaping_bonuses))
        
        shaped_returns = episode.user_data.get("shaped_returns", [])
        if shaped_returns:
            episode.custom_metrics["shaped_return_mean"] = float(np.mean(shaped_returns))
        
        # Observation dimension logging (to show Full Ontology vs Ontology difference)
        obs_dims = episode.user_data.get("obs_dims", [])
        if obs_dims:
            # Use first value since it must be the same for all steps
            episode.custom_metrics["obs_dim"] = float(obs_dims[0] if obs_dims else 0)
            
        # Also collect episode length (RLlib provides it but we store explicitly)
        if hasattr(episode, "length") and episode.length is not None:
            episode.custom_metrics["episode_length"] = float(episode.length)


def _ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def custom_log_creator(custom_path: str, custom_config: Optional[Dict[str, Any]] = None):
    """Custom logger creator to save progress.csv and other logs to specified directory."""
    def logger_creator(config: Dict[str, Any]):
        if custom_config:
            config.update(custom_config)
        # UnifiedLogger creates CSV, JSON, and TensorBoard logs.
        return UnifiedLogger(config, custom_path)
    return logger_creator


def main() -> None:
    args = parse_args()

    # ===== Global seed for reproducibility =====
    # Set all random seeds (for reproducibility)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    print(f"[Seed Setup] Global seeds set to: {args.seed}")
    print(f"  - numpy.random.seed({args.seed})")
    print(f"  - random.seed({args.seed})")
    if TORCH_AVAILABLE:
        print(f"  - torch.manual_seed({args.seed})")

    # Ensure all workers see the same mode/λ.
    _set_ablation_env_vars(args.ablation, args.nesy_lam)

    # Log directory setup
    log_dir = Path(args.logdir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Copy source code to result dir (reproducibility / versioning)
    _src_dir = Path(__file__).resolve().parent
    for _name in ("cyborg_env_maker_paper_ablation.py", "rllib_train_cyborg_with_metrics_paper_ablation.py"):
        _src = _src_dir / _name
        if _src.is_file():
            try:
                shutil.copy2(_src, log_dir / _name)
                print(f"[SOURCE] Copied {_name} -> {log_dir}")
            except Exception as e:
                print(f"[SOURCE] Copy failed for {_name}: {e}")

    env_name = "CybORG_CAGE4_Gymnasium"

    def env_creator(env_config: Dict[str, Any]):
        # Propagate ablation into env_config as well (more explicit than env vars).
        env_config = dict(env_config or {})
        # Per-worker seed: if worker_index present, use base_seed + worker_index
        worker_seed = args.seed
        if "worker_index" in env_config:
            worker_index = env_config.get("worker_index", 0)
            worker_seed = args.seed + worker_index
        env_config.update(
            {
                "seed": worker_seed,  # Unique seed per worker
                "max_episode_steps": args.max_episode_steps,
                "nesy_mode": args.ablation,
                "nesy_lam": args.nesy_lam,
                "blue_agent": "blue_agent_0",  # Same as 1st_success
            }
        )
        return cyborg_env_maker.create_cyborg_env(env_config)

    register_env(env_name, env_creator)

    # Ray init (settings for stability)
    # Memory and resource management for parallel runs
    # Use /tmp to avoid path length limit (107 bytes)
    ray_tmp_dir = getattr(args, 'ray_temp', None) or os.environ.get("RAY_TMPDIR")
    if not ray_tmp_dir:
        # Use short path (avoid socket path length limit)
        import tempfile
        ray_tmp_dir = str(Path(tempfile.gettempdir()) / "ray_tmp" / args.exp_name[:20])
    # Ray requires absolute path
    ray_tmp_dir = str(Path(ray_tmp_dir).resolve())
    
    spill_cfg_env = os.environ.get("RAY_object_spilling_config")
    if not spill_cfg_env:
        spill_dir = str(Path(ray_tmp_dir) / "spill")
        _ensure_dir(Path(spill_dir))
        ray_system_config = {
            "automatic_object_spilling_enabled": True,  # Same as 1st_success
            "min_spilling_size": 10 * 1024 * 1024,  # 10MB
            "memory_monitor_refresh_ms": 250,
        }
    else:
        ray_system_config = {"memory_monitor_refresh_ms": 250}
    
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,  # Disable dashboard (save resources)
        _temp_dir=ray_tmp_dir,  # Resolved to absolute path
        _system_config=ray_system_config,
    )

    # [Parallel run stability] Create dummy env to verify observation/action space
    # Check in main process first to avoid init failures in workers
    print("[Setup] Creating dummy environment to verify observation/action spaces...")
    dummy_env = cyborg_env_maker.create_cyborg_env({
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "nesy_mode": args.ablation,
        "nesy_lam": args.nesy_lam,
        "blue_agent": "blue_agent_0",
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    if hasattr(dummy_env, "close"):
        dummy_env.close()
    print(f"[Setup] Observation space: {obs_space}")
    print(f"[Setup] Action space: {act_space}")

    # Build config (keep aligned with paper defaults).
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={},
            disable_env_checking=True,  # [Parallel stability] disable env check
            observation_space=obs_space,  # [Parallel stability] explicit space
            action_space=act_space,  # [Parallel stability] explicit space
        )
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.rollout_fragment_length,
            num_envs_per_worker=1,
            batch_mode="truncate_episodes",  # [Parallel stability] truncate episodes
            compress_observations=True,  # [Parallel stability] compress obs (save memory)
        )
        .training(
            train_batch_size=args.train_batch_size,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.0,
            sgd_minibatch_size=min(128, args.train_batch_size),  # Smaller than train_batch_size
            num_sgd_iter=10,
            grad_clip=0.5,
        )
        .callbacks(CybUptimeCallbacks)  # Callback to collect uptime metrics
        .debugging(log_level="WARN")
    )

    # [Parallel stability] Disable connectors and use Simple Optimizer
    config.enable_connectors = False
    
    # Use simple optimizer (save memory and improve stability)
    cfg_dict = config.to_dict()
    cfg_dict["simple_optimizer"] = True
    
    # Seed for reproducibility (RLlib 2.x)
    cfg_dict["seed"] = args.seed
    # Per-worker seed: each worker uses base_seed + worker_index
    if "rollouts" not in cfg_dict:
        cfg_dict["rollouts"] = {}
    cfg_dict["rollouts"]["seed"] = args.seed
    
    config = PPOConfig.from_dict(cfg_dict)

    # Custom logger creator to save progress.csv to the given directory
    algo = config.build(logger_creator=custom_log_creator(str(log_dir)))

    print(f"\n[START] {args.exp_name} | ablation={args.ablation} | seed={args.seed} | lam={args.nesy_lam}")
    print(f"[LOG] Results will be saved to: {log_dir}\n")

    # Open progress.log for per-iteration metrics
    progress_log_path = log_dir / "progress.log"
    progress_log_file = open(progress_log_path, "w", encoding="utf-8")
    progress_log_file.write(f"# Training Progress Log: {args.exp_name}\n")
    progress_log_file.write(f"# Ablation: {args.ablation} | Seed: {args.seed} | Lambda: {args.nesy_lam}\n")
    progress_log_file.write("# Format: iteration,reward,uptime,episode_length,timesteps_total\n")
    progress_log_file.flush()

    last_result = None
    try:
        for i in range(1, args.stop_iters + 1):
            result = algo.train()
            last_result = result

            # Standard RLlib metrics
            rw = result.get("episode_reward_mean", float("nan"))
            ep_len = result.get("episode_len_mean", float("nan"))
            timesteps = result.get("timesteps_total", 0)

            # Custom uptime metric (set in env infos and aggregated by RLlib via Callback)
            uptime = float("nan")
            cm = result.get("custom_metrics") or {}
            
            # RLlib aggregates episode metrics: uptime_rate_mean -> uptime_rate_mean_mean
            # (Callback stores uptime_rate_mean; RLlib re-averages over episodes)
            for k in ("uptime_rate_mean_mean", "uptime_rate_mean", "uptime_mean", "uptime", "episode_uptime_mean"):
                if k in cm:
                    uptime = cm[k]
                    break

            # Console output
            print(f"[{args.exp_name}] Iter {i:03d} | Rw: {rw: .2f} | Uptime: {uptime: .4f} | EpLen: {ep_len: .2f}")
            
            # Write to progress.log
            progress_log_file.write(f"{i},{rw:.4f},{uptime:.4f},{ep_len:.2f},{timesteps}\n")
            progress_log_file.flush()
    finally:
        progress_log_file.close()
        print(f"[LOG] Progress log saved to: {progress_log_path}")

    # Save final summary
    if last_result is not None:
        final_cm = last_result.get("custom_metrics") or {}
        final_uptime = float("nan")
        for k in ("uptime_rate_mean_mean", "uptime_rate_mean", "uptime_mean", "uptime", "episode_uptime_mean"):
            if k in final_cm:
                final_uptime = float(final_cm[k])
                break
        
        summary = {
            "exp_name": args.exp_name,
            "ablation": args.ablation,
            "seed": args.seed,
            "nesy_lam": args.nesy_lam,
            "timesteps_total": int(last_result.get("timesteps_total", 0)),
            "episode_reward_mean": float(last_result.get("episode_reward_mean")) if last_result.get("episode_reward_mean") is not None else None,
            "episode_len_mean": float(last_result.get("episode_len_mean")) if last_result.get("episode_len_mean") is not None else None,
            "uptime_rate_mean": final_uptime if not np.isnan(final_uptime) else None,
            "logdir": str(log_dir),
        }
        
        # Save final_summary.json
        summary_path = log_dir / "final_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n[FINAL] Summary saved to: {summary_path}")
        print(f"[FINAL] {json.dumps(summary, ensure_ascii=False)}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

import os
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import ray
from ray.rllib.env import EnvContext
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ResultDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# [NEW] 로거 설정을 위해 추가
from ray.tune.logger import UnifiedLogger

try:
    import gymnasium as gym  # noqa
except Exception:
    try:
        import gym  # type: ignore # noqa
    except Exception:
        gym = None

from cyborg_env_maker import create_cyborg_env as cyborg_create_env

torch, _ = try_import_torch()


class CybUptimeSuccessCallbacks(DefaultCallbacks):
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

        atks_raw = [d.get("attack_success") for d in arr if d.get("attack_success") is not None]
        if atks_raw:
            flags = [1 if bool(x) else 0 for x in atks_raw]
            episode.user_data.setdefault("attack_flags", []).extend(flags)

        nesy = [d.get("nesy_bonus") for d in arr if d.get("nesy_bonus") is not None]
        if nesy:
            episode.user_data.setdefault("nesy_bonus_list", []).extend([float(x) for x in nesy])
            
        drops = [d.get("uptime_drop") for d in arr if d.get("uptime_drop") is not None]
        if drops:
            episode.user_data.setdefault("uptime_drops", []).extend([float(x) for x in drops])

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

        atks = episode.user_data.get("attack_flags", [])
        if atks:
            arr = np.array(atks, dtype=np.float32)
            episode.custom_metrics["attack_success_rate"] = float(arr.mean())
            episode.custom_metrics["attack_success_count"] = float(arr.sum())

        nb = episode.user_data.get("nesy_bonus_list", [])
        if nb:
            episode.custom_metrics["nesy_bonus_mean"] = float(np.mean(nb))

        drops = episode.user_data.get("uptime_drops", [])
        if drops:
            episode.custom_metrics["uptime_drop_max"] = float(max(drops))


def _register_env():
    from ray.tune.registry import register_env

    def _env_creator(config: EnvContext):
        return cyborg_create_env(
            seed=int(config.get("seed", 0)),
            max_episode_steps=int(config.get("max_ep_steps", 800)),
            blue_agent=config.get("blue_agent", "blue_agent_0"),
        )

    register_env("CybORG-DroneSwarm", _env_creator)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_metric(eval_metrics: Dict[str, Any], last_metrics: Dict[str, Any], base_key: str):
    for d, key in (
        (eval_metrics, f"{base_key}_mean"),
        (eval_metrics, base_key),
        (last_metrics, f"{base_key}_mean"),
        (last_metrics, base_key),
    ):
        if d is not None and key in d:
            return d.get(key)
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="ray_results/cyborg_swarm_metrics_enhanced")
    ap.add_argument("--exp-name", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--num-envs-per-worker", type=int, default=1)
    ap.add_argument("--max-ep-steps", type=int, default=800)
    ap.add_argument("--local-mode", type=int, default=0)
    ap.add_argument("--ray-temp", type=str, default=None)
    ap.add_argument("--blue-agent", type=str, default="blue_agent_0")
    ap.add_argument("--rollout-fragment-length", type=int, default=128)
    ap.add_argument("--train-batch-size", type=int, default=2048)
    ap.add_argument("--sgd-minibatch-size", type=int, default=256)
    return ap.parse_args()


# [NEW] 커스텀 로거 생성 함수
def custom_log_creator(custom_path, custom_config=None):
    def logger_creator(config):
        if custom_config:
            config.update(custom_config)
        # UnifiedLogger는 CSV, JSON, TensorBoard 로그를 모두 생성합니다.
        return UnifiedLogger(config, custom_path)
    return logger_creator


def main():
    args = parse_args()
    local_dir = Path(args.logdir).resolve()
    _ensure_dir(local_dir)

    # 실험 이름 및 최종 로그 경로 확정
    exp_name = args.exp_name or time.strftime("exp_%Y%m%d_%H%M%S")
    final_log_path = local_dir / exp_name
    _ensure_dir(final_log_path)

    ray_tmp_dir = args.ray_temp or os.environ.get("RAY_TMPDIR") or str(local_dir / "_ray_tmp")
    spill_cfg_env = os.environ.get("RAY_object_spilling_config")
    if not spill_cfg_env:
        spill_dir = str(Path(ray_tmp_dir) / "spill")
        _ensure_dir(Path(spill_dir))
        system_config = {
            "automatic_object_spilling_enabled": True,
            "min_spilling_size": 10 * 1024 * 1024,
            "memory_monitor_refresh_ms": 250,
        }
    else:
        system_config = {"memory_monitor_refresh_ms": 250}

    ray.init(
        local_mode=bool(args.local_mode),
        _temp_dir=ray_tmp_dir,
        _system_config=system_config,
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    _register_env()

    dummy_env = cyborg_create_env(
        seed=int(args.seed),
        max_episode_steps=int(args.max_ep_steps),
        blue_agent=args.blue_agent,
    )
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    if hasattr(dummy_env, "close"):
        dummy_env.close()

    ppo_cfg: PPOConfig = (
        PPOConfig()
        .environment(
            env="CybORG-DroneSwarm",
            env_config={
                "seed": args.seed,
                "max_ep_steps": args.max_ep_steps,
                "blue_agent": args.blue_agent,
            },
            disable_env_checking=True,
            observation_space=obs_space,
            action_space=act_space,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=max(0, int(args.n_workers)),
            num_envs_per_worker=int(args.num_envs_per_worker),
            rollout_fragment_length=int(args.rollout_fragment_length),
            batch_mode="truncate_episodes",
            compress_observations=True,
        )
        .training(
            model={"fcnet_hiddens": [128, 128]},
            train_batch_size=int(args.train_batch_size),
            sgd_minibatch_size=int(args.sgd_minibatch_size),
        )
        .resources(num_gpus=0)
        .callbacks(CybUptimeSuccessCallbacks)
        .debugging(log_level="INFO")
    )
    ppo_cfg.enable_connectors = False

    cfg_dict = ppo_cfg.to_dict()
    cfg_dict["simple_optimizer"] = True
    ppo_cfg = PPOConfig.from_dict(cfg_dict)

    evaluation_num_episodes = max(1, int(args.eval_episodes))
    eval_seed = int(args.eval_seed)
    ppo_cfg = ppo_cfg.evaluation(
        evaluation_interval=None,
        evaluation_num_workers=1,
        evaluation_duration=evaluation_num_episodes,
        evaluation_duration_unit="episodes",
        evaluation_config={"seed": eval_seed, "explore": False},
    )

    # [NEW] 로거 크리에이터 주입! 
    # 이제 progress.csv가 final_log_path (우리가 원하는 폴더)에 저장됩니다.
    algo: Algorithm = ppo_cfg.build(logger_creator=custom_log_creator(str(final_log_path)))

    target_timesteps = int(args.timesteps)
    trained_timesteps = 0
    iteration = 0
    
    try:
        while trained_timesteps < target_timesteps:
            iteration += 1
            result: ResultDict = algo.train()
            trained_timesteps = int(
                result.get("num_env_steps_sampled", result.get("timesteps_total", 0))
            )
            
            # 중간 체크포인트 저장
            if iteration % 10 == 0:
                print(f"[Check] Saving checkpoint at iter {iteration}...")
                algo.save() 

            if trained_timesteps % 5000 == 0 or trained_timesteps >= target_timesteps:
                print(f"[train] {trained_timesteps}/{target_timesteps} timesteps")

        # 최종 체크포인트 저장
        print("[Check] Saving final checkpoint...")
        save_path = algo.save()
        print(f"Final checkpoint saved at: {save_path}")

        eval_results = algo.evaluate()
        last_metrics = result.get("custom_metrics", {}) if "result" in locals() else {}
        eval_metrics: Dict[str, Any] = {}
        try:
            eval_metrics = eval_results.get("evaluation_results", {}).get("custom_metrics", {})
        except Exception:
            pass

        uptime = _pick_metric(eval_metrics, last_metrics, "uptime_rate_mean")
        attack = _pick_metric(eval_metrics, last_metrics, "attack_success_rate")
        nesy_b = _pick_metric(eval_metrics, last_metrics, "nesy_bonus_mean")
        drop_max = _pick_metric(eval_metrics, last_metrics, "uptime_drop_max")

        summary = {
            "seed": args.seed,
            "eval_seed": eval_seed,
            "timesteps_total": trained_timesteps,
            "uptime_rate_mean": float(uptime) if uptime is not None else None,
            "attack_success_rate": float(attack) if attack is not None else None,
            "nesy_bonus_mean": float(nesy_b) if nesy_b is not None else None,
            "uptime_drop_max": float(drop_max) if drop_max is not None else None,
            "logdir": str(final_log_path), # 경로 업데이트
        }

        print("[FINAL] " + json.dumps(summary, ensure_ascii=False))
        # final_summary.json도 같은 곳에 저장
        with open(final_log_path / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    finally:
        try:
            algo.stop()
        except Exception:
            pass
        ray.shutdown()


if __name__ == "__main__":
    main()
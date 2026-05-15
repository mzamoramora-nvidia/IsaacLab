# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless deterministic eval for a Factory rl_games checkpoint.

Runs ``num_envs`` parallel envs for one episode (``num_steps``) with
the deterministic policy mean (no exploration noise), reads the
FactoryEnv's per-step ``curr_engaged`` and ``curr_successes`` masks
every tick, and reports a one-line CSV-friendly summary:

  success_rate       fraction of envs that succeeded at least once
  engage_rate        fraction of envs that engaged at least once
  mean_engage_step   avg step index of first engage (NaN if none)
  mean_success_step  avg step index of first success
  fps                wall-clock policy ticks/sec across the run

Use this to compare backend / OSC-tuning configs without rendering
video. Each config takes ~30-60 s on Newton at num_envs=64,
num_steps=150 (one 10 s episode).

Examples:

    # Newton, current config
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval_policy.py \
        --task Isaac-Factory-NutThread-Direct-v0 \
        --num_envs 64 --num_steps 150 \
        --checkpoint logs/rl_games/Factory/2026-05-09_17-28-18/nn/Factory.pth \
        --label "newton_fkp1000_kd0" \
        presets=newton

    # PhysX baseline
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/eval_policy.py \
        --task Isaac-Factory-NutThread-Direct-v0 \
        --num_envs 64 --num_steps 150 \
        --checkpoint logs/rl_games/Factory/2026-05-09_17-28-18/nn/Factory.pth \
        --label "physx_baseline"
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time

# isort: off
from isaaclab.app import AppLauncher  # noqa: F401  -- imported via launch_simulation

# isort: on

import torch

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Headless deterministic eval for Factory rl_games checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Factory-NutThread-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=150, help="Policy ticks per env (one episode = 150 at 15 Hz).")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, default="config", help="Label printed in the summary row.")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed

        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        backend = "newton" if bool(getattr(env.unwrapped, "_is_newton", False)) else "physx"

        # Reset and roll the deterministic policy. Track first-engage / first-success step per env.
        e = env.unwrapped
        num_envs = args_cli.num_envs
        device = e.device
        sentinel = args_cli.num_steps + 1  # "never" marker for argmax-based tracking
        first_engage = torch.full((num_envs,), sentinel, dtype=torch.long, device=device)
        first_success = torch.full((num_envs,), sentinel, dtype=torch.long, device=device)

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        t0 = time.time()
        with torch.inference_mode():
            for step in range(args_cli.num_steps):
                obs_t = agent.obs_to_torch(obs)
                # Force deterministic: rl_games yaml defaults to False for
                # this task, but for eval we want the policy mean, not
                # samples — otherwise success rate collapses to ~0.
                actions = agent.get_action(obs_t, is_deterministic=True)
                obs, _, _, _ = env.step(actions)

                # FactoryEnv recomputes intermediate values inside step(); these
                # masks read the post-step state. ``check_rot=True`` matches the
                # rl-train ``Metrics/success_rate`` definition.
                engaged = e._get_curr_successes(success_threshold=e.cfg_task.engage_threshold, check_rot=False)
                succeeded = e._get_curr_successes(success_threshold=e.cfg_task.success_threshold, check_rot=True)
                first_engage = torch.where((first_engage == sentinel) & engaged, step, first_engage)
                first_success = torch.where((first_success == sentinel) & succeeded, step, first_success)

        elapsed = time.time() - t0
        fps = args_cli.num_steps / elapsed if elapsed > 0 else float("nan")

        engaged_mask = first_engage < sentinel
        success_mask = first_success < sentinel
        engage_rate = float(engaged_mask.float().mean().item())
        success_rate = float(success_mask.float().mean().item())
        mean_engage_step = (
            float(first_engage[engaged_mask].float().mean().item()) if engaged_mask.any() else float("nan")
        )
        mean_success_step = (
            float(first_success[success_mask].float().mean().item()) if success_mask.any() else float("nan")
        )

        print(
            f"\n[eval] label={args_cli.label} backend={backend} num_envs={num_envs} "
            f"num_steps={args_cli.num_steps} ckpt={os.path.basename(resume_path)}",
            flush=True,
        )
        print(
            f"[eval] success_rate={success_rate:.3f}  engage_rate={engage_rate:.3f}  "
            f"mean_engage_step={mean_engage_step:.1f}  mean_success_step={mean_success_step:.1f}  "
            f"fps={fps:.1f}",
            flush=True,
        )
        print(
            f"CSV,{args_cli.label},{backend},{success_rate:.4f},{engage_rate:.4f},"
            f"{mean_engage_step:.2f},{mean_success_step:.2f},{fps:.2f}",
            flush=True,
        )

        env.close()


if __name__ == "__main__":
    main()

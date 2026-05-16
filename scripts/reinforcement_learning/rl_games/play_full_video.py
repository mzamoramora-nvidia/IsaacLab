# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script that captures every frame into a proper video."""

import argparse
import math
import os
import sys
import types

# Shim ``wp.context`` so the replicator extension's class-body type
# annotations (``wp.context.Kernel``) parse cleanly under newer warp
# versions where ``wp.context`` was removed and ``Kernel`` lives at
# the top level. Replicator is auto-imported by Kit when
# ``enable_cameras=True`` (which we set unconditionally below), so the
# shim must land before ``launch_simulation`` boots Kit.
import warp as wp  # noqa: E402

if not hasattr(wp, "context"):
    wp.context = types.SimpleNamespace(Kernel=getattr(wp, "Kernel", None))

# Shim ``wp.types.array`` (renamed/removed in newer warp; some kit
# render paths still reference it).
if not hasattr(wp.types, "array"):
    wp.types.array = wp.array

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Full video capture play script.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=450)
parser.add_argument("--output", type=str, required=True, help="Output mp4 path")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--seed", type=int, default=42)
# Newton-only viewer-state toggles. They have no effect on the Kit/RTX
# recorder backend (PhysX path), so they're safe to pass unconditionally
# but they're meaningful only with ``presets=newton``.
parser.add_argument(
    "--show_collision", action="store_true", help="(Newton only) draw collision geometry instead of visual meshes."
)
parser.add_argument("--enable_wireframe", action="store_true", help="(Newton only) render scene in wireframe mode.")
parser.add_argument(
    "--show_hydro_contact_surface",
    action="store_true",
    help="(Newton only) overlay the hydroelastic SDF contact surface (the 'isosurface').",
)
parser.add_argument(
    "--show_contacts",
    action="store_true",
    help=(
        "(Newton only) overlay contact-point arrows. The viewer flag is set unconditionally, "
        "but actual arrow drawing requires the recorder to forward contacts to "
        "``viewer.log_contacts`` — currently no-op pending the hydro-surface-viz port to "
        "the post-PR-#5128 SceneDataProvider."
    ),
)
parser.add_argument(
    "--newton_sdf_only",
    action="store_true",
    help=(
        "(Newton only) Run the SDF-only penalty-spring contact mode by nulling "
        "``sdf_hydroelastic_config``. Same Python-side mutation as ``eval_policy.py`` since "
        "Hydra struct-mode rejects the equivalent CLI override against PresetCfg."
    ),
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args


def _p(msg: str) -> None:
    print(f"[play_full_video] {msg}", flush=True)


def main():
    _p(f"start: task={args_cli.task} ckpt={args_cli.checkpoint} output={args_cli.output}")
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rl_games_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        agent_cfg["params"]["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

        if args_cli.newton_sdf_only:
            physics = getattr(env_cfg.sim, "physics", None)
            for candidate in (getattr(physics, "newton", None), physics):
                cc = getattr(candidate, "collision_cfg", None)
                if cc is not None and hasattr(cc, "sdf_hydroelastic_config"):
                    cc.sdf_hydroelastic_config = None
                    _p(f"env_cfg: sdf_hydroelastic_config nulled via {type(candidate).__name__}")
                    break

        # The Factory Newton cfg ships with ``output_contact_surface=False``
        # to save compute, but the viewer's "Show Hydro Surface" overlay
        # has nothing to draw without it. Flip it on for this run so the
        # collision pipeline actually produces the contact-surface mesh.
        #
        # Walk the cfg tree defensively because the Hydra preset
        # resolution may collapse ``physics.newton`` to ``physics`` or
        # keep the PresetCfg wrapper depending on how the run was
        # launched.
        if args_cli.show_hydro_contact_surface:
            physics = getattr(env_cfg.sim, "physics", None)
            for candidate in (getattr(physics, "newton", None), physics):
                cc = getattr(candidate, "collision_cfg", None)
                hcfg = getattr(cc, "sdf_hydroelastic_config", None) if cc is not None else None
                if hcfg is not None and hasattr(hcfg, "output_contact_surface"):
                    hcfg.output_contact_surface = True
                    _p(f"env_cfg: output_contact_surface flipped on via {type(candidate).__name__}")
                    break
            else:
                _p("env_cfg: could not locate output_contact_surface; hydro overlay will be empty")

        resume_path = retrieve_file_path(args_cli.checkpoint)

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        env_unwrapped = env.unwrapped

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)

        # Apply Newton viewer toggles to the recorder's GL viewer. We have
        # to force-trigger lazy-init (via update_camera) before we can
        # reach into the viewer; on the Kit/RTX recorder backend
        # ``_capture`` is a different class, so the inner ``_viewer``
        # attribute simply won't exist and we skip silently.
        if (
            args_cli.show_collision
            or args_cli.enable_wireframe
            or args_cli.show_hydro_contact_surface
            or args_cli.show_contacts
        ):
            recorder = getattr(env_unwrapped, "video_recorder", None)
            capture = getattr(recorder, "_capture", None)
            if capture is not None and hasattr(capture, "update_camera"):
                capture.update_camera(capture.cfg.eye, capture.cfg.lookat)
                viewer = getattr(capture, "_viewer", None)
                if viewer is not None:
                    if args_cli.show_collision:
                        viewer.show_collision = True
                    if args_cli.enable_wireframe and hasattr(viewer, "renderer"):
                        viewer.renderer.draw_wireframe = True
                    if args_cli.show_hydro_contact_surface and hasattr(viewer, "show_hydro_contact_surface"):
                        viewer.show_hydro_contact_surface = True
                    if args_cli.show_contacts:
                        viewer.show_contacts = True
                    _p(
                        f"viewer flags applied: show_collision={args_cli.show_collision} "
                        f"wireframe={args_cli.enable_wireframe} "
                        f"show_hydro_contact_surface={args_cli.show_hydro_contact_surface} "
                        f"show_contacts={args_cli.show_contacts}"
                    )

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        writer = None
        frame_count = 0

        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=True)
                obs, _, dones, _ = env.step(actions)

                if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            try:
                frame = env_unwrapped.render()
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    rgb = frame[:, :, :3]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if writer is None:
                        h, w = bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")
                        writer = cv2.VideoWriter(args_cli.output, fourcc, args_cli.fps, (w, h))
                        if not writer.isOpened():
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(args_cli.output, fourcc, args_cli.fps, (w, h))
                        _p(f"writer opened {w}x{h} -> {args_cli.output}")
                    writer.write(bgr)
                    frame_count += 1
                    if step % 50 == 0:
                        _p(f"step {step}/{args_cli.num_steps}: frame captured (mean={rgb.mean():.1f})")
            except Exception as exc:
                if step % 50 == 0:
                    _p(f"step {step}: render error: {exc}")

        if writer is not None:
            writer.release()
        print(f"\nSaved {frame_count} frames to {args_cli.output}")
        env.close()


if __name__ == "__main__":
    main()

# How to train + record a Factory policy video (PhysX or Newton)

Step-by-step recipe verified on `nut-thread-newton`, warp 1.13,
isaacsim with `omni.replicator.core-1.13.4`. Both backends share the
same training and video-capture scripts — the only difference is
appending `presets=newton` (hydroelastic SDF) or `presets=newton_sdf`
(vanilla SDF, no hydroelastic) to the command line.

There are **two** video-capture scripts:

| Script | What it produces |
|---|---|
| `play_full_video.py` | Plain rollout video — robot + assets only. |
| `play_full_video_with_frames.py` | Same rollout + RGB axis-triad overlay on the nut and bolt. |

## TL;DR

| | PhysX | Newton |
|---|---|---|
| Train | `train.py … --max_iterations 200` | `train.py … --max_iterations 200 presets=newton` |
| Record (no frames) | `play_full_video.py … --checkpoint <ckpt>` | `play_full_video.py … --checkpoint <ckpt> presets=newton` |
| Record (with frames) | `play_full_video_with_frames.py … --checkpoint <ckpt>` | `play_full_video_with_frames.py … --checkpoint <ckpt> presets=newton` |

## 1. Train a policy

### PhysX

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 128 \
    --max_iterations 200 \
    --headless \
    --seed 0
```

- Wall time ≈ **3 h** on a single L4 (step_fps ~ 340). Drop to
  `--max_iterations 30` for a quick smoke test (~19 min, already
  reaches ~83 % success rate).
- Final metrics over a 200-iter run we ran: reward ≈ **798**, success
  rate ≈ **89 %**.

### Newton

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 128 \
    --max_iterations 30 \
    --headless \
    --seed 0 \
    presets=newton
```

- The trailing `presets=newton` is a Hydra arg — must come **after**
  the argparse flags, no `--` prefix.
- Newton training is currently slower wall-clock than PhysX (step_fps
  ~170 with random policy actions, even though the OSC-hold perf probe
  measures Newton at 570 fps vs PhysX at 442). The hydroelastic
  contact pipeline does much more work when the policy is exploring
  vs. holding pose.
- 30 iter is a smoke check — it does not converge yet on Newton with
  current cfg (`success_rate = 0` at iter 30, under investigation).

### Common

- Checkpoint lands at
  `logs/rl_games/Factory/<YYYY-MM-DD_HH-MM-SS>/nn/Factory.pth` (204 MB).
- Tensorboard scalars in the matching `summaries/` dir. Key tags:
  `rewards/iter`, `Episode/Metrics/success_rate`,
  `logs_rew_curr_engaged/iter`, `logs_rew_curr_success/iter`,
  `performance/step_fps`.

Quick reward-progress check from CLI:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/rl_games/Factory/<RUN_DIR>/summaries')
ea.Reload()
evs = ea.Scalars('rewards/iter')
print(f'reward: {evs[0].value:.2f} -> {evs[-1].value:.2f}  (n={len(evs)})')
```

## 2. Record a plain video (no frame overlay)

### PhysX

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play_full_video.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 1 \
    --num_steps 450 \
    --checkpoint logs/rl_games/Factory/<RUN_DIR>/nn/Factory.pth \
    --output /tmp/physx_policy.mp4 \
    --fps 15
```

### Newton

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play_full_video.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 1 \
    --num_steps 450 \
    --checkpoint logs/rl_games/Factory/<RUN_DIR>/nn/Factory.pth \
    --output /tmp/newton_policy.mp4 \
    --fps 15 \
    presets=newton
```

## 3. Record a video with nut + bolt frame overlays

Replace `play_full_video.py` with `play_full_video_with_frames.py`,
plus an optional `--frame_scale` flag (axis length in metres, default
5 cm). Frame color convention is RGB → XYZ.

### PhysX

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play_full_video_with_frames.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 1 \
    --num_steps 450 \
    --checkpoint logs/rl_games/Factory/<RUN_DIR>/nn/Factory.pth \
    --output /tmp/physx_policy_frames.mp4 \
    --fps 15 \
    --frame_scale 0.05
```

### Newton

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play_full_video_with_frames.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 1 \
    --num_steps 450 \
    --checkpoint logs/rl_games/Factory/<RUN_DIR>/nn/Factory.pth \
    --output /tmp/newton_policy_frames.mp4 \
    --fps 15 \
    --frame_scale 0.05 \
    presets=newton
```

### How the frame overlays are drawn

The two backends use **different renderers**, so the script dispatches
internally:

- **PhysX** → Kit RTX camera. Frames are spawned as
  `VisualizationMarkers` (USD `PointInstancer` from `FRAME_MARKER_CFG`)
  under `/World/Visuals/{nut,bolt}_frame`. Per-step `marker.visualize(
  translations, orientations)` updates the USD attributes; Kit
  re-renders through Fabric automatically.

- **Newton** → `NewtonGlPerspectiveVideo` → Newton's OpenGL
  `ViewerGL` (no Kit at all). USD writes are invisible to that
  pipeline because Newton only Fabric-syncs prims it tracks. So the
  script grabs the recorder's viewer instance
  (`env.video_recorder._capture._viewer`) and calls
  `viewer.log_lines("debug/{nut,bolt}_axis_{0,1,2}", starts, ends,
  colors)` directly — the same API the
  `FactoryNewtonDebugPanel._draw_axis_triads` method uses
  interactively. Frames are thin lines rather than 3D arrows because
  that's all `log_lines` does.

### Quat-convention gotcha (Newton)

`isaaclab_newton.assets.rigid_object.rigid_object_data` documents
`root_link_quat_w` as **(x, y, z, w)** (line 556). The IsaacLab
default elsewhere is **(w, x, y, z)**. If you roll your own frame
overlay, pick the right layout for the backend — getting it wrong
makes the bolt look rotated by some fixed offset and the nut's frame
drift out of sync as the body rotates. The current script handles
this via a `quat_layout` switch.

## 4. Common cfg / workarounds (top of `play_full_video.py` and `…_with_frames.py`)

Both scripts ship with two warp-ABI shims needed under warp 1.13 +
isaacsim's `omni.replicator.core-1.13.4`. If you upgrade warp /
replicator and these become no-ops, you can remove them.

| Shim | Why |
|---|---|
| `wp.context = SimpleNamespace(Kernel=wp.Kernel)` | replicator's `Augmentation` class body annotates `wp.context.Kernel`; newer warp moved `Kernel` to the top level and dropped `wp.context`. |
| `wp.types.array = wp.array` | a kit render path still references `wp.types.array`, also gone in newer warp. |

`DirectRLEnv.seed()` also catches `(ModuleNotFoundError, AttributeError)`
around `import omni.replicator.core` (the old code only caught
`ModuleNotFoundError`, but the warp-context failure raises
`AttributeError`).

## 5. Quick sanity check (skip training)

If you already have a checkpoint and just want to verify the video
pipeline works, run with a few steps:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play_full_video.py \
    --task Isaac-Factory-NutThread-Direct-v0 \
    --num_envs 1 \
    --num_steps 10 \
    --checkpoint <path>/Factory.pth \
    --output /tmp/smoke.mp4 \
    [presets=newton]
```

You should see
```
[play_full_video] step 0/10: frame captured (mean=…)
[play_full_video] writer opened 1280x720 -> /tmp/smoke.mp4
```
and a small mp4 in `/tmp`.

## 6. Verified outputs

Last confirmed by direct test:

| Backend | Source checkpoint | Plain video | Video with frames |
|---|---|---|---|
| PhysX (200 iter, 89 % success) | `logs/rl_games/Factory/2026-05-09_17-28-18/nn/Factory.pth` | `/tmp/verify_physx.mp4` (3.4 MB, 450 frames) | `/tmp/physx_policy_frames_v2.mp4` (3.8 MB, 450 frames) |
| Newton (30 iter, smoke) | `logs/rl_games/Factory/2026-05-11_09-26-15/nn/Factory.pth` | `/tmp/verify_newton.mp4` (2.7 MB, 450 frames) | `/tmp/newton_policy_frames_v2.mp4` (2.8 MB, 450 frames) |

`ffprobe` confirms all four videos are 1280×720 mpeg4 @ 15 fps,
450 frames, 30 s duration.

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: procedural Newton-only post-load asset setup.

Two entry points:

* :func:`register_model_init_callback` — registers a Newton ``MODEL_INIT``
  callback that runs *before* model finalization, where the
  :class:`newton.ModelBuilder` is still mutable. This is where the
  MuJoCo-Warp gravity-compensation custom attributes
  (``mujoco:jnt_actgravcomp`` per DOF, ``mujoco:gravcomp`` per body) and
  the joint armature have to be written, since they're consumed during
  finalize. Called from :meth:`FactoryEnv._setup_scene`.

* :func:`apply` — small post-finalization fix-ups that need the
  finalized :class:`newton.Model` (for example refreshing the OSC
  buffers' cached armature). Called from :meth:`FactoryEnv.__init__`
  right after ``super().__init__()`` returns.

PhysX runs never import this module.

Reference: panda-osc work in the Newton repo (commits ``06b3b087``,
``1a67437b``) and the ``mzamora/newton-dev`` Factory branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import warp as wp

if TYPE_CHECKING:
    from .factory_env import FactoryEnv

_ARM_DOF_COUNT = 7

_ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
# Path-prefix used to identify every body that belongs to the Franka
# articulation in IsaacLab's USD layout (``/World/envs/env_*/Robot/...``).
# Path-based matching is more robust than a hand-curated name list —
# panda-nut-bolt-osc brackets ``builder.body_count`` across the
# ``builder.add_usd(...)`` call to grab everything the parser added,
# but in IsaacLab we run from the MODEL_INIT callback after the cloner
# has already replicated the prototype N times, so we can't use the
# count-bracket trick. Matching on ``/Robot/`` substring catches every
# body the Robot articulation contributed (links, hand, fingers, force
# sensor, fingertip-centered virtual frame, plus any camera mount /
# tool the user might attach later) without us having to maintain an
# explicit list and risk missing one.
_ROBOT_BODY_PATH_SUBSTR = "/Robot/"

# Arm DOF armature [kg·m²]. Higher armature stabilises the OSC's
# ``Lambda = (J H^-1 J^T)^-1`` and damps high-frequency torque on the
# wrist joints. Numbers from the panda-osc reference / mzamora/newton-dev.
_ARM_ARMATURE = (0.3, 0.3, 0.3, 0.3, 0.11, 0.11, 0.11)
_FINGER_ARMATURE = 0.15


def register_model_init_callback(env: FactoryEnv) -> None:
    """Wire a Newton MODEL_INIT callback that mutates the builder.

    Args:
        env: The :class:`FactoryEnv`. The callback closes over ``env`` so it
            can read the current ctrl-mode flag at fire time.
    """
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        lambda _ev: _model_init_callback(env), PhysicsEvent.MODEL_INIT, name="factory_newton_setup"
    )


def apply(env: FactoryEnv) -> None:
    """Post-finalization fix-ups (refresh OSC buffers' cached armature).

    Called from :meth:`FactoryEnv.__init__` after ``super().__init__()``.
    """
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    if model is None:
        return
    _refresh_osc_buffers_armature(model, env)


# ---------------------------------------------------------------------------
# Builder-time setup (MODEL_INIT callback body).
# ---------------------------------------------------------------------------


def _model_init_callback(env: FactoryEnv) -> None:
    """Body of the MODEL_INIT callback. Operates on the live builder."""
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    if builder is None:
        return

    use_ik = _is_ik_mode(env)
    _set_joint_target_mode(builder, use_ik=use_ik)
    _set_ctrl_source_joint_target(builder)
    _set_per_dof_gravity_compensation(builder)
    _set_per_body_gravity_compensation(builder)
    _filter_base_table_contacts(builder)
    _tune_nut_bolt_contacts(builder)
    _build_collision_sdfs(builder)
    # NOTE: Per-DOF armature is set on the Franka via ``ImplicitActuatorCfg``
    # in :class:`FactoryEnvCfg.robot.actuators`, not via the builder.
    # Newton's finalize step seeds ``model.joint_armature`` for the robot's
    # DOFs from the actuator cfg and overwrites any per-DOF builder writes
    # we make at those indices. The actuator-cfg path is the only reliable
    # way to apply armature to a parsed-USD articulation today.


def _is_ik_mode(env: FactoryEnv) -> bool:
    """Return True iff the Factory ctrl cfg requested IK control on Newton.

    Defaults to OSC mode (which keeps arm DOFs in EFFORT mode so the
    OSC's torque writes via ``Control.joint_f`` drive the arm directly).
    """
    ctrl_mode = getattr(env.cfg.ctrl, "newton_ctrl_mode", "osc")
    return ctrl_mode == "ik"


def _joint_label_indices(builder, name_substrs: list[str]) -> list[int]:
    """Return joint indices whose label contains any of ``name_substrs``."""
    return [i for i, label in enumerate(builder.joint_label) if any(s in label for s in name_substrs)]


def _joint_indices_to_dof_indices(builder, joint_idxs: list[int]) -> list[int]:
    """Translate joint indices into DOF indices via ``joint_qd_start``.

    Newton's per-DOF arrays (``joint_target_mode``, ``joint_target_ke``, …)
    are indexed by *DOF*, not joint. With free-floating joints in the
    scene (each contributes 6 qd entries but only 1 joint label),
    ``joint_index != dof_index`` past env 0. ``joint_qd_start[j]`` gives
    the qd-index where joint ``j``'s DOFs begin; ``joint_qd_start[j+1]``
    gives the end. We walk that range so a finger (1-DOF revolute) gives
    1 entry and a free joint (6-DOF) would give 6 — though we only call
    this for finite-DOF revolute joints.
    """
    qd_start = list(builder.joint_qd_start)
    total_dofs = qd_start[-1] if qd_start else 0
    out: list[int] = []
    for j in joint_idxs:
        dof_lo = int(qd_start[j])
        dof_hi = int(qd_start[j + 1]) if (j + 1) < len(qd_start) else int(total_dofs)
        out.extend(range(dof_lo, dof_hi))
    return out


def _arm_dof_indices(builder) -> list[int]:
    """DOF indices in ``builder.joint_target_mode`` for the arm joints."""
    js = _joint_label_indices(builder, _ARM_JOINT_NAMES)
    return _joint_indices_to_dof_indices(builder, js)


def _finger_dof_indices(builder) -> list[int]:
    """DOF indices in ``builder.joint_target_mode`` for the finger joints."""
    js = _joint_label_indices(builder, _FINGER_JOINT_NAMES)
    return _joint_indices_to_dof_indices(builder, js)


def _robot_body_indices(builder) -> list[int]:
    """Indices into ``builder.body_label`` that belong to the Franka.

    Matched by USD prim-path substring (``/Robot/``) — the IsaacLab
    convention for the robot articulation hierarchy. Catches every
    body under the articulation regardless of whether it's an arm
    link, the gripper, the fingertip-centered virtual frame, the
    force sensor, or any camera/tool the user might add later.
    Equivalent in spirit to panda-nut-bolt-osc's body-count bracket
    around ``builder.add_usd(...)``.
    """
    return [i for i, label in enumerate(builder.body_label) if _ROBOT_BODY_PATH_SUBSTR in label]


def _set_joint_target_mode(builder, use_ik: bool) -> None:
    """Force every robot DOF (arm + fingers) into POSITION mode.

    The arm runs with ``joint_target_ke = 0`` (no spring), so the position
    target has no effect on the arm — but ``joint_target_kd > 0`` then gives
    mjwarp a per-joint ``-kd * qd`` damping term that kills the redundant-7th
    DOF mode the OSC would otherwise have to chase. The OSC's own torques
    written to ``Control.joint_f`` are added on top.

    Pattern from ``newton/examples/robot/example_robot_panda_nut_bolt_osc.py``
    (lines 1240-1270) — explicit comment there:

        "A small joint_target_kd on the arm gives MuJoCo a -kd*qd damping
        torque per joint, which kills the slow under-damped mode the
        redundant 7th DOF would otherwise exhibit when the OSC tracks
        a moving target."

    Without this we leave the arm in the parser's EFFORT-mode default,
    where mjwarp ignores ``joint_target_kd`` entirely and the OSC has
    to provide all damping itself.

    The ``use_ik`` argument is now redundant (POSITION mode is correct
    for both control modes); kept for backwards compatibility with
    existing callers.
    """
    del use_ik
    for dof_idx in _arm_dof_indices(builder) + _finger_dof_indices(builder):
        builder.joint_target_mode[dof_idx] = int(newton.JointTargetMode.POSITION)


def _set_ctrl_source_joint_target(builder) -> None:
    """Pin every robot DOF's ``mujoco:ctrl_source`` to ``JOINT_TARGET``.

    With ``mujoco:ctrl_source = JOINT_TARGET``, mjwarp reads the actuator's
    target from ``Control.joint_target_pos`` (which IsaacLab's
    ``set_joint_position_target_index`` writes) instead of the unused
    ``Control.mujoco.ctrl`` array. Required for the POSITION-mode arm
    to actually consume our position targets.

    Same pattern as ``example_robot_panda_nut_bolt_osc.py:1268-1270``.
    """
    from newton.solvers import SolverMuJoCo

    custom = builder.custom_attributes.get("mujoco:ctrl_source")
    if custom is None:
        return
    n_acts = len(builder.joint_target_mode)
    if custom.values is None or len(custom.values) != n_acts:
        custom.values = [int(SolverMuJoCo.CtrlSource.JOINT_TARGET)] * n_acts
    else:
        target_value = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)
        for dof_idx in _arm_dof_indices(builder) + _finger_dof_indices(builder):
            custom.values[dof_idx] = target_value


def _set_per_dof_gravity_compensation(builder) -> None:
    """Enable MuJoCo-Warp's per-DOF actuator gravcomp for the arm.

    ``mujoco:jnt_actgravcomp`` is a custom attribute consumed by the
    MuJoCo-Warp solver. When ``True`` for an actuator's joint, MJW adds
    a feed-forward torque equal to the gravity load along that DOF —
    that is, the OSC no longer has to fight gravity through task-space
    PD, and torque-at-zero-error becomes ~zero. Without this, our
    Jacobian-transpose OSC drifts the arm by hundreds of mm during
    rollouts.
    """
    custom = builder.custom_attributes.get("mujoco:jnt_actgravcomp")
    if custom is None:
        return
    if custom.values is None:
        custom.values = {}
    for dof_idx in _arm_dof_indices(builder):
        custom.values[dof_idx] = True


def _filter_base_table_contacts(builder) -> None:
    """Filter spurious robot-base ↔ table contacts.

    The Franka base (``panda_link0``, ``panda_link1``) sits on the table.
    Without explicit collision-filter pairs, mjwarp generates a continuous
    contact between the base collision mesh and the table top surface every
    step, even when there's no physical interpenetration. Those tiny contact
    forces propagate through the kinematic chain and show up as
    multi-millimeter TCP drift during OSC hold (verified via
    ``scripts/probe_active_contacts.py``: 98 active contacts on R0 hold,
    including ``Table <-> panda_link0`` firing every tick).

    Mirrors the pattern used by ``newton/examples/robot/example_robot_panda_osc.py``
    (lines 334-339), which filters base ↔ ground at scene-build time. On our
    setup we filter base ↔ table.
    """
    base_suffixes = ("/panda_link0", "/panda_link1")
    table_substr = "/Table"
    base_shape_idxs = [
        i
        for i, body_idx in enumerate(builder.shape_body)
        if 0 <= body_idx < len(builder.body_label) and builder.body_label[body_idx].endswith(base_suffixes)
    ]
    table_shape_idxs = [
        i
        for i, body_idx in enumerate(builder.shape_body)
        if 0 <= body_idx < len(builder.body_label) and table_substr in builder.body_label[body_idx]
    ]
    if not base_shape_idxs or not table_shape_idxs:
        return
    for base_i in base_shape_idxs:
        for table_i in table_shape_idxs:
            pair = (min(base_i, table_i), max(base_i, table_i))
            builder.shape_collision_filter_pairs.append(pair)
    print(f"[factory-newton] filtered {len(base_shape_idxs) * len(table_shape_idxs)} base↔table collision pairs")


def _tune_nut_bolt_contacts(builder) -> None:
    """Apply panda-osc / mzamora/newton-dev contact-material tuning.

    The Factory NutThread task is contact-rich: the gripper has to
    close around the nut without the nut squirting out, then the nut
    has to slide down the bolt threads on contact. Newton's parser
    uses default low-stiffness contact materials that produce
    "trampoline" behaviour when the gripper closes — the nut bounces
    out of the gripper. Override the per-shape material gains on
    every nut/bolt collision shape:

    * ``shape_material_mu`` — friction. ``0.2`` on the **nut**, ``0.5``
      on the **bolt**. mjwarp combines pair friction as ``max(mu_a, mu_b)``,
      so with finger ``mu ≈ 1.0`` the effective frictions are:
        - nut↔bolt   = max(0.2, 0.5) = 0.5  (stiction on threads)
        - nut↔finger = max(0.2, 1.0) = 1.0  (strong grip)
        - bolt↔finger = max(0.5, 1.0) = 1.0
      The nut was previously at ``0.0`` for the same effective frictions
      (max() picks the larger operand) but mjwarp emits a warning at
      startup — ``geom N: friction[0] (0.0) < MJ_MINMU (1e-05) with
      condim=3 may cause NaN`` — when any shape has ``mu < 1e-5``.
      Bumping to ``0.2`` (matches the panda-nut-bolt OSC reference) kills
      the warning without changing any pair-friction outcome.
    * ``shape_material_ke`` / ``kd`` — normal stiffness / damping
      (1e4 / 100). 1-2 orders of magnitude stiffer than the parser
      default; needed for the gripper-close to produce a stable
      grip rather than a soft squish.
    * ``shape_gap = 0.0`` — no contact margin so the nut sits on
      the bolt at the right height.
    """
    if not all(hasattr(builder, attr) for attr in ("shape_label", "shape_material_mu", "shape_gap")):
        return
    for i in range(builder.shape_count):
        label = str(builder.shape_label[i]).lower()
        if "nut" in label:
            builder.shape_material_mu[i] = 0.2
            builder.shape_material_ke[i] = 1.0e4
            builder.shape_material_kd[i] = 100.0
            builder.shape_gap[i] = 0.0
        elif "bolt" in label:
            builder.shape_material_mu[i] = 0.5
            builder.shape_material_ke[i] = 1.0e4
            builder.shape_material_kd[i] = 100.0
            builder.shape_gap[i] = 0.0


def _set_per_body_gravity_compensation(builder) -> None:
    """Enable MuJoCo-Warp's per-body gravcomp on every Franka body.

    The body-level ``mujoco:gravcomp`` attribute scales the gravity
    contribution of each body's mass for actuator load computation
    (1.0 = full compensation). Set on every parser-added robot body
    so the per-DOF gravcomp above sees the right total mass.
    """
    custom = builder.custom_attributes.get("mujoco:gravcomp")
    if custom is None:
        return
    if custom.values is None:
        custom.values = {}
    for body_idx in _robot_body_indices(builder):
        custom.values[body_idx] = 1.0


# ---------------------------------------------------------------------------
# SDF / hydroelastic collision setup
# ---------------------------------------------------------------------------
#
# Mirrors the ``dev/mzamoramora/factory-sim2sim`` (panda-nut-bolt) setup:
#
# * Fingers — 192-cube SDF, ``HYDROELASTIC`` flag, ``kh=1e10`` (10× softer
#   than nut/bolt so the pads compress sub-mm against the nut surface and
#   build real grip force without visibly interpenetrating the visual
#   mesh), ``mu_torsional=0.1``, MuJoCo ``condim=4`` so the torsional
#   friction constraint is actually solved (default condim=3 silently
#   ignores ``mu_torsional``).
#
# * Nut + bolt — 256-cube SDF, ``HYDROELASTIC``, ``kh=1e11`` (rigid). High
#   resolution is needed because the M16 thread pitch is ~2 mm.
#
# * Other panda links — 64-cube SDF only (no ``HYDROELASTIC``). Used by
#   MuJoCo for fast distance lookups instead of BVH walks; doesn't add
#   hydroelastic forces.
#
# These are *builder-time* writes — they only configure the geometry side.
# To engage the hydroelastic forces themselves we need to wire the solver
# with ``use_mujoco_contacts=False`` + ``sdf_hydroelastic_config`` (which
# routes external SDF contacts into mjwarp). That's a separate NewtonCfg
# change; without it, MuJoCo runs its own collision detection and the
# SDFs are available only for distance queries.

# SDF resolutions (cube edge). Match panda-nut-bolt's profile.
_SDF_RES_FINGER = 192
_SDF_RES_NUT_BOLT = 256
_SDF_RES_PANDA = 64
# Table is large + flat (1.2 × 0.6 × 0.04 m). 32³ → ~37 mm cells in xy,
# ~1.25 mm in z — plenty for a "rest on a flat surface" SDF without
# burning memory on a high-res grid.
_SDF_RES_TABLE = 32
_SDF_BAND_FINGER = (-0.01, 0.01)
_SDF_BAND_NUT_BOLT = (-0.005, 0.005)
_SDF_BAND_PANDA = (-0.01, 0.01)
# Wider outside band (2 cm) so the bolt + nut see the table SDF from
# slightly above, with a narrower inside band — nothing should ever
# be deep inside the table.
_SDF_BAND_TABLE = (-0.005, 0.02)

# Hydroelastic stiffness [Pa/m].
# panda-nut-bolt-osc default is 1e11 everywhere (rigid fingers + nut + bolt);
# factory-sim2sim policy-rollout drops finger kh to 1e10 so the pads
# compress sub-mm against the nut SDF for a one-shot grasp init.
# We follow the OSC profile here because Factory's PD finger close is
# aggressive — at 1e10 the finger SDF was visibly punching through the
# nut SDF since hydroelastic counter-force / penetration_volume was
# too low to push back ~187 N of finger PD force.
_KH_FINGER = 1e11
_KH_NUT_BOLT = 1e11

# Finger-only friction extras.
_FINGER_MU_TORSIONAL = 0.1
_FINGER_CONDIM = 4

_FINGER_BODY_NAMES = ("panda_leftfinger", "panda_rightfinger")
_NUT_BODY_SUBSTRS = ("HeldAsset/factory_nut_loose",)
_BOLT_BODY_SUBSTRS = ("FixedAsset/factory_bolt_loose",)
_TABLE_BODY_SUBSTRS = ("/Table",)


def _build_collision_sdfs(builder) -> None:
    """Build SDFs on Factory's collision meshes for hydroelastic contacts.

    Categorises every collidable mesh shape by body label, builds an
    SDF at the resolution appropriate for that category, and (for
    fingers + nut + bolt) flips the ``HYDROELASTIC`` shape flag and
    writes ``shape_material_kh`` + ``shape_material_mu_torsional``.

    Idempotent per mesh: each :class:`newton.Mesh` whose ``sdf`` is
    already populated is skipped, so re-runs on a hot builder don't
    re-bake the SDF grids. Multiple shapes can share the same mesh,
    but mesh.build_sdf is only called once per unique mesh.
    """
    finger_body_idxs = {i for i, label in enumerate(builder.body_label) if any(n in label for n in _FINGER_BODY_NAMES)}
    nut_body_idxs = {i for i, label in enumerate(builder.body_label) if any(s in label for s in _NUT_BODY_SUBSTRS)}
    bolt_body_idxs = {i for i, label in enumerate(builder.body_label) if any(s in label for s in _BOLT_BODY_SUBSTRS)}
    table_body_idxs = {i for i, label in enumerate(builder.body_label) if any(s in label for s in _TABLE_BODY_SUBSTRS)}
    panda_body_idxs = set(_robot_body_indices(builder)) - finger_body_idxs

    meshlike = (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
    counts = {
        "finger": 0,
        "nut": 0,
        "bolt": 0,
        "panda": 0,
        "table": 0,
        "skip_no_mesh": 0,
        "skip_already_built": 0,
    }

    condim_attr = builder.custom_attributes.get("mujoco:condim")
    if condim_attr is not None and condim_attr.values is None:
        condim_attr.values = {}

    for shape_idx, body_idx in enumerate(builder.shape_body):
        if int(builder.shape_type[shape_idx]) not in (int(t) for t in meshlike):
            continue
        if not (int(builder.shape_flags[shape_idx]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue

        mesh = builder.shape_source[shape_idx]
        if mesh is None:
            counts["skip_no_mesh"] += 1
            continue

        if body_idx in finger_body_idxs:
            category = "finger"
            res, band = _SDF_RES_FINGER, _SDF_BAND_FINGER
        elif body_idx in nut_body_idxs:
            category = "nut"
            res, band = _SDF_RES_NUT_BOLT, _SDF_BAND_NUT_BOLT
        elif body_idx in bolt_body_idxs:
            category = "bolt"
            res, band = _SDF_RES_NUT_BOLT, _SDF_BAND_NUT_BOLT
        elif body_idx in panda_body_idxs:
            category = "panda"
            res, band = _SDF_RES_PANDA, _SDF_BAND_PANDA
        elif body_idx in table_body_idxs:
            # Voxel SDF for visualisation under "Show Collision". Not
            # flagged HYDROELASTIC — we want the bolt-on-table contact
            # to stay on Newton's default rigid pipeline so it doesn't
            # compete with the finger / nut / bolt hydroelastic mass
            # balance.
            category = "table"
            res, band = _SDF_RES_TABLE, _SDF_BAND_TABLE
        else:
            continue

        if mesh.sdf is None:
            # Bake non-unit shape_scale into the mesh vertices first —
            # ``mesh.build_sdf`` doesn't honour shape_scale, so a mesh
            # with shape_scale != 1 ends up with an SDF in the wrong
            # coordinate system. Resetting shape_scale to (1,1,1) afterwards
            # keeps shape vs. mesh in sync.
            shape_scale = builder.shape_scale[shape_idx]
            scale_arr = (float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2]))
            if not (
                abs(scale_arr[0] - 1.0) < 1e-6 and abs(scale_arr[1] - 1.0) < 1e-6 and abs(scale_arr[2] - 1.0) < 1e-6
            ):
                import numpy as _np  # noqa: PLC0415

                scaled_verts = mesh.vertices * _np.asarray(scale_arr, dtype=_np.float32)
                mesh = mesh.copy(vertices=scaled_verts, recompute_inertia=True)
                builder.shape_source[shape_idx] = mesh
                builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
            mesh.build_sdf(max_resolution=res, narrow_band_range=band, margin=abs(band[1]))
            counts[category] += 1
        else:
            counts["skip_already_built"] += 1

        if category in ("finger", "nut", "bolt"):
            builder.shape_flags[shape_idx] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_idx] = _KH_FINGER if category == "finger" else _KH_NUT_BOLT
        if category == "finger":
            builder.shape_material_mu_torsional[shape_idx] = _FINGER_MU_TORSIONAL
            if condim_attr is not None:
                condim_attr.values[shape_idx] = _FINGER_CONDIM

    print(
        f"[factory-newton] built SDFs: finger={counts['finger']}, nut={counts['nut']}, "
        f"bolt={counts['bolt']}, panda={counts['panda']}, table={counts['table']} "
        f"(skipped: no_mesh={counts['skip_no_mesh']}, already_built={counts['skip_already_built']})"
    )


# ---------------------------------------------------------------------------
# Post-finalization fix-ups.
# ---------------------------------------------------------------------------


def _refresh_osc_buffers_armature(model, env: FactoryEnv) -> None:
    """Sync ``arm_armature_torch`` in the OSC buffers with the finalized model.

    ``factory_control_newton`` patches the OSC mass matrix's diagonal
    with ``arm_armature_torch``; if we wrote new armature in the
    builder callback, refresh the cached torch tensor so the OSC's
    ``H + diag(armature)`` matches what the integrator uses.
    """
    osc_buffers = getattr(env, "_newton_osc_buffers", None)
    if osc_buffers is None:
        return
    import torch  # noqa: PLC0415

    armature_np = model.joint_armature.numpy()[:_ARM_DOF_COUNT].copy()
    new_arm_armature = torch.as_tensor(armature_np, device=osc_buffers.arm_armature_torch.device, dtype=torch.float32)
    osc_buffers.arm_armature_torch.copy_(new_arm_armature)


# ---------------------------------------------------------------------------
# Deferred helpers (require per-shape SDF building before they're useful).
# ---------------------------------------------------------------------------


def _flag_robot_collisions_hydroelastic(model, env: FactoryEnv) -> None:
    """Set the ``HYDROELASTIC`` bit on collision shapes belonging to the robot.

    Newton's hydroelastic stack needs an actual :class:`newton.SDF`
    built per shape to do anything useful; on its own the flag is a
    no-op. Kept defined for the future SDF-build helper.
    """
    shape_labels = list(getattr(model, "shape_label", []) or [])
    if not shape_labels:
        return

    shape_flags = model.shape_flags.numpy().copy()
    collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    hydro_bit = int(newton.ShapeFlags.HYDROELASTIC)
    for i, label in enumerate(shape_labels):
        if "/Robot/" not in label:
            continue
        if (int(shape_flags[i]) & collide_bit) == 0:
            continue
        shape_flags[i] = int(shape_flags[i]) | hydro_bit

    wp.copy(model.shape_flags, wp.array(shape_flags, dtype=model.shape_flags.dtype, device=model.device))


def _set_robot_contact_gap(model, env: FactoryEnv, gap: float = 0.005) -> None:
    """Set the per-shape contact gap on robot collision shapes.

    Causes intermittent NaN observations when combined with the
    triangle-mesh contact stack we currently use. Stays disabled
    until SDF contacts are in place.
    """
    shape_labels = list(getattr(model, "shape_label", []) or [])
    if not shape_labels:
        return

    shape_gap = model.shape_gap.numpy().copy()
    collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_flags = model.shape_flags.numpy()
    for i, label in enumerate(shape_labels):
        if "/Robot/" not in label:
            continue
        if (int(shape_flags[i]) & collide_bit) == 0:
            continue
        shape_gap[i] = gap

    wp.copy(model.shape_gap, wp.array(shape_gap, dtype=model.shape_gap.dtype, device=model.device))

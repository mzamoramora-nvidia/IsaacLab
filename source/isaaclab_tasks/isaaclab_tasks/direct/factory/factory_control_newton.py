# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: Newton-backend data fetch for the OSC controller.

This module is the Newton companion to :mod:`factory_control`. The OSC math
(``compute_dof_torque``) lives there and is backend-agnostic. The PhysX path
feeds it via ``root_view.get_jacobians()`` / ``get_generalized_mass_matrices()``.
On Newton those APIs do not exist, but :class:`newton.selection.ArticulationView`
exposes equivalent ``eval_jacobian`` / ``eval_mass_matrix`` calls. This module
adapts those into the shape Factory's OSC math expects, plus two patches
required for parity:

1. The arm jacobian must average the left/right finger body jacobians,
   matching the PhysX path's ``(left + right) * 0.5`` semantics.
2. ``ArticulationView.eval_mass_matrix`` returns ``J^T M J`` only — it does
   *not* fold ``joint_armature`` into ``H``'s diagonal. We add it here at the
   consumer site; lift into ``eval_mass_matrix`` once it lands upstream and
   remove this patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import eval_fk

# Per-joint torque ceiling for the Franka FR3 (datasheet). Matches the
# ``effort_limit_sim`` values in ``isaaclab_assets.robots.franka``. Used by
# :func:`clamp_to_effort_limits` on the Newton path because Newton's
# articulation drive does not enforce ``effort_limit_sim`` on direct
# ``joint_f`` writes.
FRANKA_FR3_EFFORT_LIMITS: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


@dataclass
class NewtonOSCBuffers:
    """Pre-allocated Warp buffers for one OSC tick under the Newton backend.

    Constructed once per env in :meth:`FactoryEnv._init_tensors` (Newton path
    only). Avoids per-step ``wp.array`` allocation inside ``eval_jacobian`` /
    ``eval_mass_matrix`` and caches per-articulation joint indices for the
    fingertip bodies.

    Attributes:
        view: The Newton :class:`~newton.selection.ArticulationView` for the robot.
        n_arm_dofs: Number of arm DOFs (7 for Franka).
        j_full: ``(model.articulation_count, n_joints*6, n_dofs)`` Jacobian buffer.
            Sized to the model's full articulation count because Newton's
            ``eval_jacobian`` kernel indexes by global articulation id, not by
            view-local index.
        h_full: ``(model.articulation_count, n_dofs, n_dofs)`` mass-matrix buffer.
            Same global-index convention as :attr:`j_full`.
        joint_S_s: Per-DOF spatial-vector workspace for ``eval_*``.
        body_I_s: Per-body spatial-inertia workspace for ``eval_mass_matrix``.
        left_joint_in_art: Per-articulation joint index whose child is the
            left finger body.
        right_joint_in_art: Per-articulation joint index whose child is the
            right finger body.
        fingertip_joint_in_art: Per-articulation joint index whose child is
            the fingertip body (``panda_fingertip_centered``). Used by the
            Jacobian getter so the OSC reads a Jacobian computed at the
            same body-origin point as ``fingertip_midpoint_pos``. Falls
            back to ``-1`` when the body has no incoming joint slot, in
            which case the legacy left/right average is used.
        arm_armature_torch: ``(n_arm_dofs,)`` cached arm-armature tensor used
            to patch ``H``'s diagonal.
        articulation_indices_torch: ``(num_envs,)`` long tensor of the model
            articulation indices that belong to this view, used to gather the
            robot rows of ``j_full`` / ``h_full``.
    """

    view: object
    n_arm_dofs: int
    j_full: wp.array
    h_full: wp.array
    joint_S_s: wp.array
    body_I_s: wp.array
    left_joint_in_art: int
    right_joint_in_art: int
    fingertip_joint_in_art: int
    arm_armature_torch: torch.Tensor
    articulation_indices_torch: torch.Tensor


def build_buffers(
    robot,
    left_finger_body_idx: int,
    right_finger_body_idx: int,
    fingertip_body_idx: int = -1,
    n_arm_dofs: int = 7,
) -> NewtonOSCBuffers:
    """Allocate Newton OSC buffers and resolve fingertip joint indices once.

    Args:
        robot: An :class:`isaaclab_newton.assets.Articulation` instance.
        left_finger_body_idx: Per-world body index for the left finger
            (typically ``robot.body_names.index("panda_leftfinger")``).
        right_finger_body_idx: Per-world body index for the right finger.
        n_arm_dofs: Number of arm DOFs to slice (7 for Franka).

    Returns:
        A :class:`NewtonOSCBuffers` instance, ready for repeated use.
    """
    view = robot.root_view
    model = NewtonManager._model
    if model is None:
        raise RuntimeError("NewtonManager._model is not initialized; build_buffers must run after the scene is set up.")

    device = robot.device
    n_joints = model.max_joints_per_articulation
    n_dofs = model.max_dofs_per_articulation

    # Newton's eval_jacobian / eval_mass_matrix kernels index the output
    # buffers by *global* articulation id (0..model.articulation_count-1),
    # not by the view's local slot. When other articulations exist in the
    # scene (e.g. kinematic_enabled rigid bodies registered as 1-body
    # articulations), allocating buffers of size ``view.count`` causes the
    # kernel to write to the rows that belong to those non-robot articulations
    # too — and on environments where the robot's global id is past the
    # buffer length, the writes silently fall outside the slice. Sizing both
    # buffers to ``model.articulation_count`` and gathering the robot rows
    # afterwards is the safe, correct way.
    n_arts = model.articulation_count

    j_full = wp.zeros((n_arts, n_joints * 6, n_dofs), dtype=float, device=device)
    h_full = wp.zeros((n_arts, n_dofs, n_dofs), dtype=float, device=device)
    joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
    body_I_s = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=device)

    # Resolve which joint-within-articulation feeds each finger body and
    # the fingertip body. ``joint_child[j] == body_idx`` means joint j has
    # child body_idx.
    joint_child = model.joint_child.numpy()
    articulation_start = model.articulation_start.numpy()
    art_start = int(articulation_start[0])
    art_end = int(articulation_start[1])
    left_joint = -1
    right_joint = -1
    fingertip_joint = -1
    for j in range(art_start, art_end):
        child = int(joint_child[j])
        if child == left_finger_body_idx:
            left_joint = j - art_start
        elif child == right_finger_body_idx:
            right_joint = j - art_start
        elif child == fingertip_body_idx:
            fingertip_joint = j - art_start
    if left_joint < 0 or right_joint < 0:
        raise RuntimeError(
            f"Could not resolve finger joints from body indices "
            f"(left_finger_body_idx={left_finger_body_idx}, "
            f"right_finger_body_idx={right_finger_body_idx}). "
            f"Newton joint_child: {joint_child[art_start:art_end].tolist()}"
        )

    # Cache joint_armature[:n_arm_dofs] as a torch tensor for the H-diagonal patch.
    armature_np = model.joint_armature.numpy()[:n_arm_dofs].copy()
    armature_torch = torch.as_tensor(armature_np, device=device, dtype=torch.float32)

    # Resolve which global articulation indices this view owns. ``articulation_ids``
    # is shape ``(world_count, count_per_world)`` (e.g. ``[[0], [3], [6], [9]]``);
    # we want a flat ``(num_envs,)`` long tensor for use with ``torch.index_select``.
    art_ids_np = view.articulation_ids
    if hasattr(art_ids_np, "numpy"):
        art_ids_np = art_ids_np.numpy()
    import numpy as np  # noqa: PLC0415

    art_ids_flat = np.asarray(art_ids_np).reshape(-1)
    articulation_indices_torch = torch.as_tensor(art_ids_flat, device=device, dtype=torch.long)

    return NewtonOSCBuffers(
        view=view,
        n_arm_dofs=n_arm_dofs,
        j_full=j_full,
        h_full=h_full,
        joint_S_s=joint_S_s,
        body_I_s=body_I_s,
        left_joint_in_art=left_joint,
        right_joint_in_art=right_joint,
        fingertip_joint_in_art=fingertip_joint,
        arm_armature_torch=armature_torch,
        articulation_indices_torch=articulation_indices_torch,
    )


def compute_arm_jacobian(buffers: NewtonOSCBuffers) -> torch.Tensor:
    """Compute the fingertip-midpoint arm Jacobian for the current Newton state.

    On Newton, ``eval_jacobian`` writes a per-joint spatial Jacobian
    expressed at each child body's **center of mass**. The fingertip body
    (``panda_fingertip_centered``) is a virtual fixed-joint frame with
    zero COM offset, so its Jacobian is identical to a body-origin
    Jacobian — which is the convention Factory's OSC and
    ``fingertip_midpoint_pos`` (read from ``body_pos_w`` = body link
    pose) both assume.

    Using the fingertip-body's Jacobian directly is the right thing to
    do here. The legacy "average the left and right finger Jacobians"
    path is preserved as a fallback for builds that don't expose a
    fingertip body — but on Factory the fingertip joint is always
    present, and using it removes a ~7.5% relative-norm error caused
    by the COM offset of the actual finger bodies (validated by the
    finite-difference probe at ``scripts/probe_newton_jacobian_fd.py``).

    Returns:
        ``(num_envs, 6, n_arm_dofs)`` torch tensor on the robot device,
        rows ordered ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]``.
    """
    state = NewtonManager._state_0
    buffers.view.eval_jacobian(state, J=buffers.j_full, joint_S_s=buffers.joint_S_s)

    # ``j_full`` has shape ``(model.articulation_count, n_joints*6, n_dofs)``;
    # gather only the rows owned by this view.
    j_full_t = wp.to_torch(buffers.j_full)
    j_robot = torch.index_select(j_full_t, 0, buffers.articulation_indices_torch)
    n = buffers.n_arm_dofs

    if buffers.fingertip_joint_in_art >= 0:
        ft = buffers.fingertip_joint_in_art * 6
        return j_robot[:, ft : ft + 6, :n]

    lj = buffers.left_joint_in_art * 6
    rj = buffers.right_joint_in_art * 6
    left = j_robot[:, lj : lj + 6, :n]
    right = j_robot[:, rj : rj + 6, :n]
    return (left + right) * 0.5


def compute_arm_mass_matrix(buffers: NewtonOSCBuffers) -> torch.Tensor:
    """Compute the arm-block joint-space mass matrix with armature on diag.

    ``ArticulationView.eval_mass_matrix`` returns ``H = J^T M J`` and does not
    fold ``joint_armature`` into the diagonal. We add ``diag(joint_armature)``
    here so the result matches the *effective* inertia the integrator uses,
    keeping :math:`\\Lambda = (J H^{-1} J^T)^{-1}` in agreement with the
    dynamics that torques will actually produce.

    Returns:
        ``(num_envs, n_arm_dofs, n_arm_dofs)`` torch tensor on the robot device.
    """
    state = NewtonManager._state_0
    # NOTE: Do NOT pass ``J=buffers.j_full`` here. ``eval_mass_matrix``
    # treats the supplied ``J`` as a *pre-computed* Jacobian and uses it
    # directly in ``H = J^T M J``. Our ``j_full`` workspace is only
    # populated when :func:`compute_arm_jacobian` is called explicitly
    # (which the IsaacLab Factory env does not — it builds its own
    # Jacobian via FK in ``_compute_fk_arm_jacobian``). With ``J=zeros``
    # the result is ``H = 0`` and the OSC ends up using only the
    # armature diagonal as inertia — a major silent OSC-quality bug.
    # ``J=None`` lets ``eval_mass_matrix`` compute the per-articulation
    # spatial Jacobian internally (allocating a temporary, but the cost
    # is negligible vs. the OSC calc itself).
    buffers.view.eval_mass_matrix(
        state,
        H=buffers.h_full,
        J=None,
        body_I_s=buffers.body_I_s,
        joint_S_s=buffers.joint_S_s,
    )
    # Same global-index convention as :func:`compute_arm_jacobian` — gather
    # the robot rows out of the full-articulation buffer.
    h_t = wp.to_torch(buffers.h_full)
    h_robot = torch.index_select(h_t, 0, buffers.articulation_indices_torch)
    n = buffers.n_arm_dofs
    h_arm = h_robot[:, :n, :n]
    return h_arm + torch.diag_embed(buffers.arm_armature_torch).to(h_arm.dtype).expand_as(h_arm)


def compute_arm_jacobian_and_mass_matrix(buffers: NewtonOSCBuffers) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: one call returning ``(J_arm, M_arm)`` for the OSC.

    ``J_arm`` shape ``(num_envs, 6, n_arm_dofs)``, ``M_arm`` shape
    ``(num_envs, n_arm_dofs, n_arm_dofs)``. Both on the robot device.
    """
    return compute_arm_jacobian(buffers), compute_arm_mass_matrix(buffers)


def clamp_to_effort_limits(
    dof_torque: torch.Tensor, limits: tuple[float, ...] = FRANKA_FR3_EFFORT_LIMITS
) -> torch.Tensor:
    """Per-joint elementwise torque clamp [N·m].

    PhysX's articulation drive enforces ``effort_limit_sim`` automatically.
    Newton does not on direct ``joint_f`` writes, so the Newton path applies
    the clamp explicitly here. The first ``len(limits)`` columns of
    ``dof_torque`` are clamped in-place against the symmetric limits;
    remaining columns (e.g. gripper DOFs) are left untouched.

    Args:
        dof_torque: ``(num_envs, num_dofs)`` torch tensor on any device.
        limits: Per-DOF symmetric clamp values, one per arm DOF. Defaults
            to :data:`FRANKA_FR3_EFFORT_LIMITS`.

    Returns:
        The same ``dof_torque`` tensor with arm columns clamped in-place.
    """
    n = len(limits)
    lim = torch.as_tensor(limits, device=dof_torque.device, dtype=dof_torque.dtype)
    dof_torque[..., :n] = torch.clamp(dof_torque[..., :n], min=-lim, max=lim)
    return dof_torque


def refresh_kinematics_only() -> None:
    """Refresh body poses from joint_q without stepping the integrator.

    Factory's ``set_pos_inverse_kinematics`` runs a 30-iteration DLS loop where
    each iteration writes ``joint_q`` and then needs the resulting ``body_q``
    (for fingertip pose) and Jacobian (for the next DLS solve). The legacy
    PhysX path uses ``sim.step`` for that — and on Newton, ``sim.step`` runs
    the *full* per-step pipeline (actuator model, integrator, contact
    resolution, scene update), even though IK only needs forward kinematics.
    On Factory's FR3+nut+bolt scene each such step is hundreds of milliseconds
    (kernels + Python-side actuator math) outside the captured CUDA graph,
    making the 30-iteration loop unusably slow.

    This helper does the kinematic part *only*: ``eval_fk`` on the main scene's
    Newton model, which refreshes ``state_0.body_q`` from the freshly-written
    ``state_0.joint_q``. It skips the integrator (no time advance), the
    actuator model (no joint_f write), the contact pipeline, and any scene
    bookkeeping. The follow-up Jacobian/mass-matrix recompute happens through
    the existing :func:`compute_arm_jacobian_and_mass_matrix` path which is
    already kinematic.

    Caller is responsible for:

    * Writing the updated ``joint_q`` to ``NewtonManager._state_0`` first
      (typically via ``robot.write_joint_position_to_sim_index``).
    * Calling :func:`compute_arm_jacobian_and_mass_matrix` afterwards if the
      Jacobian is needed for the next DLS iteration.
    * Refreshing any IsaacLab-side caches that depend on ``body_q`` (e.g.
      ``self.fingertip_midpoint_pos``).
    """
    model = NewtonManager._model
    state_0 = NewtonManager._state_0
    if model is None or state_0 is None:
        raise RuntimeError(
            "refresh_kinematics_only must run after the scene is set up"
            " (NewtonManager._model / _state_0 are not initialized)."
        )
    # ``mask=None`` runs FK over all articulations. For per-env masked FK,
    # we'd plumb in NewtonManager._fk_reset_mask, but that's an optimization
    # only meaningful when a subset of envs reset; Factory always resets all
    # envs together.
    eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)


# -----------------------------------------------------------------------------
# Robot-only IK solver (newton.ik.IKSolver wrapped for Factory)
# -----------------------------------------------------------------------------


class NewtonRobotIKSolver:
    """Robot-only Newton IK solver for Factory's reset path.

    Factory's :meth:`set_pos_inverse_kinematics` runs a 30-iteration DLS loop
    that calls ``step_sim_no_action`` per iteration. On the Newton backend
    that drives the full multi-env scene through the captured CUDA graph
    plus several outside-graph kernels per iteration, and DLS observably
    diverges (see ``scripts/probe_ik_convergence.py``).

    This class replaces that loop with one call to :class:`newton.ik.IKSolver`,
    which:

    * Owns its own robot-only Newton ``Model`` built at construction time
      from the Franka prim on the live USD stage. No nut, no bolt, no table,
      no contact pipeline.
    * Solves N IK problems (one per env) in parallel inside its own
      captured CUDA graph; each :meth:`solve` call is a single graph replay.
    * Uses Levenberg-Marquardt with analytic Jacobians to reach sub-mm
      convergence in a small number of iterations.

    PhysX is unaffected — this class is only constructed on the Newton path.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        ee_link_name: str = "panda_fingertip_centered",
        n_arm_dofs: int = 7,
        ik_iterations: int = 32,
    ) -> None:
        from newton import ModelBuilder, ik
        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

        from isaaclab.sim.utils.queries import find_first_matching_prim

        self.num_envs = num_envs
        self.device = device
        self.n_arm_dofs = n_arm_dofs
        self.ik_iterations = ik_iterations

        # ---- Build a robot-only Newton model from the live USD stage.
        # We reuse the same Franka prim spawned at /World/envs/env_0/Robot
        # — this guarantees identical kinematics (link offsets, joint axes,
        # masses, inertias) to the main-scene articulation, so a joint_q
        # produced here is valid when written back to the main scene.
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        env0_robot_prim = find_first_matching_prim("/World/envs/env_.*/Robot")
        if env0_robot_prim is None:
            raise RuntimeError(
                "NewtonRobotIKSolver: could not find a Franka prim under /World/envs/env_.*/Robot."
                " The IK solver must be constructed after the scene is populated."
            )
        self._robot_prim_path = env0_robot_prim.GetPath().pathString

        builder = ModelBuilder(up_axis="Z")
        builder.add_usd(
            stage,
            root_path=self._robot_prim_path,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        self.model = builder.finalize(device=device)
        self.state = self.model.state()

        # Resolve EE link index in the robot-only model. Body labels in
        # ``model.body_label`` (or ``body_key`` on older newton) include the
        # link name. We match by suffix to be USD-path-resolution-agnostic.
        body_keys = getattr(self.model, "body_label", None) or self.model.body_key
        self._ee_link_index = -1
        for i, key in enumerate(body_keys):
            # ``key`` might be a full prim path like ``/World/.../panda_fingertip_centered``
            if str(key).rsplit("/", 1)[-1] == ee_link_name:
                self._ee_link_index = i
                break
        if self._ee_link_index < 0:
            raise RuntimeError(
                f"NewtonRobotIKSolver: could not find EE link '{ee_link_name}' in robot-only model."
                f" body keys: {[str(k) for k in body_keys]}"
            )

        # ---- Set up IK objectives + solver
        # Initialize targets to body's current world transform (replaced per call).
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        body_q_np = self.state.body_q.numpy()
        ee_tf = body_q_np[self._ee_link_index]
        init_pos = wp.vec3(float(ee_tf[0]), float(ee_tf[1]), float(ee_tf[2]))
        init_quat = wp.vec4(float(ee_tf[3]), float(ee_tf[4]), float(ee_tf[5]), float(ee_tf[6]))

        # Boost position weight relative to rotation — Factory's reset
        # primarily cares about TCP position (orientation is roughly
        # specified by the hand-down convention but exact alignment is
        # not load-bearing). Without this asymmetry the LM solver finds
        # a local minimum that trades position for rotation accuracy and
        # ends up ~4 cm off in Z; with pos_weight=10 it converges sub-mm.
        self._pos_obj = ik.IKObjectivePosition(
            link_index=self._ee_link_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([init_pos] * num_envs, dtype=wp.vec3, device=device),
            weight=1000.0,
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self._ee_link_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([init_quat] * num_envs, dtype=wp.vec4, device=device),
            weight=100.0,
        )
        self._joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=num_envs,
            # Drop the joint-limit objective: empirically with it in the
            # mix the LM solver settles to a 4 cm-off local minimum on
            # Factory's targets. Without it, position+rotation reach
            # sub-mm. Joint limits are enforced naturally by seeding
            # near a valid pose plus modest step sizes.
            objectives=[self._pos_obj, self._rot_obj],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            lambda_initial=0.001,
        )

        # ``joint_q_ik`` shape ``(num_envs, joint_coord_count)`` — the
        # solver's working buffer. Allocated once.
        self._joint_q_ik = wp.zeros((num_envs, self.model.joint_coord_count), dtype=float, device=device)

        # The robot-only model's root body inherits whatever USD transform
        # sat on the source prim's parent chain (in our case env_0/Robot
        # picked up env_0's env_origin). Targets coming in from Factory are
        # already env-local (env_origin subtracted), so we cache the root
        # body's world translation here and add it to per-call targets.
        # body_q[0] is panda_link0's world transform; the first 3 floats
        # are the position. Quaternion offset is identity for IsaacLab
        # env grids (env transforms are translation-only), so we don't
        # rotate the orientation target — verified empirically by
        # comparing FK output to Factory's main-scene fingertip pose.
        body_q_np_initial = self.state.body_q.numpy()
        self._robot_base_offset = torch.as_tensor(body_q_np_initial[0, :3], device=device, dtype=torch.float32)

    def solve(self, initial_q: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Solve IK for ``num_envs`` per-env targets and return the resulting joint_q.

        Args:
            initial_q: ``(num_envs, n_arm_dofs)`` torch tensor of starting
                arm joint positions [rad]. Other DOFs (gripper) are seeded
                to the model's default.
            target_pos: ``(num_envs, 3)`` torch tensor of TCP target
                positions [m, world frame].
            target_quat: ``(num_envs, 4)`` torch tensor of TCP target
                rotations as ``(x, y, z, w)`` quaternions.

        Returns:
            ``(num_envs, n_arm_dofs)`` torch tensor of solved arm joint
            positions [rad]. The caller writes these into the main scene's
            robot via ``write_joint_position_to_sim_index``.
        """
        # Seed: copy initial_q[arm] into joint_q_ik[:, :n_arm_dofs], leave
        # gripper DOFs at the model's default (joint_q from finalize).
        joint_q_t = wp.to_torch(self._joint_q_ik)
        joint_q_t.zero_()
        # Tile model.joint_q across env rows as the gripper seed; overwrite
        # arm dofs with caller's initial_q.
        model_jq_t = wp.to_torch(self.model.joint_q)
        joint_q_t[:] = model_jq_t.unsqueeze(0).expand_as(joint_q_t)
        joint_q_t[:, : self.n_arm_dofs] = initial_q[:, : self.n_arm_dofs]

        # Push targets into the IK objectives. Caller provides target_pos
        # in env-local frame (matches Factory's ``fingertip_midpoint_pos``
        # convention); shift to the solver model's world frame by adding
        # the robot's base offset captured at construction.
        target_pos_solver = (target_pos + self._robot_base_offset).contiguous()
        target_pos_wp = wp.from_torch(target_pos_solver, dtype=wp.vec3)
        target_quat_wp = wp.from_torch(target_quat.contiguous(), dtype=wp.vec4)
        self._pos_obj.set_target_positions(target_pos_wp)
        self._rot_obj.set_target_rotations(target_quat_wp)

        # Run IK iterations on the robot-only model. The solver does its own
        # FK + Jacobian per iteration; we don't touch sim or main-scene state.
        self.solver.step(self._joint_q_ik, self._joint_q_ik, iterations=self.ik_iterations)

        return wp.to_torch(self._joint_q_ik)[:, : self.n_arm_dofs].clone()

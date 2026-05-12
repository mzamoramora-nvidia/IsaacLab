# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import warp as wp

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as torch_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg


def _newton_rigid_object_cfg(art_cfg: ArticulationCfg, kinematic: bool) -> RigidObjectCfg:
    """Adapt a joint-less Factory ArticulationCfg into a RigidObjectCfg for Newton.

    Mirrors the pattern Dexsuite (the canonical Newton-ready manipulation env)
    uses for its joint-less assets: wrap as :class:`RigidObjectCfg`, mark static
    targets ``kinematic_enabled=True`` (Dexsuite's table), leave manipulated
    objects dynamic (Dexsuite's cube).

    The resulting cfg has ``class_type = RigidObject``, so
    ``cfg.class_type(cfg)`` constructs the right wrapper at scene-setup time.
    PhysX is unaffected: it keeps using the original ArticulationCfg.
    """
    rigid_props = art_cfg.spawn.rigid_props
    if kinematic:
        # Replace the rigid_props with kinematic_enabled set; keep every other
        # field. RigidBodyPropertiesCfg is a configclass dataclass, so .replace
        # gives us a copy with one field overridden.
        rigid_props = (
            rigid_props.replace(kinematic_enabled=True)
            if rigid_props is not None
            else (sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        )
    spawn = art_cfg.spawn.replace(rigid_props=rigid_props)
    return RigidObjectCfg(
        prim_path=art_cfg.prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=art_cfg.init_state.pos,
            rot=art_cfg.init_state.rot,
            lin_vel=art_cfg.init_state.lin_vel,
            ang_vel=art_cfg.init_state.ang_vel,
        ),
    )


def _set_sim_gravity(cfg: FactoryEnvCfg, gravity: tuple[float, float, float]) -> None:
    """Set the live simulator gravity vector, dispatching by backend.

    Factory's reset path temporarily zeroes gravity to settle assets, then
    restores the cfg value. PhysX exposes ``physics_sim_view.set_gravity``
    via ``carb.Float3``; Newton has no equivalent on ``physics_sim_view``
    (which is a plain list of registered views), but exposes
    ``NewtonManager._model.set_gravity`` directly. This helper hides the
    difference at the two Factory call sites.
    """
    if _is_newton_backend(cfg):
        from isaaclab_newton.physics import NewtonManager
        from newton.solvers import SolverNotifyFlags

        if NewtonManager._model is not None:
            NewtonManager._model.set_gravity(gravity)
            # Newton's ``set_gravity`` only updates the host-side model; the
            # active solver (e.g. mjwarp) keeps the previous gravity inside
            # its ``opt.gravity`` device buffer until ``notify_model_changed``
            # re-uploads model properties. Without this, "disable gravity"
            # in the debug panel has no visible effect during stepping.
            NewtonManager.add_model_change(SolverNotifyFlags.MODEL_PROPERTIES)
        return
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


def _is_newton_backend(cfg: FactoryEnvCfg) -> bool:
    """Return True when the cfg has been resolved to the Newton physics backend.

    PresetCfg resolution swaps :attr:`FactoryEnvCfg.sim.physics` for either a
    :class:`~isaaclab_physx.physics.PhysxCfg` or :class:`~isaaclab_newton.physics.NewtonCfg`
    instance before the env is constructed. We dispatch on the runtime type of
    ``cfg.sim.physics`` to avoid hard-importing :mod:`isaaclab_newton` on the
    PhysX path.
    """
    physics = getattr(cfg.sim, "physics", None)
    if physics is None:
        return False
    return type(physics).__module__.startswith("isaaclab_newton")


class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        # Newton-side Factory uses Dexsuite's pattern for joint-less assets:
        # wrap them as :class:`RigidObjectCfg` (not :class:`ArticulationCfg`),
        # with ``kinematic_enabled=True`` for static targets (the bolt — the
        # nut threads onto it; it doesn't move). Dynamic objects (the nut)
        # stay non-kinematic. Then ``cfg.class_type(cfg)`` in _setup_scene
        # picks the right wrapper. PhysX uses the original ArticulationCfg.
        #
        # NOTE: setting ``kinematic=False`` for the fixed_asset would not
        # actually let the bolt fall — the Factory bolt USD authors a
        # ``PhysicsFixedJoint`` from world to root, which Newton's parser
        # honours regardless of our ``kinematic_enabled`` toggle. (Verified
        # via ``probe_newton_bolt_drop.py``: bolt's incoming joint type is
        # FIXED, body_inv_mass>0 but gravity has no effect since the joint
        # has no DOFs.) Same convention as panda-nut-bolt-osc, which
        # explicitly sets ``floating=False`` for the bolt. So we keep the
        # bolt kinematic, and align the cuboid table top to the bolt's
        # default z=0.05 (in ``_setup_scene``) so it sits flush.
        if _is_newton_backend(cfg):
            kinematic_for = {"fixed_asset", "small_gear_cfg", "large_gear_cfg"}
            for asset_attr in ("fixed_asset", "held_asset", "small_gear_cfg", "large_gear_cfg"):
                asset_cfg = getattr(self.cfg_task, asset_attr, None)
                if isinstance(asset_cfg, ArticulationCfg):
                    setattr(
                        self.cfg_task,
                        asset_attr,
                        _newton_rigid_object_cfg(asset_cfg, kinematic=asset_attr in kinematic_for),
                    )

            # Zero the Z component of the bolt's reset randomisation. The
            # default ``fixed_asset_init_pos_noise = [0.05, 0.05, 0.05]``
            # pushes the bolt's z uniformly into ``[0.0, 0.10]`` — and
            # because the bolt is parsed as a fixed (welded) body via its
            # USD ``PhysicsFixedJoint`` (verified in
            # ``probe_newton_bolt_drop.py``: bolt joint type = FIXED), it
            # *stays* wherever the reset places it. Without the z-zero
            # the bolt visibly intersects the table on half the random
            # samples and floats above it on the other half. XY
            # randomisation is preserved. Same convention as the
            # panda-nut-bolt OSC reference (no Z-noise on the bolt).
            noise = getattr(self.cfg_task, "fixed_asset_init_pos_noise", None)
            if noise is not None and len(noise) >= 3:
                self.cfg_task.fixed_asset_init_pos_noise = list(noise[:2]) + [0.0]

            # Drop the bolt's default ``init_state.pos.z`` from the
            # Factory cfg's 0.05 (chosen to match the original
            # ``table_instanceable.usd`` top height in PhysX) to 0.0,
            # which is the top of our Newton cuboid table at
            # ``pos=(0.55, 0.0, -0.02), size_z=0.04``. Keeping the
            # cuboid at z=-0.02 (its original position) avoids lifting
            # it into the panda base; remapping the bolt to z=0.0
            # keeps it flush on that table top instead of floating
            # 5 cm above. The held_asset (nut) goes to the gripper
            # TCP via the reset's grasp-settle phase, so its init_state
            # pose is not used at runtime — leave it untouched.
            bolt = getattr(self.cfg_task, "fixed_asset", None)
            if bolt is not None and hasattr(bolt, "init_state") and hasattr(bolt.init_state, "pos"):
                cur_pos = bolt.init_state.pos
                new_init = bolt.init_state.replace(pos=(float(cur_pos[0]), float(cur_pos[1]), 0.0))
                self.cfg_task.fixed_asset = bolt.replace(init_state=new_init)
            # Bump arm actuator armature *before* ``super().__init__`` so the
            # ImplicitActuatorCfg values get baked into ``model.joint_armature``
            # at finalize time (which otherwise resets the builder's per-DOF
            # armature back to the cfg default of 0.0). Numbers from the
            # ``mzamora/newton-dev`` Factory branch — they stabilise the
            # OSC's ``Lambda = (J H^-1 J^T)^-1`` and damp wrist chatter.
            arm_actuators = cfg.robot.actuators
            arm_actuators["panda_arm1"].armature = 0.3
            arm_actuators["panda_arm2"].armature = 0.11
            arm_actuators["panda_hand"].armature = 0.15

            # Arm joint damping. With ``factory_newton_setup`` flipping
            # the arm into POSITION mode + ke=0, mjwarp applies a per-joint
            # ``-kd * qd`` damping term.
            #
            # We previously used ``kd=10`` matching
            # ``example_robot_panda_nut_bolt_osc.py:1249``, but that's a
            # standalone-robot setting; in the Factory env (held nut +
            # stiff grasp contacts) it acts as a 4× brake on OSC tracking
            # — a +x action that PhysX tracks to +48 mm in 20 ticks only
            # reaches +12 mm on Newton with kd=10. Empirically (probe
            # tracking sweep with the correct ``JOINT_DOF_PROPERTIES``
            # notify flag), ``kd=0`` restores tracking to +46 mm AND
            # tightens R0 hold drift from ~7 mm to ~1 mm — the OSC's own
            # ``Kd = 2√Kp`` plus the arm's armature already provide
            # sufficient damping.
            arm_actuators["panda_arm1"].damping = 0.0
            arm_actuators["panda_arm2"].damping = 0.0

            # Finger PD gains. Factory's PhysX defaults are 7500 / 173
            # (very stiff close, designed to clamp the nut hard before
            # the policy takes over). panda-nut-bolt-osc uses the much
            # softer 100 / 10 — gentler close, equilibrium grip force
            # determined by ``shape_material_kh`` * penetration rather
            # than PD * spring. With our hydroelastic stack now wired
            # in, the softer gains are appropriate; the stiff gains
            # were over-driving the SDFs into the nut surface.
            arm_actuators["panda_hand"].stiffness = 1000.0
            arm_actuators["panda_hand"].damping = 10.0

            # Disable IsaacLab's per-prototype convex-hull mesh
            # approximation. The cloner runs ``approximate_meshes(
            # "convex_hull", ...)`` on every spawned prototype right
            # after USD parsing, replacing each finger / nut / bolt
            # collision mesh with its convex hull. For threaded
            # geometry (M16 hex flats + screw threads) the convex hull
            # is essentially a smooth shell — the SDFs we build on top
            # in ``factory_newton_setup._build_collision_sdfs`` are
            # built on that smoothed-out mesh, so:
            #   * Show Collision in the viewer renders nothing useful.
            #   * Hydroelastic contacts can't engage hex faces or
            #     thread features — gripper just slips on the convex
            #     proxy.
            # Match panda-nut-bolt's setup by skipping the convex-hull
            # pass. We monkey-patch the cloner export so InteractiveScene
            # (constructed inside ``super().__init__``) picks up the
            # patched callable. Keyed off ``_is_newton_backend`` so
            # PhysX runs are unaffected.
            from isaaclab_newton.cloner import newton_replicate as _nr

            _orig_replicate = _nr.newton_physics_replicate

            def _factory_no_convex_hull_replicate(*args, **kwargs):
                kwargs.setdefault("simplify_meshes", False)
                return _orig_replicate(*args, **kwargs)

            _nr.newton_physics_replicate = _factory_no_convex_hull_replicate
            import isaaclab_newton.cloner as _isaaclab_newton_cloner

            _isaaclab_newton_cloner.newton_physics_replicate = _factory_no_convex_hull_replicate

            # Scale the hydroelastic contact buffer with the number of
            # worlds. Newton's ``CollisionPipeline`` allocates one
            # ``rigid_contact_max``-sized buffer covering every world;
            # panda-nut-bolt's 1-env setup uses 2048 with njmax=2000
            # (ratio ≈ njmax × world_count). Below that, the buffer
            # overflows during gripper close — visible as
            # ``rigid contact output overflow: N > rigid_contact_max``
            # warnings plus visible interpenetration. Sizing it
            # dynamically here means the default works as the user
            # bumps ``cfg.scene.num_envs`` for training without having
            # to re-tune the buffer by hand.
            if (
                getattr(cfg.sim.physics, "collision_cfg", None) is not None
                and getattr(cfg.sim.physics, "solver_cfg", None) is not None
                and hasattr(cfg.sim.physics.solver_cfg, "njmax")
            ):
                njmax = int(cfg.sim.physics.solver_cfg.njmax)
                num_worlds = int(cfg.scene.num_envs)
                cfg.sim.physics.collision_cfg.rigid_contact_max = njmax * num_worlds

        super().__init__(cfg, render_mode, **kwargs)

        # Newton-only post-load asset setup (POSITION-mode override on the
        # arm DOFs, hydroelastic flag on robot collision shapes, contact-gap
        # tuning, etc.). Runs *after* ``super().__init__`` so the Newton
        # model is finalized; PhysX never imports this module.
        if _is_newton_backend(self.cfg):
            from . import factory_newton_setup

            factory_newton_setup.apply(self)

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # Newton-only kernel warm-up. Factory's reset path includes a 30-step
        # IK loop in ``set_pos_inverse_kinematics`` whose per-iteration call
        # to ``step_sim_no_action`` (write_data_to_sim → sim.step →
        # scene.update → _compute_intermediate_values) bypasses the captured
        # CUDA graph. Each first-time invocation JIT-compiles a Newton kernel
        # (``eval_fk``, the actuator-model kernels, ``eval_jacobian`` and
        # ``eval_mass_matrix`` via ``factory_control_newton`` on the OSC
        # tail) — these take 10-30 s each on Factory's FR3+nut+bolt scene.
        # Pre-pay that cost here on a couple of dummy steps so the 30 IK
        # iterations all hit warm caches. PhysX is unaffected.
        if _is_newton_backend(self.cfg):
            for _ in range(2):
                self.step_sim_no_action()

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Newton OSC buffers (None on PhysX so the per-step dispatch is a single
        # cheap is-not-None check rather than calling into the resolver each step).
        self._newton_osc_buffers = None
        if _is_newton_backend(self.cfg):
            from . import factory_control_newton

            self._newton_osc_buffers = factory_control_newton.build_buffers(
                self._robot,
                left_finger_body_idx=self.left_finger_body_idx,
                right_finger_body_idx=self.right_finger_body_idx,
                fingertip_body_idx=self.fingertip_body_idx,
                n_arm_dofs=7,
            )

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # Spawn the Seattle-lab table. On PhysX we drop in the USD as a plain
        # static prim — the existing path. On Newton we wrap it as a
        # kinematic-enabled :class:`RigidObject` so that:
        #   * it shows up as a body in ``model.body_label`` (the Newton viewer
        #     only renders shapes that belong to a body, so a world-static
        #     UsdFileCfg-spawned table appears nowhere and the bolt looks like
        #     it is floating);
        #   * the bolt / nut / robot collide against a single articulated
        #     ground-truth body, matching the Dexsuite pattern.
        # PhysX behaviour is unchanged because ``_setup_scene`` is the only
        # caller and we keep the ``cfg.func`` branch verbatim there.
        if _is_newton_backend(self.cfg):
            # The instanceable Seattle-lab-table USD has no ``UsdPhysics.RigidBodyAPI``
            # on its root, so wrapping it in a :class:`RigidObjectCfg` errors out at
            # initialize-time. Match the Dexsuite pattern instead and spawn a thin
            # kinematic-enabled cuboid the size of the table top — enough for the
            # Newton viewer to render a visible body and for the bolt / nut to sit
            # on a real collider. Top surface aligned to **z=0.0** to match the
            # original USD table-top height; we also remap the bolt's
            # ``init_state.pos.z`` to 0.0 in :meth:`__init__` so the bolt sits
            # flush on the table without a 5 cm "floor under the floor" lifting
            # the table into the panda base. 4 cm thick → cuboid centre at
            # z = 0.0 - 0.02 = -0.02.
            # ``MeshCuboidCfg`` (not ``CuboidCfg``) — analytical boxes
            # use Newton's exact box SDF for narrow-phase but produce no
            # voxel-grid SDF for the viewer's "Show Collision" overlay,
            # so the table was invisible under that toggle. A mesh box
            # gets picked up by ``_build_collision_sdfs`` and baked into
            # a voxel SDF, matching the panda / nut / bolt path.
            table_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Table",
                spawn=sim_utils.MeshCuboidCfg(
                    size=(1.2, 0.6, 0.04),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, -0.02), rot=(1.0, 0.0, 0.0, 0.0)),
            )
            self._table = RigidObject(table_cfg)
        else:
            # PhysX path: keep the original Seattle-lab-table USD spawn.
            cfg = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            )
            cfg.func(
                "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.0, 0.0, 0.70711, 0.70711)
            )

        self._robot = Articulation(self.cfg.robot)
        # Joint-less assets dispatch via cfg.class_type so PhysX gets
        # Articulation and Newton gets RigidObject (after the Newton-only
        # cfg conversion in __init__). Same call site, both backends.
        self._fixed_asset = self.cfg_task.fixed_asset.class_type(self.cfg_task.fixed_asset)
        self._held_asset = self.cfg_task.held_asset.class_type(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = self.cfg_task.small_gear_cfg.class_type(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = self.cfg_task.large_gear_cfg.class_type(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        # Joint-less assets register on rigid_objects (Newton: RigidObject,
        # PhysX: Articulation also satisfies the rigid-object dict contract;
        # mirrors what shadow_hand_vision does). Either backend can read its
        # own object back via the dict on reset/randomization paths.
        is_newton = _is_newton_backend(self.cfg)
        asset_registry = self.scene.rigid_objects if is_newton else self.scene.articulations
        asset_registry["fixed_asset"] = self._fixed_asset
        asset_registry["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            asset_registry["small_gear"] = self._small_gear_asset
            asset_registry["large_gear"] = self._large_gear_asset

        # Newton-only: register the kinematic table RigidObject so the scene
        # can clone/bind it and the data layer can resolve its body. PhysX
        # spawned the table as a plain static prim and doesn't need this.
        if is_newton:
            self.scene.rigid_objects["table"] = self._table

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Newton-only: register the MODEL_INIT callback that sets per-DOF /
        # per-body MuJoCo gravity-compensation custom attributes and the
        # Franka armature. This MUST happen *before* model finalization,
        # which is why we register here (during ``_setup_scene``) rather
        # than calling a function in ``__init__`` after ``super().__init__``.
        if is_newton:
            from . import factory_newton_setup

            factory_newton_setup.register_model_init_callback()

    def _compute_fingertip_velocity_from_newton_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read fingertip linear + angular velocity directly from mjwarp state.

        The IsaacLab Newton articulation data adapter returns 0 for
        ``body_lin_vel_w`` / ``body_link_lin_vel_w`` on the
        ``panda_fingertip_centered`` body (a zero-mass virtual link).
        That kills the OSC's task-space Kd*e_dot damping term. The raw
        mjwarp ``state.body_qd`` has the correct velocity for that body,
        so we read it directly and apply the COM->link transport:

            v_link = v_com + omega x (p_link - p_com)

        Mirrors the kernel at ``newton/examples/robot/osc.py:96-100``.

        Returns:
            Tuple ``(linvel, angvel)``, each ``(num_envs, 3)`` torch
            tensors on the robot device, in world frame at the
            fingertip body origin.
        """
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager._state_0
        model = NewtonManager._model
        if state is None or model is None:
            zero = torch.zeros((self.num_envs, 3), device=self.device)
            return zero, zero

        # Cache the per-env fingertip body global indices on first call.
        if not hasattr(self, "_fingertip_body_global_idx"):
            labels = list(model.body_label)
            idxs = []
            for env_idx in range(self.num_envs):
                env_token = f"/env_{env_idx}/"
                for i, lab in enumerate(labels):
                    if lab.endswith("/panda_fingertip_centered") and env_token in lab:
                        idxs.append(i)
                        break
            if len(idxs) != self.num_envs:
                raise RuntimeError(
                    f"Could not resolve panda_fingertip_centered for all envs (found {len(idxs)} of {self.num_envs})."
                )
            self._fingertip_body_global_idx = torch.tensor(idxs, dtype=torch.long, device=self.device)

        # body_qd layout: [v_com_x, v_com_y, v_com_z, w_x, w_y, w_z] in world frame.
        body_qd_t = wp.to_torch(state.body_qd)  # (n_bodies, 6)
        ft_idx = self._fingertip_body_global_idx
        v_com = body_qd_t[ft_idx, 0:3]
        omega = body_qd_t[ft_idx, 3:6]

        # Transport from body COM to body link origin: v_link = v_com + omega x (p_link - p_com).
        # body_q[fingertip] is the link world transform; body_com is local COM offset.
        body_q_t = wp.to_torch(state.body_q)  # (n_bodies, 7) = (px, py, pz, qx, qy, qz, qw)
        body_com_t = wp.to_torch(model.body_com)  # (n_bodies, 3) local-frame COM offset
        link_pos = body_q_t[ft_idx, 0:3]
        link_quat_xyzw = body_q_t[ft_idx, 3:7]
        com_local = body_com_t[ft_idx]
        # Rotate com_local into world frame using link_quat.
        com_world_offset = torch_utils.quat_apply(link_quat_xyzw, com_local)
        r_com_w = link_pos + com_world_offset
        v_link = v_com + torch.cross(omega, link_pos - r_com_w, dim=-1)
        return v_link, omega

    def _compute_fk_arm_jacobian(self) -> torch.Tensor:
        """Build the fingertip-midpoint arm Jacobian directly from FK.

        For a chain of revolute joints with axes ``z_i`` (in the joint
        frame) and joint origins ``p_i``, the geometric Jacobian column
        for joint i, evaluated at point ``p_ee``, is:

            J_lin[i] = z_i_world × (p_ee_world - p_i_world)
            J_ang[i] = z_i_world

        This matches the ``J^T`` formulation Factory's OSC uses and
        avoids the body-origin-vs-COM frame mismatch in Newton's
        ``eval_jacobian`` (which is written at each child body's COM).

        Returns:
            ``(num_envs, 6, n_arm_dofs)`` torch tensor, rows ``[v, ω]``.
        """
        body_pos_w = self._robot.data.body_pos_w.torch
        body_quat_w = self._robot.data.body_quat_w.torch
        env_origins = self.scene.env_origins
        n_arm = 7

        if not hasattr(self, "_arm_joint_body_indices"):
            names = self._robot.body_names
            self._arm_joint_body_indices = torch.tensor(
                [names.index(f"panda_link{i}") for i in range(1, n_arm + 1)],
                dtype=torch.long,
                device=self.device,
            )
            self._fk_local_z = torch.zeros(self.num_envs * n_arm, 3, device=self.device)
            self._fk_local_z[:, 2] = 1.0

        joint_pos_w = body_pos_w[:, self._arm_joint_body_indices] - env_origins.unsqueeze(1)
        joint_quat = body_quat_w[:, self._arm_joint_body_indices]
        joint_axes = torch_utils.quat_apply(joint_quat.reshape(-1, 4), self._fk_local_z).reshape(
            self.num_envs, n_arm, 3
        )

        p_ee = body_pos_w[:, self.fingertip_body_idx] - env_origins
        r = p_ee.unsqueeze(1) - joint_pos_w  # (N, 7, 3)
        j_lin = torch.cross(joint_axes, r, dim=-1)  # (N, 7, 3)

        j_arm = torch.cat([j_lin.permute(0, 2, 1), joint_axes.permute(0, 2, 1)], dim=1)
        # ``torch.nan_to_num`` here is defensive — Newton occasionally returns
        # NaN body_q during contact blow-ups; bringing those into the
        # Jacobian explodes the OSC.
        return torch.nan_to_num(j_arm, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w.torch - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w.torch

        self.held_pos = self._held_asset.data.root_pos_w.torch - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w.torch

        self.fingertip_midpoint_pos = (
            self._robot.data.body_pos_w.torch[:, self.fingertip_body_idx] - self.scene.env_origins
        )
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w.torch[:, self.fingertip_body_idx]
        if self._newton_osc_buffers is not None:
            # The IsaacLab Newton data adapter returns 0 for both
            # ``body_lin_vel_w`` (COM frame) and ``body_link_lin_vel_w``
            # (transport-corrected) on ``panda_fingertip_centered`` -- a
            # zero-mass virtual body. mjwarp ``state.body_qd`` does have
            # the real velocity for that body, so we read it directly and
            # apply the transport ourselves (matches the working pattern
            # in ``newton/examples/robot/osc.py:96-100``).
            self.fingertip_midpoint_linvel, self.fingertip_midpoint_angvel = (
                self._compute_fingertip_velocity_from_newton_state()
            )
        else:
            self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w.torch[:, self.fingertip_body_idx]
            self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w.torch[:, self.fingertip_body_idx]

        if self._newton_osc_buffers is not None:
            # Newton path: build the arm Jacobian ourselves from FK rather
            # than reading Newton's ``eval_jacobian`` output. ``eval_jacobian``
            # writes per-joint Jacobians at each child body's COM in world
            # frame; the OSC and ``fingertip_midpoint_pos`` both want a
            # body-origin Jacobian. The cross-product formulation
            # ``J_lin = z_axis × (p_finger - p_joint)`` gives exactly that
            # for a chain of revolute joints. Mass matrix still comes from
            # ``factory_control_newton`` (and absorbs the joint_armature
            # diagonal patch).
            from . import factory_control_newton

            self.fingertip_midpoint_jacobian = self._compute_fk_arm_jacobian()
            self.arm_mass_matrix = factory_control_newton.compute_arm_mass_matrix(self._newton_osc_buffers)
            self.left_finger_jacobian = self.fingertip_midpoint_jacobian
            self.right_finger_jacobian = self.fingertip_midpoint_jacobian
        else:
            jacobians = wp.to_torch(self._robot.root_view.get_jacobians())

            self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
            self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
            self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
            self.arm_mass_matrix = wp.to_torch(self._robot.root_view.get_generalized_mass_matrices())[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.torch.clone()
        self.joint_vel = self._robot.data.joint_vel.torch.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 3]).unsqueeze(-1)  # W component is at index 3 in XYZW format
        rot_diff_aa = torch_utils.axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        # Defensive NaN/Inf guard on Newton: a contact blow-up can corrupt
        # ``joint_q`` / ``body_q`` mid-rollout, propagating to ``ee_*_fd``
        # and then into the policy's log-std, which crashes ``rl_games``
        # with ``normal expects all elements of std >= 0``. Map any
        # non-finite entry to zero so the bad sample is benign and the
        # next reset clears the env. PhysX runs are unaffected.
        if self._newton_osc_buffers is not None:
            obs_tensors = torch.nan_to_num(obs_tensors, nan=0.0, posinf=0.0, neginf=0.0)
            state_tensors = torch.nan_to_num(state_tensors, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.linalg.norm(rot_actions, ord=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.euler_xyz_from_quat(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.linalg.norm(rot_actions, ord=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.euler_xyz_from_quat(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        # Newton's articulation drive does not enforce per-joint effort_limit_sim
        # on direct joint_f writes (PhysX does). Without this clamp, OSC torque
        # saturation produces 100 N·m on the wrist (factory_control.compute_dof_torque's
        # global ±100 ceiling) instead of the FR3 datasheet 12 N·m. Same patch
        # the panda-osc reference applies at osc.py:556.
        if self._newton_osc_buffers is not None:
            from . import factory_control_newton

            factory_control_newton.clamp_to_effort_limits(self.joint_torque)
            # Defensive: if the OSC produced NaN/Inf (singular mass matrix,
            # degenerate Jacobian, blow-up from a previous step) clamp the
            # offending entries to zero before they reach the actuator.
            # Without this guard the bad torque feeds back into joint_q,
            # then into observations, and the policy log_std silently goes
            # to NaN, which crashes ``rl_games`` mid-rollout with the
            # ``normal expects all elements of std >= 0`` assertion.
            self.joint_torque = torch.nan_to_num(self.joint_torque, nan=0.0, posinf=0.0, neginf=0.0)

        self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target_index(target=self.joint_torque)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.euler_xyz_from_quat(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras.setdefault("log", {})["Metrics/success_rate"] = (
                torch.count_nonzero(curr_successes) / self.num_envs
            ).item()

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx], _ = torch_utils.combine_frame_transforms(
                held_base_pos,
                held_base_quat,
                keypoint_offset.repeat(self.num_envs, 1),
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            )
            keypoints_fixed[:, idx], _ = torch_utils.combine_frame_transforms(
                target_held_base_pos,
                target_held_base_quat,
                keypoint_offset.repeat(self.num_envs, 1),
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            )
        keypoint_dist = torch.linalg.norm(keypoints_held - keypoints_fixed, ord=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.linalg.norm(self.actions, ord=2)
        action_grad_penalty = torch.linalg.norm(self.actions - self.prev_actions, ord=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_pose = self._held_asset.data.default_root_pose.torch.clone()[env_ids]
        held_vel = self._held_asset.data.default_root_vel.torch.clone()[env_ids]
        held_pose[:, 0:3] += self.scene.env_origins[env_ids]
        held_vel[:] = 0.0
        self._held_asset.write_root_link_pose_to_sim_index(root_pose=held_pose, env_ids=env_ids)
        self._held_asset.write_root_link_velocity_to_sim_index(root_velocity=held_vel, env_ids=env_ids)
        self._held_asset.reset()

        fixed_pose = self._fixed_asset.data.default_root_pose.torch.clone()[env_ids]
        fixed_vel = self._fixed_asset.data.default_root_vel.torch.clone()[env_ids]
        fixed_pose[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_vel[:] = 0.0
        self._fixed_asset.write_root_link_pose_to_sim_index(root_pose=fixed_pose, env_ids=env_ids)
        self._fixed_asset.write_root_link_velocity_to_sim_index(root_velocity=fixed_vel, env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_position_to_sim_index(position=self.joint_pos)
            self._robot.write_joint_velocity_to_sim_index(velocity=self.joint_vel)
            self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos)

            # Simulate and update tensors. ``step_sim_no_action`` itself
            # dispatches per backend: PhysX runs the full physics step;
            # Newton skips the integrator and only refreshes FK + Jacobian
            # (see :meth:`step_sim_no_action`).
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = factory_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target_index(target=joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.

        Both backends now run the full ``sim.step``. An earlier optimization
        attempt skipped the integrator on Newton in favor of a direct
        ``eval_fk`` refresh, but that bypassed Newton's captured-CUDA-graph
        scatter/gather between staging arrays and ``state_0``, leaving the
        IsaacLab data-layer bindings (``body_pos_w`` etc.) reading stale
        values. DLS IK then diverged because the Jacobian and the cached
        fingertip pose disagreed about the current state. The captured
        graph is fast (~1.5 ms after warm-up), so paying it per IK
        iteration is acceptable.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        self._full_reset(env_ids)

    def _full_reset(self, env_ids):
        """Original PhysX reset path: IK + asset randomization + grasp settle."""

        # Disable gravity (PhysX/Newton-portable).
        _set_sim_gravity(self.cfg, (0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_pose = self._fixed_asset.data.default_root_pose.torch.clone()[env_ids]
        fixed_vel = self._fixed_asset.data.default_root_vel.torch.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_pose[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_pose[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_vel[:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_link_pose_to_sim_index(root_pose=fixed_pose, env_ids=env_ids)
        self._fixed_asset.write_root_link_velocity_to_sim_index(root_velocity=fixed_vel, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        fixed_tip_pos, _ = torch_utils.combine_frame_transforms(
            self.fixed_pos,
            self.fixed_quat,
            fixed_tip_pos_local,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.linalg.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_pose = self._small_gear_asset.data.default_root_pose.torch.clone()[env_ids]
            small_gear_vel = self._small_gear_asset.data.default_root_vel.torch.clone()[env_ids]
            small_gear_pose[:, 0:7] = fixed_pose[:, 0:7]
            small_gear_vel[:] = 0.0  # vel
            self._small_gear_asset.write_root_link_pose_to_sim_index(root_pose=small_gear_pose, env_ids=env_ids)
            self._small_gear_asset.write_root_link_velocity_to_sim_index(root_velocity=small_gear_vel, env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_pose = self._large_gear_asset.data.default_root_pose.torch.clone()[env_ids]
            large_gear_vel = self._large_gear_asset.data.default_root_vel.torch.clone()[env_ids]
            large_gear_pose[:, 0:7] = fixed_pose[:, 0:7]
            large_gear_vel[:] = 0.0  # vel
            self._large_gear_asset.write_root_link_pose_to_sim_index(root_pose=large_gear_pose, env_ids=env_ids)
            self._large_gear_asset.write_root_link_velocity_to_sim_index(root_velocity=large_gear_vel, env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_pos, fingertip_flipped_quat = torch_utils.combine_frame_transforms(
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            torch.zeros((self.num_envs, 3), device=self.device),
            flip_z_quat,
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        # Compute inverse of relative transform: inv(quat, pos) = (quat_conjugate, -quat_rotate_inverse(quat, pos))
        asset_in_hand_quat = torch_utils.quat_inv(held_asset_relative_quat)
        asset_in_hand_pos = torch_utils.quat_apply_inverse(held_asset_relative_quat, -held_asset_relative_pos)

        translated_held_asset_pos, translated_held_asset_quat = torch_utils.combine_frame_transforms(
            fingertip_flipped_pos, fingertip_flipped_quat, asset_in_hand_pos, asset_in_hand_quat
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_pos, translated_held_asset_quat = torch_utils.combine_frame_transforms(
            translated_held_asset_pos,
            translated_held_asset_quat,
            held_asset_pos_noise,
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        )

        held_pose = self._held_asset.data.default_root_pose.torch.clone()
        held_vel = self._held_asset.data.default_root_vel.torch.clone()
        held_pose[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_pose[:, 3:7] = translated_held_asset_quat
        held_vel[:] = 0.0
        self._held_asset.write_root_link_pose_to_sim_index(root_pose=held_pose)
        self._held_asset.write_root_link_velocity_to_sim_index(root_velocity=held_vel)
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        # Restore gravity (PhysX/Newton-portable).
        # On Newton, keep global gravity at zero post-reset to replicate
        # PhysX's per-body ``disable_gravity=True`` regime. Every dynamic
        # body in our Factory scene has ``disable_gravity=True`` under
        # PhysX (robot in :class:`FactoryEnvCfg.robot.spawn.rigid_props`,
        # held nut in :class:`factory_tasks_cfg.FactoryTask.held_asset`).
        # Newton's IsaacLab adapter doesn't honour per-body
        # ``disable_gravity``, so leaving gravity on causes the held
        # nut to fall at ~g, pull the gripper down via grip friction, and
        # produce ~10 mm TCP drift during R0 hold (probe_zero_gravity_drift
        # showed drift 9.28 mm with gravity on, 0.015 mm with gravity off).
        # With gravity off, gravcomp on the robot ``-m·g·factor = 0`` —
        # the OSC has no gravity load to compensate, matching PhysX exactly.
        if _is_newton_backend(self.cfg):
            _set_sim_gravity(self.cfg, (0.0, 0.0, 0.0))
        else:
            _set_sim_gravity(self.cfg, tuple(self.cfg.sim.gravity))

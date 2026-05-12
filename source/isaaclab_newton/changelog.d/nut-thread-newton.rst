Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.HydroelasticSDFCfg.moment_matching`
  to surface Newton's PhysX-patch-friction analog on the IsaacLab
  config side.

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.RigidObjectData` crashing with
  ``IndexError: tuple index out of range`` for kinematic-enabled
  single-body fixed-base rigid objects (e.g. Factory's bolt). The
  ``is_fixed_base`` branch indexed ``[:, 0, 0]`` assuming a 3D
  ``(count, links, 1)`` layout, but Newton returns a 2D
  ``(count, links)`` array when the view contains a single body.
  Dispatch on actual ``ndim`` instead. Also fixes the
  ``derive_body_acceleration_from_body_com_velocities`` kernel
  rejecting a 1D ``body_com_vel`` by allocating the
  ``_sim_bind_body_com_vel_w`` fallback as 2D.

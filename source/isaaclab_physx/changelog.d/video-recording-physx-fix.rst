Added
^^^^^

* Added :meth:`~isaaclab_physx.physics.PhysxManager.pre_render` to sync physics
  state to Fabric before rendering. Ensures Kit cameras observe up-to-date
  transforms when capturing frames between simulation steps (e.g. for video
  recording during policy rollouts).

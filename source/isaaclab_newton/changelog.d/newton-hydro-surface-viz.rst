Added
^^^^^

* Added ``Show Hydro Surface`` toggle to the Newton viewer's
  Visualization panel and wired
  :meth:`~newton.viewer.Viewer.log_hydro_contact_surface` into the
  per-frame log block in
  :class:`~isaaclab_visualizers.newton.NewtonVisualizer` and
  :class:`~isaaclab_newton.video_recording.NewtonGlPerspectiveVideo`,
  so the hydroelastic SDF contact surface (the "isosurface") can be
  overlaid in both interactive viewer sessions and headless GL video
  recordings.

* Added :meth:`~isaaclab_newton.scene_data_providers.NewtonSceneDataProvider.get_collision_pipeline`
  for the visualizer + GL video recorder to fetch
  ``hydroelastic_sdf.get_contact_surface()``.

* Replaced :meth:`~isaaclab_newton.scene_data_providers.NewtonSceneDataProvider.get_contacts`
  hardcoded ``return None`` stub with a working implementation that
  also runs ``solver.update_contacts(contacts, state)`` so the
  per-frame ``log_contacts`` call reflects the current step instead
  of the captured-graph snapshot.

Changed
^^^^^^^

* Dropped the duplicate ``_paused_rendering`` cache in
  :class:`~isaaclab_visualizers.newton.NewtonViewerGL`; the
  ``Pause Rendering`` button label now reads directly from
  ``ViewerGL._paused`` so it tracks Space-key / Newton's own Pause
  checkbox without drifting out of sync.

"""Auto-install the LeRobot PyAV decode shim in spawn-started DataLoader workers.

openpi's ``TorchDataLoader`` uses ``multiprocessing.get_context("spawn")`` whenever
``num_workers > 0``, so a monkeypatch applied in the parent never reaches the workers. Python
imports ``sitecustomize`` at interpreter startup in *every* process, including spawned ones, and
``PYTHONPATH`` is inherited -- so putting this directory on ``PYTHONPATH`` makes the shim stick
everywhere.

Put this directory FIRST only if nothing else provides a sitecustomize; it is a no-op when
lerobot is absent.
"""

try:
    import lerobot_video_shim

    lerobot_video_shim.install()
except Exception:  # noqa: BLE001 - never break interpreter startup
    pass

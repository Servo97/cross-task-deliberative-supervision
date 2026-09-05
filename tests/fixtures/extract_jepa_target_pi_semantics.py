"""Extract the pi0.5 JEPA-target selection semantics as a fixture, by calling the REAL pi loader.

Run under the openpi env (`cd robocasa_openpi && uv run python <this>`). Imports
`openpi.groot_utils.groot_openpi_dataset._wsm_jepa_target` and drives it over synthetic omega grids
written to a temp feats root, so the fixture records what the SHIPPED pi code actually selects --
not a reimplementation of it, which would be able to agree with a bug.

The groot side must reproduce `w_next` / `valid` exactly for the same (frame_indices, frame, k).
"""

import tempfile
from pathlib import Path

import numpy as np
import openpi.groot_utils.groot_openpi_dataset as loader

TASK = "SyntheticTask"
CASES = [
    # (frame_indices, list of frames to probe)
    (np.arange(0, 80, 8, dtype=np.int64), [0, 1, 7, 8, 9, 40, 63, 71, 72, 79]),
    (np.arange(0, 24, 8, dtype=np.int64), [0, 8, 15, 16, 23]),  # short episode: heavy tail padding
    (np.array([0], dtype=np.int64), [0, 5]),  # degenerate single-row grid
]
KS = [1, 2, 16]

root = Path(tempfile.mkdtemp())
out = {}
for case_i, (fi, frames) in enumerate(CASES):
    rng = np.random.default_rng(100 + case_i)
    w = rng.normal(0, 1, (len(fi), 8)).astype(np.float16)  # small Dw; selection is dim-agnostic
    demo_dir = root / TASK / f"demo_{case_i:06d}"
    demo_dir.mkdir(parents=True, exist_ok=True)
    np.savez(demo_dir / "w.npz", w=w, frame_indices=fi)
    out[f"case{case_i}_w"] = w.astype(np.float32)
    out[f"case{case_i}_fi"] = fi
    out[f"case{case_i}_frames"] = np.asarray(frames, dtype=np.int64)

    for k in KS:
        # Drive the real loader: point it at our temp root, stub task resolution, clear its cache.
        loader._WSM_FEATS_ROOT = str(root)
        loader._wsm_task_dirs = [TASK]
        loader._WSM_JEPA_NUM_FUTURES = k
        loader._wsm_demo_cache.clear() if hasattr(loader._wsm_demo_cache, "clear") else None
        loader._wsm_demo_cache = type(loader._wsm_demo_cache)()

        tgts, valids = [], []
        for f in frames:
            t, v = loader._wsm_jepa_target(str(root / TASK / "x"), case_i, int(f))
            tgts.append(np.asarray(t, dtype=np.float32))
            valids.append(np.asarray(v, dtype=np.bool_))
        out[f"case{case_i}_k{k}_target"] = np.stack(tgts)
        out[f"case{case_i}_k{k}_valid"] = np.stack(valids)

out["ks"] = np.asarray(KS, dtype=np.int64)
out["num_cases"] = np.asarray(len(CASES), dtype=np.int64)
dest = str(Path(__file__).resolve().parent / "jepa_target_pi_semantics.npz")
np.savez(dest, **out)
print("wrote", dest)
for k in KS:
    print(f"  k={k} case0 valid[:5] =", out[f"case0_k{k}_valid"][:5].tolist())

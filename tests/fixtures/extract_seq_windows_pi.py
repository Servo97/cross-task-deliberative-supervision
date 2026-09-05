"""Extract pi's contiguous-window enumeration as a fixture, by calling the REAL pi function.

Run under the openpi env. The groot sequence dataset's `contiguous_episode_windows` must reproduce
these window boundaries exactly, or the two backbones' RoboTTT arms see different recipes.
"""

import json
import pathlib

from openpi.groot_utils.groot_openpi_dataset import contiguous_episode_windows

LENGTHS = [0, 7, 8, 63, 64, 65, 120, 200, 511]
CASES = [(8, 8), (8, 1), (4, 8), (2, 16), (1, 8)]

out = []
for wl, cs in CASES:
    got = contiguous_episode_windows(LENGTHS, wl, cs)
    per = {}
    for traj, steps in got:
        per.setdefault(int(traj), []).append([int(s) for s in steps])
    out.append({"window_len": wl, "chunk_stride": cs, "by_traj": {str(k): v for k, v in per.items()}})

dest = str(pathlib.Path(__file__).resolve().parent / "seq_windows_pi.json")
json.dump({"lengths": LENGTHS, "cases": out}, open(dest, "w"), indent=1)
print("wrote", dest)
for c in out:
    n = sum(len(v) for v in c["by_traj"].values())
    print(f"  L={c['window_len']} stride={c['chunk_stride']}: {n} windows")

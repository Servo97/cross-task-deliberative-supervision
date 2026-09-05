#!/usr/bin/env python3
"""Fold the A5 (Stage-Q fast-weight / RoboTTT-proxy) arm into RoboCerebra protocol v3.

A5 is NOT a full 800-trial cell. Its launcher gave all 8 shard runners the SAME
--wsm-env-id ('env0'); the Stage-Q server's duplicate-env guard therefore rejected
7 of 8 runners on their first batched inference, and only one shard per cell survived:

    v3_a5_stageq_q2_6mode.json     shard 5  -> 60 trials  (6 modes x 10 cases x trial 5)
    v3_a5_stageq_q2_memtopup.json  shard 7  -> 20 trials  (2 modes x 10 cases x trial 17)

Sharding in this harness is over TRIAL INDEX, not case, so the surviving 80 trials
still span all 10 cases and all 6 modes -- one trial per (mode, case) coordinate
instead of ten. Case-level pairing therefore still has 10 pairs, but each pair rests
on 1 rollout per mode rather than 10.

Because the trial coordinates are a strict subset, the honest contrast is against the
base arm RESTRICTED TO THE SAME (mode, case, trial) coordinates -- the CRN seed rule
blake2b(mode|case|trial|step) makes those rollouts genuinely paired. Comparing A5's 80
against base's 800 would mix denominators, which the v3 protocol forbids.

Emits: matched-coordinate stratified rates + paired deltas for every arm, so A5's
ranking can be read against a like-for-like N.
"""

import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "figures" / "presentation"))
from make_robocerebra_v3 import LOGS, MEM, NOMEM  # noqa: E402

ARMS = [
    ("a0_base", "Plain fine-tune (no memory)"),
    ("a1_gdn_w8", "GDN memory (8-frame window)"),
    ("a2_gdn_w16_hd05", "GDN (16-frame) + history dropout"),
    ("a3_jepa", "JEPA auxiliary loss (train-only)"),
    ("a4_ptrm", "PTRM recursive head"),
    ("a5_stageq_q2", "Stage-Q fast weights (test-time)"),
]


def load_rows(tag, require_complete=False):
    """Union the 6-mode cell and the memory top-up block on (mode, case, trial).

    Unlike make_robocerebra_v3.load_arm this tolerates complete=False and reports it,
    so a partially-served arm can be folded with its true N quoted rather than dropped.
    """
    rows, flags = {}, {}
    for suffix in ("6mode", "memtopup"):
        p = LOGS / f"v3_{tag}_{suffix}.json"
        if not p.exists():
            return None, None
        blob = json.load(open(p))
        flags[suffix] = dict(
            complete=blob.get("complete"),
            shards=blob["provenance"].get("merged_shards"),
            n=len(blob["per_trial"]),
        )
        if require_complete and blob.get("complete") is not True:
            return None, flags
        for r in blob["per_trial"]:
            if r.get("protocol") != "v3":
                raise SystemExit(f"{p}: non-v3 row")
            rows[(r["mode"], r["case"], r["trial"])] = r
    return rows, flags


def rate(rows, modes):
    sel = [r for r in rows.values() if r["mode"] in modes]
    a = sum(r["agent_subtasks"] for r in sel)
    p = sum(r["possible_subtasks"] for r in sel)
    return (100 * a / p if p else float("nan")), a, p, len(sel)


def by_case(rows, modes):
    sel = [r for r in rows.values() if r["mode"] in modes]
    cases = sorted({r["case"] for r in sel}, key=lambda c: int(c[4:]))
    out = []
    for c in cases:
        a = sum(r["agent_subtasks"] for r in sel if r["case"] == c)
        p = sum(r["possible_subtasks"] for r in sel if r["case"] == c)
        out.append(100 * a / p if p else 0.0)
    return np.array(out)


def paired(arm_cases, base_cases):
    """Same procedure the v3 tables use: per-case paired difference, case-level SE, 1.96 SE CI."""
    d = arm_cases - base_cases
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    return d.mean(), se, (d.mean() - 1.96 * se, d.mean() + 1.96 * se)


def main():
    full = {}
    for tag, label in ARMS:
        rows, flags = load_rows(tag)
        if rows is None:
            print(f"!! {label}: artifact absent")
            continue
        full[tag] = (label, rows, flags)

    a5_label, a5, a5_flags = full["a5_stageq_q2"]
    coords = set(a5.keys())

    print("=" * 78)
    print("A5 COVERAGE (true, from all artifacts)")
    print("=" * 78)
    for suffix, f in a5_flags.items():
        print(f"  {suffix:9s} complete={f['complete']}  merged_shards={f['shards']}  n={f['n']}")
    print(f"  union n = {len(a5)} trials")
    print(f"  modes  : {dict(Counter(m for m, _c, _t in coords))}")
    print(
        f"  cases  : {len({c for _m, c, _t in coords})} "
        f"({sorted({c for _m, c, _t in coords}, key=lambda c: int(c[4:]))})"
    )
    print(f"  trials : {sorted({t for _m, _c, t in coords})}")
    base_label, base_full, _bf = full["a0_base"]
    print(f"  base full-cell n = {len(base_full)} (for reference)")

    missing = coords - set(base_full)
    print(
        f"  A5 coordinates absent from base: {len(missing)} -> "
        f"{'pairing is total' if not missing else sorted(missing)[:5]}"
    )

    # ---- CRN validity on the matched coordinates -------------------------------
    print("\n" + "=" * 78)
    print("CRN / MECHANISM CHECKS on the 80 matched coordinates")
    print("=" * 78)
    same_init = sum(1 for k in coords if a5[k].get("first_obs_digest") == base_full[k].get("first_obs_digest"))
    same_act = sum(1 for k in coords if a5[k].get("action_digest") == base_full[k].get("action_digest"))
    print(
        f"  identical env init vs base : {same_init}/{len(coords)} "
        f"({100 * same_init / len(coords):.1f}%) -- paired design {'HOLDS' if same_init == len(coords) else 'BROKEN'}"
    )
    print(
        f"  identical action trajectory: {same_act}/{len(coords)} "
        f"-- {'mechanism is NOT inert at serve time' if same_act == 0 else 'WARNING: arm may be inert'}"
    )

    # ---- matched-coordinate table for every arm --------------------------------
    print("\n" + "=" * 78)
    print("MATCHED-COORDINATE TABLE (all arms restricted to A5's 80 coordinates)")
    print("=" * 78)
    matched = {}
    for tag, (label, rows, _f) in full.items():
        sub = {k: rows[k] for k in coords if k in rows}
        if len(sub) != len(coords):
            print(f"  !! {label}: only {len(sub)}/{len(coords)} coordinates present")
        matched[tag] = (label, sub)

    base_m = matched["a0_base"][1]
    print(f"\n{'arm':36s} {'no-mem':>18s} {'mem':>18s}")
    for tag, (label, sub) in matched.items():
        nm, na, np_, nn = rate(sub, NOMEM)
        mm, ma, mp, mn = rate(sub, MEM)
        print(f"{label:36s} {nm:6.2f}% ({na:3d}/{np_:4d}) {mm:6.2f}% ({ma:3d}/{mp:4d})")

    print("\npaired vs base on the SAME 80 coordinates (10 case pairs, 1.96*SE CI):")
    print(f"{'arm':36s} {'delta no-mem [95% CI]':>34s} {'delta mem [95% CI]':>34s}")
    for tag, (label, sub) in matched.items():
        if tag == "a0_base":
            print(f"{label:36s} {'(base)':>34s} {'(base)':>34s}")
            continue
        dn, sn, cn = paired(by_case(sub, NOMEM), by_case(base_m, NOMEM))
        dm, sm, cm = paired(by_case(sub, MEM), by_case(base_m, MEM))
        fn = f"{dn:+6.2f} [{cn[0]:+6.2f}, {cn[1]:+6.2f}]"
        fm = f"{dm:+6.2f} [{cm[0]:+6.2f}, {cm[1]:+6.2f}]"
        sig_n = "*" if (cn[0] > 0 or cn[1] < 0) else " "
        sig_m = "*" if (cm[0] > 0 or cm[1] < 0) else " "
        print(f"{label:36s} {fn:>33s}{sig_n} {fm:>33s}{sig_m}")

    # ---- A5 headline + per-mode -------------------------------------------------
    print("\n" + "=" * 78)
    print("A5 HEADLINE")
    print("=" * 78)
    for name, modes in (("no-memory", NOMEM), ("memory", MEM)):
        ar, aa, ap, an = rate(a5, modes)
        br, ba, bp, bn = rate(base_m, modes)
        bfr, _, _, bfn = rate(base_full, modes)
        print(
            f"  {name:10s}  A5 {ar:6.2f}% ({aa}/{ap}, n={an} trials)   "
            f"matched base {br:6.2f}% ({ba}/{bp})   full base {bfr:6.2f}% (n={bfn})"
        )

    MODES = ["Ideal", "Observation_Mismatching", "Random_Disturbance", "Memory_Execution", "Memory_Exploration", "Mix"]
    print("\n  per-mode (A5 / matched base), subtask completion %:")
    for m in MODES:
        ar, aa, ap, an = rate(a5, (m,))
        br, ba, bp, _ = rate(base_m, (m,))
        print(f"    {m:26s} {ar:6.2f}% ({aa:3d}/{ap:3d}, n={an:2d})   base {br:6.2f}% ({ba:3d}/{bp:3d})")

    # ---- episode success + v2-legacy ratio -------------------------------------
    print("\n  episode success (A5 / matched base):")
    for m in MODES:
        a_sel = [r for r in a5.values() if r["mode"] == m]
        b_sel = [r for r in base_m.values() if r["mode"] == m]
        print(
            f"    {m:26s} {sum(1 for r in a_sel if r['success'])}/{len(a_sel)}"
            f"   base {sum(1 for r in b_sel if r['success'])}/{len(b_sel)}"
        )

    a = sum(r["agent_subtasks"] for r in a5.values())
    legacy = sum(r["agent_subtasks_v2legacy"] for r in a5.values())
    p = sum(r["possible_subtasks"] for r in a5.values())
    print(
        f"\n  v3 vs v2-legacy scoring on A5's own rollouts: "
        f"{100 * a / p:.2f}% vs {100 * legacy / p:.2f}% ({a / max(legacy, 1):.1f}x)"
    )

    # ---- power: what could this N have detected? --------------------------------
    print("\n" + "=" * 78)
    print("POWER OF THE REDUCED CELL")
    print("=" * 78)
    for name, modes in (("no-memory", NOMEM), ("memory", MEM)):
        dm, sm, cm = paired(by_case(a5, modes), by_case(base_m, modes))
        half = 1.96 * sm
        print(
            f"  {name:10s}  delta {dm:+.2f} pp, 95% CI half-width {half:.2f} pp "
            f"-> anything smaller than +/-{half:.1f} pp is indistinguishable at this N"
        )


if __name__ == "__main__":
    main()

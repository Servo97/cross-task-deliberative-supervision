#!/usr/bin/env python3
"""Judge the retrained encoder against the PRE-REGISTERED G1b bar. No thresholds are chosen here.

Bar (written into internal_planning_and_todos/aug_10/robocerebra_ablation_tree.md before the run
finished, and reproduced verbatim below):

| metric                      | FAIL if | PASS needs |
|-----------------------------|---------|------------|
| temporal coherence gap      | < 0.15  | >= 0.40    |
| effective rank (of 512)     | < 6.5   | >= 8.0     |
| between-episode var frac    | < 0.08  | >= 0.20    |

PASS = all three PASS thresholds met. FAIL = any FAIL trigger fires. Otherwise INDETERMINATE.
Metrics come from the heldout-episode evaluation of the checkpoint selected on the primary
discriminator (temporal coherence gap), not the last step.
"""

from __future__ import annotations

import argparse
import json
import pathlib

BAR = {
    "temporal_coherence_gap": {"fail_below": 0.15, "pass_at_or_above": 0.40, "in_domain_ref": 0.785, "frozen": 0.030},
    "effective_rank": {"fail_below": 6.5, "pass_at_or_above": 8.0, "in_domain_ref": 10.94, "frozen": 6.15},
    "between_episode_variance_fraction": {
        "fail_below": 0.08,
        "pass_at_or_above": 0.20,
        "in_domain_ref": 0.441,
        "frozen": 0.043,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    history = json.loads(pathlib.Path(args.history).read_text())
    best = max(history, key=lambda h: h["temporal_coherence_gap"])

    rows, fails, passes = {}, [], []
    for metric, bounds in BAR.items():
        value = float(best[metric])
        failed = value < bounds["fail_below"]
        passed = value >= bounds["pass_at_or_above"]
        rows[metric] = {"value": value, **bounds, "fails": failed, "passes": passed}
        fails.append(failed)
        passes.append(passed)

    verdict = "FAIL" if any(fails) else ("PASS" if all(passes) else "INDETERMINATE")

    # The bar must discriminate: the frozen encoder has to trip every FAIL trigger.
    control_ok = all(b["frozen"] < b["fail_below"] for b in BAR.values())

    report = {
        "verdict": verdict,
        "selected_checkpoint": {
            "path": args.best,
            "step": best["step"],
            "selected_on": "temporal_coherence_gap (primary discriminator)",
        },
        "heldout_metrics": rows,
        "sigreg_stat_at_best": best.get("sigreg_stat"),
        "n_heldout_frames": best.get("n_frames"),
        "negative_control_bar_discriminates": control_ok,
        "final_step_metrics": history[-1],
        "note": (
            "Thresholds were pre-registered before training finished; this script only "
            "applies them. Effective rank far ABOVE the in-domain reference is over-"
            "whitening, not extra quality — the coherence gap is the primary discriminator."
        ),
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

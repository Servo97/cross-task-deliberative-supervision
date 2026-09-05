from __future__ import annotations

import json

import pytest

from gcp_tpu.wait_then_run_upstream_ttt import validate_finished_eval


def _write_eval(tmp_path, *, episodes=50, failures=None, launcher_returncode=0):
    eval_root = tmp_path / "result"
    (eval_root / "eval").mkdir(parents=True)
    (eval_root / "supervisor.json").write_text(
        json.dumps(
            {
                "finished_utc": "2026-08-03T00:00:00+00:00",
                "launcher_returncode": launcher_returncode,
                "eval_id": "sealed-eval",
                "task": "PickXtimes",
                "arm": "s0",
            }
        )
    )
    (eval_root / "eval" / "launch_manifest.json").write_text(
        json.dumps(
            {
                "returncodes": [0] * 8,
                "episode_audit": {
                    "episodes": episodes,
                    "harness_failures": [] if failures is None else failures,
                },
            }
        )
    )
    return eval_root


def test_validate_finished_eval_accepts_exact_sealed_result(tmp_path):
    gate = validate_finished_eval(_write_eval(tmp_path), 50)
    assert gate["episodes"] == 50
    assert gate["harness_failures"] == 0
    assert gate["task"] == "PickXtimes"


@pytest.mark.parametrize(
    ("episodes", "failures", "launcher_returncode"),
    [(49, None, 0), (50, [{"reason": "timeout"}], 0), (50, None, 1)],
)
def test_validate_finished_eval_fails_closed(tmp_path, episodes, failures, launcher_returncode):
    eval_root = _write_eval(
        tmp_path,
        episodes=episodes,
        failures=failures,
        launcher_returncode=launcher_returncode,
    )
    with pytest.raises(RuntimeError):
        validate_finished_eval(eval_root, 50)

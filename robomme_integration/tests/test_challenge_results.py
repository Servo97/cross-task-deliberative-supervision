from __future__ import annotations

import csv

from robomme_integration.eval.challenge_results import (
    CHALLENGE_TASK_ORDER,
    SUITES,
    aggregate_claims,
    aggregate_materialized,
    write_matrix,
)


def _claims(*, one_model: bool) -> list[dict]:
    claims = []
    for index, task in enumerate(CHALLENGE_TASK_ORDER):
        run = "mt-all16-s0" if one_model else f"st-{task}-s0"
        claims.append(
            {
                "kind": "robomme_fixed50_complete",
                "run_id": run,
                "eval_id": f"{run}-{task}-fixed50",
                "task": task,
                "arm": "s0",
                "episodes": 50,
                "successes": index,
                "checkpoint_uri": f"s3://checkpoints/{run}/19999",
            }
        )
    return claims


def test_one_multitask_model_exactly_qualifies_for_official_800_episode_score():
    scorecard = aggregate_claims(_claims(one_model=True), method_id="S0", training_scope="multitask")
    assert scorecard["challenge_comparable"] is True
    assert scorecard["qualification_failures"] == []
    assert scorecard["overall"]["episodes"] == 800
    assert scorecard["overall"]["successes"] == sum(range(16))
    assert all(scorecard["suite_scores"][suite]["episodes"] == 200 for suite in SUITES)
    assert scorecard["task_scores"]["BinFill"]["episodes"] == 50


def test_single_task_specialists_never_masquerade_as_challenge_entry():
    scorecard = aggregate_claims(_claims(one_model=False), method_id="S0 specialists", training_scope="single_task")
    assert scorecard["challenge_comparable"] is False
    assert scorecard["overall"]["episodes"] == 800
    assert any("single-task specialists" in item for item in scorecard["qualification_failures"])
    assert len(scorecard["provenance"]["run_ids"]) == 16


def test_matrix_has_official_task_and_suite_columns(tmp_path):
    scorecard = aggregate_claims(_claims(one_model=True), method_id="S0", training_scope="multitask")
    output = tmp_path / "matrix.csv"
    write_matrix([scorecard], output)
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["method_id"] == "S0"
    assert rows[0]["ButtonUnmaskSwap"] != ""
    assert rows[0]["suite:permanence_spatial"] != ""
    assert rows[0]["episodes"] == "800"


def test_materialized_official_suites_validate_into_challenge_scorecard():
    run_id = "mt-all16-s0-fixed800"
    benchmarks = {
        "counting_temporal": "RoboMMEOfficialHistoryBenchmark_counting",
        "permanence_spatial": "RoboMMEOfficialHistoryBenchmark_permanence",
        "reference_object": "RoboMMEOfficialHistoryBenchmark_reference",
        "imitation_procedural": "RoboMMEOfficialHistoryBenchmark_imitation",
    }
    aggregates = []
    for suite, benchmark in benchmarks.items():
        tasks = []
        for task in SUITES[suite]:
            episodes = [{"episode_idx": index, "metrics": {"success": index == 0}} for index in range(50)]
            tasks.append({"task": task, "episodes": episodes, "num_episodes": 50, "mean_success": 0.02})
        aggregates.append(
            {
                "benchmark": benchmark,
                "eval_id": f"{run_id}-{benchmark}",
                "num_episodes_total": 200,
                "mean_success": 0.02,
                "tasks": tasks,
            }
        )

    scorecard = aggregate_materialized(
        aggregates,
        method_id="S0",
        training_scope="multitask",
        run_id=run_id,
        arm="s0",
        checkpoint_uri="s3://checkpoints/s0/59999",
    )
    assert scorecard["challenge_comparable"] is True
    assert scorecard["overall"]["successes"] == 16
    assert scorecard["overall"]["episodes"] == 800

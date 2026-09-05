from pathlib import Path

from robomme_integration.ablation_matrix import METHODS, expanded_cells, validate_registry
from robomme_integration.eval.challenge_results import CHALLENGE_TASK_ORDER


def test_full_matrix_covers_every_method_on_all_tasks_plus_one_multitask_model():
    validate_registry(Path(__file__).resolve().parents[2])
    cells = expanded_cells()
    assert len(cells) == len(METHODS) * 17
    for method in METHODS:
        selected = [cell for cell in cells if cell["method_id"] == method["id"]]
        assert {cell["task"] for cell in selected if cell["training_scope"] == "single_task"} == set(
            CHALLENGE_TASK_ORDER
        )
        challenge = [cell for cell in selected if cell["training_scope"] == "multitask"]
        assert len(challenge) == 1 and challenge[0]["eval_episodes"] == 800


def test_scientific_identity_guardrails_are_explicit():
    by_id = {method["id"]: method for method in METHODS}
    assert by_id["r1"]["robomme_arm"] is None
    assert by_id["r1"]["single_task"] == "implementation_required"
    assert by_id["q2"]["label"].startswith("legacy project Q2")
    assert by_id["wsm_d8"]["fast_weights"] is False
    assert by_id["ptrm"]["robomme_arm"] == "ptrm"
    assert by_id["ptrm"]["fast_weights"] is False
    assert by_id["ptrm"]["eval_protocol"] == "E0_only_K1_sigma0"
    assert by_id["q3"]["fast_weights"] is True and by_id["q3"]["workspace"] is True
    assert by_id["q1"]["fast_weights"] is False and by_id["q1"]["workspace"] is True
    assert by_id["q1"]["steering"] == "tanh"
    assert by_id["wsm_d8_drop05"]["paper_priority"] == "high"
    assert by_id["wsm_d16_drop05"]["paper_priority"] == "high"
    assert by_id["causal_v1"]["single_task"] == "workspace_supervision_required"
    assert by_id["gdn8_jepa_l01_k1"]["paper_priority"] == "deprioritized"
    cells = {cell["cell_id"]: cell for cell in expanded_cells()}
    assert cells["mt::all16::s0"]["readiness"] == "ready"
    assert cells["mt::all16::q2"]["readiness"] == "ready"
    assert cells["mt::all16::wsm_cfg"]["readiness"] == "all16_workspace_artifacts_required"
    assert cells["mt::all16::ptrm"]["readiness"] == "all16_workspace_artifacts_required"

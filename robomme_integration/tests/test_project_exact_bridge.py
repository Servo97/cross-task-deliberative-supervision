from __future__ import annotations

import asyncio

import numpy as np
import pytest

from robomme_integration.eval.project_exact_server import (
    PROTOCOL_ID,
    Q2_BLOCKER,
    ProjectExactBridge,
    validate_server_metadata,
)
from robomme_integration.eval.project_exact_source_audit import (
    POLICY_SOURCE_COMMIT,
    audit_imported_git_source,
    require_file_in_source,
    require_pinned_commit,
    require_pinned_file,
    sha256_file,
)


class _Context:
    def __init__(self, session_id: str, episode_id: str):
        self.session_id = session_id
        self.episode_id = episode_id
        self.step = 0

    def _increment_step(self):
        self.step += 1


class _Core:
    def __init__(self):
        self.starts = []
        self.ends = []
        self.predictions = []
        self.result = {"actions": np.zeros((20, 8), dtype=np.float32)}

    async def on_episode_start(self, config, ctx):
        self.starts.append((config, ctx))

    async def on_episode_end(self, result, ctx):
        self.ends.append((result, ctx))

    def predict(self, observation, ctx):
        self.predictions.append((observation, ctx.step))
        return self.result


def _bridge(core: _Core, arm: str = "s0") -> ProjectExactBridge:
    return ProjectExactBridge(
        "/unused",
        arm=arm,
        checkpoint_sha256="a" * 64,
        project_source_sha256="b" * 64,
        openpi_source_sha256="c" * 64,
        core=core,
        context_factory=_Context,
        known_tasks=frozenset({"PickXtimes"}),
    )


def _reset_payload(**updates):
    payload = {
        **_bridge(_Core()).metadata,
        "task_name": "PickXtimes",
        "episode_idx": 3,
    }
    payload.update(updates)
    return payload


def _observation():
    return {
        "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((256, 256, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "pick the cube three times",
    }


def test_exact_connection_preserves_full_plan_and_advances_the_environment_clock_by_16():
    core = _Core()
    connection = _bridge(core).connection()
    response = asyncio.run(connection.reset(_reset_payload()))
    assert response == {"reset_finished": True, "protocol_id": PROTOCOL_ID}
    first = connection.infer(_observation())
    second = connection.infer(_observation())
    assert first["actions"].shape == second["actions"].shape == (20, 8)
    assert [step for _obs, step in core.predictions] == [0, 16]
    assert core.predictions[0][0]["images"]["agentview"].dtype == np.uint8
    assert core.predictions[0][0]["states"].shape == (8,)
    assert connection.ctx.step == 32
    asyncio.run(connection.close())
    assert len(core.ends) == 1


def test_reset_and_history_paths_fail_closed():
    connection = _bridge(_Core()).connection()
    with pytest.raises(ValueError, match="contract mismatch"):
        asyncio.run(connection.reset(_reset_payload(execution_horizon=10)))

    connection = _bridge(_Core()).connection()
    asyncio.run(connection.reset(_reset_payload()))
    with pytest.raises(RuntimeError, match="execution-only"):
        connection.add_buffer({"add_buffer": True})
    with pytest.raises(RuntimeError, match="exactly one"):
        asyncio.run(connection.reset(_reset_payload()))


def test_action_and_observation_contracts_fail_closed():
    core = _Core()
    connection = _bridge(core).connection()
    asyncio.run(connection.reset(_reset_payload()))
    bad_obs = _observation()
    bad_obs["observation/state"] = np.zeros((9,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        connection.infer(bad_obs)

    core.result = {"actions": np.zeros((16, 8), dtype=np.float32)}
    with pytest.raises(RuntimeError, match=r"finite \(20,8\)"):
        connection.infer(_observation())


def test_q2_is_rejected_instead_of_silently_changing_its_update_semantics():
    with pytest.raises(ValueError, match="stride/commit=10") as error:
        _bridge(_Core(), arm="q2")
    assert str(error.value) == Q2_BLOCKER


def test_evaluator_must_match_every_sealed_server_identity_field():
    metadata = _bridge(_Core()).metadata
    validate_server_metadata(metadata, metadata.copy())
    wrong = metadata.copy()
    wrong["checkpoint_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="checkpoint_sha256"):
        validate_server_metadata(metadata, wrong)


def test_source_audit_binds_the_imported_file_to_one_clean_exact_git_tree(tmp_path):
    root = tmp_path / "source"
    imported = root / "pkg" / "module.py"
    sibling = root / "client" / "wire.py"
    imported.parent.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)
    imported.write_text("VALUE = 1\n", encoding="utf-8")
    sibling.write_text("VALUE = 2\n", encoding="utf-8")

    def clean_git(_cwd, arguments):
        if tuple(arguments[:3]) == ("ls-files", "--error-unmatch", "--"):
            return f"{arguments[3]}\n"
        return {
            ("rev-parse", "--show-toplevel"): f"{root}\n",
            ("rev-parse", "HEAD"): f"{POLICY_SOURCE_COMMIT}\n",
            ("status", "--porcelain=v1", "--untracked-files=no"): "",
        }[tuple(arguments)]

    audit = audit_imported_git_source(
        "policy",
        imported,
        POLICY_SOURCE_COMMIT,
        git_reader=clean_git,
    )
    assert audit.manifest_record() == {
        "commit": POLICY_SOURCE_COMMIT,
        "tracked_tree_clean": True,
    }
    require_file_in_source(audit, "wire client", sibling, git_reader=clean_git)


@pytest.mark.parametrize(
    ("head", "status", "message"),
    (
        ("d" * 40, "", "HEAD drifted"),
        (POLICY_SOURCE_COMMIT, " M pkg/module.py", "tracked changes"),
    ),
)
def test_source_audit_rejects_wrong_commit_or_tracked_edits(tmp_path, head, status, message):
    root = tmp_path / "source"
    imported = root / "pkg" / "module.py"
    imported.parent.mkdir(parents=True)
    imported.write_text("VALUE = 1\n", encoding="utf-8")

    def drifted_git(_cwd, arguments):
        if tuple(arguments[:3]) == ("ls-files", "--error-unmatch", "--"):
            return f"{arguments[3]}\n"
        return {
            ("rev-parse", "--show-toplevel"): f"{root}\n",
            ("rev-parse", "HEAD"): f"{head}\n",
            ("status", "--porcelain=v1", "--untracked-files=no"): status,
        }[tuple(arguments)]

    with pytest.raises(RuntimeError, match=message):
        audit_imported_git_source(
            "policy",
            imported,
            POLICY_SOURCE_COMMIT,
            git_reader=drifted_git,
        )


def test_source_audit_rejects_an_untracked_import_inside_a_clean_tree(tmp_path):
    root = tmp_path / "source"
    imported = root / "pkg" / "injected.py"
    imported.parent.mkdir(parents=True)
    imported.write_text("VALUE = 'untracked'\n", encoding="utf-8")

    def clean_but_untracked(_cwd, arguments):
        if tuple(arguments[:3]) == ("ls-files", "--error-unmatch", "--"):
            return ""
        return {
            ("rev-parse", "--show-toplevel"): f"{root}\n",
            ("rev-parse", "HEAD"): f"{POLICY_SOURCE_COMMIT}\n",
            ("status", "--porcelain=v1", "--untracked-files=no"): "",
        }[tuple(arguments)]

    with pytest.raises(RuntimeError, match="not the tracked source"):
        audit_imported_git_source(
            "policy",
            imported,
            POLICY_SOURCE_COMMIT,
            git_reader=clean_but_untracked,
        )


def test_pinned_labels_and_reference_bytes_are_both_required(tmp_path):
    reference = tmp_path / "official_reference_eval.py"
    reference.write_bytes(b"sealed reference\n")
    digest = sha256_file(reference)
    require_pinned_commit("policy source", POLICY_SOURCE_COMMIT, POLICY_SOURCE_COMMIT)
    require_pinned_file("reference", reference, digest)
    with pytest.raises(RuntimeError, match="must be pinned"):
        require_pinned_commit("policy source", "d" * 40, POLICY_SOURCE_COMMIT)
    reference.write_bytes(b"drifted reference\n")
    with pytest.raises(RuntimeError, match="SHA256 drifted"):
        require_pinned_file("reference", reference, digest)

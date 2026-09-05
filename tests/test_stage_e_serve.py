"""CPU-only tests for the serve-side Stage-E ω consumer (`vla_training/eval/stage_e_serve.py`).

A tiny random Stage-E encoder blob in the real checkpoint schema, a synthetic ω store exported with
the D7-gated reference (`StageEOmegaProducer.omega_episode`), and the REAL sealed serve pieces
(`WSMEvalConditioner`, `WSMPiInferWrapper`) prove: the online front end reproduces the export at
the fp16 floor, the served K-row window is the training-side `window_at` selection, the wrapper
resets and windows Stage-E ω exactly as it does WSMv1 ω, and every provenance check fails closed.

Run: PYTHONPATH=. python -m pytest tests/test_stage_e_serve.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_training.eval import stage_e_serve as ses  # noqa: E402
from vla_training.eval._groot_wsm_eval import WSMEvalConditioner  # noqa: E402
from vla_training.eval.serve_pi_05_wsm import WSMPiInferWrapper  # noqa: E402
from workspace_models.features.stage_e_omega_producer import StageEOmegaProducer, load_stage_e  # noqa: E402
from workspace_models.features.wsm_align import window_at  # noqa: E402
from workspace_models.networks.stage_e_encoder import StageEEncoder  # noqa: E402
from workspace_models.networks.workspace_latent import PatchPool  # noqa: E402

FEAT, LANG, DIM, PATCHES, RAW = 8, 2048, 16, 4, 6  # pooled dim, lang dim, trunk dim, patches/frame, raw patch dim
TASKS = ("MemHeatPot", "MemFruitInSinkLeftFar")


def _tiny_cfg():
    return dict(
        dim=DIM,
        n_layers=1,
        n_dec_layers=1,
        n_heads=4,
        k_slots=4,
        backbone_dim=FEAT,
        proprio_dim=5,
        lang_dim=LANG,
        c_horizon=12,
        max_t=48,
        mlp_ratio=2.0,
        input_norm=False,
    )


def _tiny_blob(seed: int = 0, encoder_id: str = "deadbeef01234567", step: int = 7) -> dict:
    torch.manual_seed(seed)
    cfg = SimpleNamespace(**_tiny_cfg())
    specs = {
        "remembench": {"feat_dim": FEAT, "lang_dim": LANG, "index": 1},
        "robocasa": {"feat_dim": FEAT, "lang_dim": LANG, "index": 0},
    }
    enc = StageEEncoder(cfg, specs)
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model = {f"encoder.{k}": v.clone() for k, v in enc.trunk.state_dict().items()}
    return {
        "model": model,
        "adapters": {k: v.clone() for k, v in enc.adapters.state_dict().items()},
        "cfg": _tiny_cfg(),
        "domains": ["remembench", "robocasa"],
        "domain_index": [1, 0],
        "encoder_id": encoder_id,
        "step": step,
    }


def _write_encoder(tmp_path: Path, **kw) -> Path:
    path = tmp_path / "encoder.pt"
    torch.save(_tiny_blob(**kw), path)
    return path


def _lang_for(task: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(task)) % 2**32)
    return rng.standard_normal(LANG).astype(np.float16).astype(np.float32)


def _make_store(
    tmp_path: Path,
    encoder_path: Path,
    *,
    frames=(9, 13, 7),
    per_episode_lang=False,
    encoder_id="deadbeef01234567",
    step=7,
):
    """Synthetic wsm_pooled + ω store in the export schema, ω written by the reference producer."""
    encoder, _blob = load_stage_e(encoder_path, "cpu")
    omega_root = tmp_path / "omega" / "cell" / "remembench"
    pooled_root = tmp_path / "pooled"
    rng = np.random.default_rng(1)
    for task in TASKS:
        for demo, n in enumerate(frames):
            p = rng.standard_normal((n, FEAT)).astype(np.float16)
            grid = np.arange(n, dtype=np.int64) * ses.STORE_STRIDE
            lang = _lang_for(task) if not per_episode_lang else _lang_for(f"{task}/{demo}")
            producer = StageEOmegaProducer(encoder, "remembench", lang, "cpu")
            w = producer.omega_episode(p).half().cpu().numpy()
            wd = omega_root / task / f"demo_{demo:06d}"
            pd = pooled_root / task / f"demo_{demo:06d}"
            wd.mkdir(parents=True)
            pd.mkdir(parents=True)
            np.savez(
                wd / "w.npz",
                w=w,
                frame_indices=grid,
                lang_global=lang.astype(np.float16),
                encoder_id=np.array(encoder_id),
            )
            np.savez(pd / "p.npz", p=p, frame_indices=grid, lang_global=lang.astype(np.float32))
    (omega_root.parent / "_meta.json").write_text(
        json.dumps({"encoder_id": encoder_id, "encoder_step": step, "n_episodes": len(TASKS) * len(frames)})
    )
    return omega_root, pooled_root


def test_window_rule_lockstep_against_the_fork_and_refuses_a_divergent_copy(tmp_path):
    try:
        info = ses.assert_window_rule_lockstep()
    except FileNotFoundError:
        pytest.skip("fork loader source not available on this host")
    assert info["cases"] > 400
    # A loader whose padding convention drifted (zero-pad instead of repeat-oldest) must be refused.
    source = Path(info["fork_dataset_py"]).read_text(encoding="utf-8")
    bad = source.replace("np.full(k - len(win), win[0], dtype=np.int64)", "np.zeros(k - len(win), dtype=np.int64) - 1")
    assert bad != source
    (tmp_path / "loader.py").write_text(bad, encoding="utf-8")
    with pytest.raises(RuntimeError, match="WINDOW RULE MISMATCH"):
        ses.assert_window_rule_lockstep(tmp_path / "loader.py")


def test_parity_passes_on_the_reference_export_and_matches_the_training_window_rule(tmp_path):
    enc_path = _write_encoder(tmp_path)
    omega_root, pooled_root = _make_store(tmp_path, enc_path)
    table = ses.load_stage_e_task_lang_table(omega_root)  # strict: task_mean store
    assert set(table) == set(TASKS)
    front, prov = ses.load_stage_e_front_end(
        enc_path, domain="remembench", device="cpu", pool_ckpt=None, pooled=True, expect_encoder_id="deadbeef01234567"
    )
    assert prov["encoder_id"] == "deadbeef01234567" and prov["encoder_step"] == 7
    for k in (1, 3, 16):
        report = ses.parity_check(
            front,
            omega_root=str(omega_root),
            pooled_root=str(pooled_root),
            table=table,
            k_window=k,
            stride=ses.STORE_STRIDE,
            device="cpu",
            demos=6,
            frames=0,
        )
        assert report["verdict"] == "PASS", report
        assert report["worst"]["cos"] >= 0.999
        assert report["worst"]["abs"] <= report["fp16_floor"]
        assert report["worst"]["window_abs"] <= report["fp16_floor"]
        assert report["demos"] == 6
    # Serve rows == the training-side window_at selection, also checked directly at one grid time.
    w_path = next((omega_root / TASKS[0]).iterdir()) / "w.npz"
    w = np.asarray(np.load(w_path)["w"], np.float32)
    grid = np.arange(len(w)) * ses.STORE_STRIDE
    expected = window_at(w, grid, int(grid[4]), 3)
    assert expected.shape == (3, DIM) and np.array_equal(expected, w[[2, 3, 4]])
    expected0 = window_at(w, grid, 0, 3)
    assert np.array_equal(expected0, w[[0, 0, 0]])  # left-pad by repeating the oldest


def test_parity_fails_on_a_corrupted_store_row_and_on_a_lang_swap(tmp_path):
    enc_path = _write_encoder(tmp_path)
    omega_root, pooled_root = _make_store(tmp_path, enc_path)
    front, _ = ses.load_stage_e_front_end(enc_path, domain="remembench", device="cpu", pool_ckpt=None, pooled=True)
    table = ses.load_stage_e_task_lang_table(omega_root)
    w_path = omega_root / TASKS[0] / "demo_000000" / "w.npz"
    blob = dict(np.load(w_path))
    blob["w"][3] = -blob["w"][3]
    np.savez(w_path, **blob)
    report = ses.parity_check(
        front,
        omega_root=str(omega_root),
        pooled_root=str(pooled_root),
        table=table,
        k_window=2,
        stride=ses.STORE_STRIDE,
        device="cpu",
        demos=6,
        frames=0,
    )
    assert report["verdict"] == "FAIL"
    # Conditioning on the WRONG task's vector must also fail: the table is load-bearing.
    omega_root2, pooled_root2 = _make_store(tmp_path / "second", enc_path)
    swapped = {TASKS[0]: table[TASKS[1]], TASKS[1]: table[TASKS[0]]}
    report = ses.parity_check(
        front,
        omega_root=str(omega_root2),
        pooled_root=str(pooled_root2),
        table=swapped,
        k_window=2,
        stride=ses.STORE_STRIDE,
        device="cpu",
        demos=4,
        frames=0,
    )
    assert report["verdict"] == "FAIL"


def test_per_episode_store_is_refused_strictly_but_replays_under_store_lang(tmp_path):
    enc_path = _write_encoder(tmp_path)
    omega_root, pooled_root = _make_store(tmp_path, enc_path, per_episode_lang=True)
    with pytest.raises(RuntimeError, match="NOT serve-consistent"):
        ses.load_stage_e_task_lang_table(omega_root)
    smoke = ses.load_stage_e_task_lang_table(omega_root, mode="task_mean_of_store")
    assert set(smoke) == set(TASKS)
    front, _ = ses.load_stage_e_front_end(enc_path, domain="remembench", device="cpu", pool_ckpt=None, pooled=True)
    report = ses.parity_check(
        front,
        omega_root=str(omega_root),
        pooled_root=str(pooled_root),
        table=None,
        k_window=4,
        stride=ses.STORE_STRIDE,
        device="cpu",
        demos=6,
        frames=0,
        lang_mode="store",
    )
    assert report["verdict"] == "PASS"


def test_front_end_provenance_checks_fail_closed(tmp_path):
    enc_path = _write_encoder(tmp_path)
    with pytest.raises(RuntimeError, match="sha256"):
        ses.load_stage_e_front_end(
            enc_path, domain="remembench", device="cpu", pool_ckpt=None, pooled=True, expect_sha256="0" * 64
        )
    with pytest.raises(RuntimeError, match="encoder_id"):
        ses.load_stage_e_front_end(
            enc_path, domain="remembench", device="cpu", pool_ckpt=None, pooled=True, expect_encoder_id="not-the-store"
        )
    with pytest.raises(RuntimeError, match="not in adapters"):
        ses.load_stage_e_front_end(enc_path, domain="robomme", device="cpu", pool_ckpt=None, pooled=True)
    encoder, _ = load_stage_e(enc_path, "cpu")
    with pytest.raises(ValueError, match="wrong pool checkpoint"):
        ses.StageEServeFrontEnd(encoder, "remembench", PatchPool(RAW, FEAT + 1, 1), None, "cpu")
    with pytest.raises(ValueError, match="needs the frozen pool"):
        ses.StageEServeFrontEnd(encoder, "remembench", None, None, "cpu")
    # A non-finite checkpoint is refused by the loader itself.
    blob = _tiny_blob()
    blob["adapters"]["remembench.feat_proj.weight"][0, 0] = float("nan")
    torch.save(blob, tmp_path / "nan.pt")
    with pytest.raises(ValueError, match="NON-FINITE"):
        load_stage_e(tmp_path / "nan.pt", "cpu")
    # The stride contract and the interface are enforced at stack build time.
    with pytest.raises(RuntimeError, match="stride 8"):
        ses.build_stage_e_serve_stack(
            encoder_ckpt=str(enc_path),
            pool_ckpt=None,
            domain="remembench",
            omega_root=None,
            task_lang_table=None,
            lang_table_mode="strict",
            expect_sha256=None,
            fork_dataset_py=None,
            k_window=2,
            stride=4,
            device="cpu",
            interface="tanh",
        )
    with pytest.raises(RuntimeError, match="tanh interface"):
        ses.build_stage_e_serve_stack(
            encoder_ckpt=str(enc_path),
            pool_ckpt=None,
            domain="remembench",
            omega_root=None,
            task_lang_table=None,
            lang_table_mode="strict",
            expect_sha256=None,
            fork_dataset_py=None,
            k_window=2,
            stride=8,
            device="cpu",
            interface="cfg2",
        )


class _FakeTap:
    """Returns raw 'patch tokens' [B, PATCHES, RAW] derived from the frame marker, like the real tap's shape role."""

    def __init__(self):
        self.calls = 0

    def tap(self, frames, state, prompts):
        self.calls += 1
        marker = np.asarray(frames["agentview_left"], dtype=np.float32)[:, 0, 0, 0]
        rng = np.random.default_rng(int(marker.sum()))
        patch = rng.standard_normal((len(prompts), PATCHES, RAW)).astype(np.float32) + marker[:, None, None] / 100.0
        return SimpleNamespace(patch_tokens=patch, lang_emb=np.zeros((len(prompts), LANG), np.float32))


class _FakePolicy:
    metadata = {}

    def __init__(self):
        self.seen = []

    def infer(self, obs, **kwargs):
        assert "wsm_t" not in obs and "wsm_prompt" not in obs
        self.seen.append(np.asarray(obs["wsm_w_window"], np.float32).copy())
        return {"actions": np.zeros((1, 1), np.float32)}


def _obs(env, task, t, marker):
    image = np.full((2, 2, 3), marker, dtype=np.uint8)
    return {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_image": image,
        "observation/state": np.zeros(3, np.float32),
        "prompt": f"terse {task}",
        "wsm_prompt": f"terse {task}",
        "wsm_env_id": env,
        "wsm_task": task,
        "wsm_demo_episode": 0,
        "wsm_t": t,
    }


def test_wrapper_serves_stage_e_windows_through_the_sealed_path(tmp_path):
    """Raw patches -> tiny pool -> Stage-E trunk inside WSMPiInferWrapper: reset at t=0, K rows, stride grid."""
    enc_path = _write_encoder(tmp_path)
    encoder, _ = load_stage_e(enc_path, "cpu")
    torch.manual_seed(3)
    pool = PatchPool(RAW, FEAT, 2).eval()
    front = ses.StageEServeFrontEnd(encoder, "remembench", pool, None, "cpu")
    k = 3
    conditioner = WSMEvalConditioner(front, k_window=k, stride=8, device="cpu")
    table = {TASKS[0]: _lang_for(TASKS[0]), TASKS[1]: _lang_for(TASKS[1])}
    policy, tap = _FakePolicy(), _FakeTap()
    wrapper = WSMPiInferWrapper(policy, tap, conditioner, table, stride=8, max_envs=2, require_wsm_prompt=True)
    markers = [10, 20, 30, 40]
    for i, t in enumerate((0, 8, 16, 24)):
        wrapper.infer(_obs("env0", TASKS[0], t, markers[i]))
        wrapper.infer(_obs("env0", TASKS[0], t + 3, markers[i]))  # same grid: no new tap/encoder work
    assert tap.calls == 4
    assert len(policy.seen) == 8 and all(w.shape == (k, DIM) for w in policy.seen)
    # Reference: the pooled path over the same raw patches, one shot, then the training-side window rule.
    raw = [
        tap.tap({"agentview_left": np.full((1, 2, 2, 3), m, np.uint8)}, None, ["x"]).patch_tokens[0] for m in markers
    ]
    with torch.no_grad():
        p = pool(torch.as_tensor(np.stack(raw))[None]).float().half().float()[0].numpy()  # [4, FEAT]
    producer = StageEOmegaProducer(encoder, "remembench", table[TASKS[0]], "cpu")
    w_ref = producer.omega_episode(p).float().numpy()
    grid = np.arange(4) * 8
    for i in range(4):
        expected = window_at(w_ref, grid, int(grid[i]), k)
        np.testing.assert_allclose(policy.seen[2 * i], expected, rtol=0, atol=2e-3)
        np.testing.assert_allclose(policy.seen[2 * i + 1], expected, rtol=0, atol=2e-3)
    # A second env resets its own prefix at t=0 and gets the FIRST-frame window (all rows the newest).
    wrapper.infer(_obs("env1", TASKS[1], 0, 10))
    first = policy.seen[-1]
    assert np.array_equal(first[0], first[1]) and np.array_equal(first[1], first[2])


def test_table_round_trips_through_the_sealed_npz_schema(tmp_path):
    from vla_training.eval._groot_wsm_eval import load_task_lang_table

    table = {t: _lang_for(t) for t in TASKS}
    out = ses.write_task_lang_table(table, tmp_path / "t" / "task_lang_table.npz")
    back = load_task_lang_table(out)
    assert set(back) == set(TASKS)
    for t in TASKS:
        assert np.array_equal(back[t], table[t])

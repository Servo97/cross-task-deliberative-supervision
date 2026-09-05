#!/usr/bin/env python3
"""M1 / M2 / M3 / M3-ctrl — the RoboMME demo-prefix arms, as a runtime overlay.

Each arm is a ONE-LINE-DIFF clone of the sealed multitask v4 GDN recipe
(`v4_wsm_gdn16_drop02`: gated DeltaNet, window 16, history-dropout 0.2, 60,000 steps, batch 64,
warmup 3,000, 5e-5 -> 5e-6, AdamW b2 .95 wd 1e-6 clip 10, EMA .999, 8 devices, full finetune).
The diff is the `(window, history_dropout)` pair in `V4_DELTANET_RECIPES`, which the sealed
`build_train_config` and `validate_train_config` both read at CALL time — so registering a new
entry gives a complete, gate-passing arm without editing a sealed file.

    arm id                                   window   what the GDN reads
    v4_wsm_gdn_live16_drop02                   16     M1  live omega only  (the standard read)
    v4_wsm_gdn_demo8_drop02                     8     M2  demo prefix only
    v4_wsm_gdn_demo8_live16_drop02             24     M3  [8 demo ; 16 live]   <- parity arm
    v4_wsm_gdn_demo8_live16_drop02_ctrl0b      24     M3-ctrl, ctrl-0b omega store

M3-ctrl is byte-identical to M3 as a RECIPE. Its only difference is the omega store it is pointed
at (structure-free ctrl-0b instead of E1b). It carries its own arm id purely so the checkpoint,
manifest and claim identities cannot collide — the ablation stays one-factor because the factor is
the store, never the code path.

WHY M1 IS NOT THE SEALED ARM. `v4_wsm_gdn16_drop02` reads the legacy dense `omega_f16.npy`
produced by `training/workspace_materialize.py`. M1 reads the Stage-E `w.npz` store on the tap's
stride-8 grid. Same window, same recipe, different representation — that IS the manipulated
variable (the Stage-P "one factor, and the factor is the omega store" pattern). M1 is therefore
the correct paired baseline for M3, and `M1 - M0` is a replication of the sealed null, not a
re-run of a sealed cell.

`install()` mutates three identity tables in memory:

    training.arms          ARM_IDS / TRAINING_ARM_IDS / V4_ARM_IDS / WORKSPACE_ARMS /
                           V4_NEW_PARAMETER_SUBTREES
    training.config        V4_DELTANET_RECIPES              (needs openpi; optional on CPU)
    eval.workspace_runner  WORKSPACE_WINDOWS / WORKSPACE_STEERING_ARMS

`stage_tree` additionally registers the D2 archive in `eval/launch_p5_campaign.py` (5 files patched)
so an A19 milestone eval queue for an M-arm is not refused as an unregistered serving archive.

It is idempotent, it never removes or reorders an existing entry, and it refuses to overwrite one.

    PYTHONPATH=<repo> python -m workspace_models.overlays.rmme_arms --check
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

#: The sealed arm every M-arm is a one-line diff of.
SEALED_PARENT = "v4_wsm_gdn16_drop02"
SEALED_WINDOW, SEALED_DROPOUT = 16, 0.2

#: PRE-REGISTERED 2026-09-02, from the measurement in `scripts/analysis/rmme_prefix_ablation.py`.
#: `gamma_i = exp(-softplus(W_decay w_i + pos_decay_bias_i))`, and `pos_decay_bias` ships zero-init.
#: At that init the 8-slot demo prefix moves the conditioning vector by 6.85e-06 relative — BELOW
#: the fp16 floor on 100 % of windows — so M3 would be mechanically identical to M1 and would have
#: no gradient with which to learn otherwise. Initialising the bias negative starts the recurrence
#: near-lossless and lets training LEARN the decay instead of starting past it:
#:
#:      init                      clamp rel-median   above fp16 floor
#:      0.0 (sealed)                   6.85e-06           0.00
#:      -8 on the demo slots only      1.79e-05           0.00     <- the decay is on the LIVE slots
#:      -4 on the live slots only      1.79e-01           1.00
#:      -4 on all slots  (chosen)      3.82e-01           1.00
#:
#: Applied IDENTICALLY to all four M-arms, so the ablation stays one-factor (the factor is the
#: window). It makes the M-family differ from the sealed parent by a second thing, which is why M1
#: — not the sealed arm — is M3's paired baseline.
#:
#: REQUIRES A 2-LINE OPENPI-FORK DIFF (the fork is content-addressed and separately versioned; this
#: is NOT a `robomme_integration/` edit):
#:     models/wsm_current_cond.py  WSMGatedDeltaNetConditioner.__init__
#:       + pos_decay_bias_init: float = 0.0          # default preserves every sealed checkpoint
#:       - jnp.zeros((self.window_len, self.num_heads), dtype=jnp.float32)
#:       + jnp.full((self.window_len, self.num_heads), pos_decay_bias_init, dtype=jnp.float32)
#:     models/pi0_config.py        plumb `wsm_cond_pos_decay_bias_init` through to the conditioner
#: It moves the openpi tarball's content address; re-derive the run_id from the dry run at fire time.
POS_DECAY_BIAS_INIT = -4.0

#: The openpi archive the M-arms pair with: the v4-advanced archive `24bd889d…` (what every sealed
#: `V4_ADVANCED_GDN_ARMS` cell pins) PLUS the D2 diff, rebuilt with `build_deterministic_archive.py`.
#: The rebuild pipeline was proven by reconstructing `24bd889d…` byte-for-byte from its own
#: published tarball BEFORE the diff was applied, so this address differs from the base by the diff
#: and nothing else — `diff -rq` reports exactly 3 files.
D2_OPENPI_SHA = "445d9902a5502d6ce4661c8c42dfa8a2b3ecc3439cb67b41941d1df2b61574dd"
D2_OPENPI_BASE_SHA = "24bd889d3c0b95a7b01cd6ad30a91fdc266fa115fb2ef5ec89fe45c9c5260900"


@dataclasses.dataclass(frozen=True)
class RmmeArm:
    arm_id: str
    label: str  # M0/M1/M2/M3/M3-ctrl
    read: str  # the `rmme_demo_prefix.window_for_step` arm name
    k_demo: int
    k_live: int
    omega_cell: str  # which Stage-E cell's omega store this arm is pointed at

    @property
    def window(self) -> int:
        return self.k_demo + self.k_live

    @property
    def history_dropout(self) -> float:
        return SEALED_DROPOUT

    @property
    def pos_decay_bias_init(self) -> float:
        return POS_DECAY_BIAS_INIT


ARMS: tuple[RmmeArm, ...] = (
    RmmeArm("v4_wsm_gdn_live16_drop02", "M1", "m1_live_only", 0, 16, "E1b-4tap"),
    RmmeArm("v4_wsm_gdn_demo8_drop02", "M2", "m2_demo_only", 8, 0, "E1b-4tap"),
    RmmeArm("v4_wsm_gdn_demo8_live16_drop02", "M3", "m3_demo_live", 8, 16, "E1b-4tap"),
    RmmeArm("v4_wsm_gdn_demo8_live16_drop02_ctrl0b", "M3-ctrl", "m3_ctrl", 8, 16, "ctrl-0b-4tap"),
)
BY_ID = {arm.arm_id: arm for arm in ARMS}
BY_LABEL = {arm.label: arm for arm in ARMS}


def _extend_tuple(module, name: str, values) -> int:
    current = getattr(module, name)
    added = tuple(v for v in values if v not in current)
    if added:
        setattr(module, name, tuple(current) + added)
    return len(added)


def _extend_frozenset(module, name: str, values) -> int:
    current = getattr(module, name)
    added = {v for v in values if v not in current}
    if added:
        setattr(module, name, frozenset(current) | added)
    return len(added)


def install(*, require_config: bool = True) -> dict:
    """Register the four arms. Returns a receipt naming every table that was touched."""
    from robomme_integration.eval import workspace_runner as wr
    from robomme_integration.training import arms as A

    ids = tuple(arm.arm_id for arm in ARMS)
    collisions = [i for i in ids if i in A.ARM_IDS]
    if collisions and any(BY_ID[i].window != wr.WORKSPACE_WINDOWS.get(i) for i in collisions):
        raise RuntimeError(f"RoboMME overlay would overwrite existing arm ids: {collisions}")

    receipt = {"arms": {}, "config": {}, "workspace_runner": {}}
    receipt["arms"]["ARM_IDS"] = _extend_tuple(A, "ARM_IDS", ids)
    receipt["arms"]["TRAINING_ARM_IDS"] = _extend_tuple(A, "TRAINING_ARM_IDS", ids)
    receipt["arms"]["V4_ARM_IDS"] = _extend_frozenset(A, "V4_ARM_IDS", ids)
    receipt["arms"]["WORKSPACE_ARMS"] = _extend_frozenset(A, "WORKSPACE_ARMS", ids)
    subtrees = 0
    for arm_id in ids:
        # Same new-parameter subtree as every gated-DeltaNet arm: the conditioner registers under
        # `wsm_tanh_cond`, so the missing_regex backfill and every eval artifact gate are unchanged.
        if arm_id not in A.V4_NEW_PARAMETER_SUBTREES:
            A.V4_NEW_PARAMETER_SUBTREES[arm_id] = ("wsm_tanh_cond",)
            subtrees += 1
    receipt["arms"]["V4_NEW_PARAMETER_SUBTREES"] = subtrees

    added = 0
    for arm in ARMS:
        if arm.arm_id not in wr.WORKSPACE_WINDOWS:
            wr.WORKSPACE_WINDOWS[arm.arm_id] = arm.window
            added += 1
    wr.WORKSPACE_STEERING_ARMS = frozenset(wr.WORKSPACE_WINDOWS)
    receipt["workspace_runner"]["WORKSPACE_WINDOWS"] = added

    try:
        from robomme_integration.training import config as C
    except ImportError as error:  # openpi is absent on a CPU validation box
        if require_config:
            raise
        receipt["config"] = {"skipped": f"{type(error).__name__}: {error}"}
        return receipt
    recipes = 0
    for arm in ARMS:
        if arm.arm_id not in C.V4_DELTANET_RECIPES:
            C.V4_DELTANET_RECIPES[arm.arm_id] = (arm.window, arm.history_dropout)
            recipes += 1
    receipt["config"]["V4_DELTANET_RECIPES"] = recipes
    parent = C.V4_DELTANET_RECIPES[SEALED_PARENT]
    if parent != (SEALED_WINDOW, SEALED_DROPOUT):
        raise RuntimeError(f"sealed parent recipe drifted: {SEALED_PARENT} = {parent}")
    return receipt


# --------------------------------------------------------------------------------- staged tree
# A NEW TRAINING ARM CANNOT BE RUN WITHOUT THE SEALED IDENTITY TABLES KNOWING ABOUT IT — the
# launcher, the trainer and the eval server all dispatch off them. `stage_tree` therefore does what
# the openpi overlay-v2 seam already does for the policy source: it writes a PATCHED COPY and
# leaves the checkout untouched. `launch.py --source-dir <staged>` then runs from the copy, so the
# sealed tree's `sanitized_source_tree_sha256` — and every existing RoboMME run_id — never moves.


def _build_patches(ids):
    """Anchored, uniqueness-checked edits. Every one is additive; nothing is reordered."""
    id_lines = "".join('    "%s",\n' % i for i in ids)
    quoted = "".join('        "%s",\n' % i for i in ids)
    windows = "".join('    "%s": %d,\n' % (a.arm_id, a.window) for a in ARMS)
    recipes = "".join('    "%s": (%d, %s),\n' % (a.arm_id, a.window, a.history_dropout) for a in ARMS)
    subtrees = "".join('    "%s": ("wsm_tanh_cond",),\n' % i for i in ids)
    inline = ", ".join(repr(i) for i in ids)
    spec_windows = "".join('        "%s": %d,\n' % (a.arm_id, a.window) for a in ARMS)
    spec_windows_split = repr(
        {
            a.arm_id: {
                "k_demo": a.k_demo,
                "k_live": a.k_live,
                "window": a.window,
                "read": a.read,
                "omega_cell": a.omega_cell,
            }
            for a in ARMS
        }
    )
    return (
        # 1. arms.py — the dependency-free identity table
        ("training/arms.py", "ARM_IDS = (\n", "ARM_IDS = (\n" + id_lines),
        ("training/arms.py", "V4_ARM_IDS = frozenset(\n    {\n", "V4_ARM_IDS = frozenset(\n    {\n" + quoted),
        ("training/arms.py", "WORKSPACE_ARMS = frozenset(\n    {\n", "WORKSPACE_ARMS = frozenset(\n    {\n" + quoted),
        ("training/arms.py", "V4_NEW_PARAMETER_SUBTREES = {\n", "V4_NEW_PARAMETER_SUBTREES = {\n" + subtrees),
        # 2. config.py — the recipe table and the ONE new model kwarg
        ("training/config.py", "V4_DELTANET_RECIPES = {\n", "V4_DELTANET_RECIPES = {\n" + recipes),
        (
            "training/config.py",
            '    if history_dropout:\n        model_kwargs["wsm_cond_history_dropout"] = history_dropout\n',
            '    if history_dropout:\n        model_kwargs["wsm_cond_history_dropout"] = history_dropout\n'
            "    if arm in (" + inline + ",):\n"
            "        # D2: at the shipped zero-init the OLDEST window slots are erased below the fp16\n"
            "        # floor before the read, so a demo prefix would be a mechanical null. Identical\n"
            "        # for all four M-arms, so the ablation stays one-factor.\n"
            '        model_kwargs["wsm_cond_pos_decay_bias_init"] = %s\n' % POS_DECAY_BIAS_INIT,
        ),
        # 3. launch.py — advanced-arm membership and the D2-paired openpi archive
        (
            "launch.py",
            "HISTORY_DROPOUT_ARMS = frozenset(\n    {",
            "HISTORY_DROPOUT_ARMS = frozenset(\n    {" + inline + ", ",
        ),
        (
            "launch.py",
            "V4_ADVANCED_GDN_ARMS = frozenset(\n    {\n",
            "V4_ADVANCED_GDN_ARMS = frozenset(\n    {\n" + quoted,
        ),
        (
            "launch.py",
            "    expected_openpi = PTRM_OPENPI if advanced_openpi else OPENPI\n"
            "    expected_openpi_sha = PTRM_OPENPI_SHA if advanced_openpi else OPENPI_SHA\n",
            '    D2_OPENPI_SHA = "%s"\n'
            % D2_OPENPI_SHA
            + '    D2_OPENPI = f"{STUDY_ROOT}/code/openpi/{D2_OPENPI_SHA}.tgz"\n'
            "    if args.arm in (" + inline + ",):\n"
            "        # The v4-advanced archive PLUS the D2 pos_decay_bias_init kwarg. Selected only\n"
            "        # for these arms, so every sealed cell keeps its own pinned archive.\n"
            "        expected_openpi, expected_openpi_sha = D2_OPENPI, D2_OPENPI_SHA\n"
            "    elif advanced_openpi:\n"
            "        expected_openpi, expected_openpi_sha = PTRM_OPENPI, PTRM_OPENPI_SHA\n"
            "    else:\n"
            "        expected_openpi, expected_openpi_sha = OPENPI, OPENPI_SHA\n",
        ),
        # 4. the eval server's steering window table
        ("eval/workspace_runner.py", "WORKSPACE_WINDOWS = {\n", "WORKSPACE_WINDOWS = {\n" + windows),
        # 4b. the p5 eval-campaign launcher's archive registry: a milestone eval queue for an M-arm
        #     pins the D2 archive as its training-matched serving source; the sealed launcher
        #     refuses any archive it does not register ("unregistered OpenPI training archive").
        (
            "eval/launch_p5_campaign.py",
            "    registered_openpi = {\n"
            '        OPENPI_SHA: {"uri": OPENPI, "sha256": OPENPI_SHA, "profile": "standard"},\n',
            "    registered_openpi = {\n"
            "        # RoboMME demo-prefix M-arms (workspace_models/overlays/rmme_arms.py): the v4-advanced\n"
            "        # archive plus the D2 pos_decay_bias_init kwarg; serves under the advanced profile.\n"
            '        "%s": {\n'
            % D2_OPENPI_SHA
            + '            "uri": f"{STUDY_ROOT}/code/openpi/%s.tgz",\n' % D2_OPENPI_SHA
            + '            "sha256": "%s",\n' % D2_OPENPI_SHA
            + '            "profile": "advanced",\n'
            "        },\n"
            '        OPENPI_SHA: {"uri": OPENPI, "sha256": OPENPI_SHA, "profile": "standard"},\n',
        ),
        # 5. launch.py `_arm_spec` -- the SCIENTIFIC MANIFEST must describe the arm it launches.
        #    Found 2026-09-02 by dry-running the staged tree: without these three patches the
        #    manifest reported `steering: null` (the arm is missing from `deltanet_windows`) and
        #    `train_history_dropout: 0.5` (the sealed 0.2 set names only the two sealed GDN arms)
        #    while config.py trained K=16/8/24 at 0.2 -- a manifest that mis-states the recipe.
        (
            "launch.py",
            '        "v4_gdn8_jepa_visreg_l01_k1": 8,\n    }\n    jepa = {\n',
            '        "v4_gdn8_jepa_visreg_l01_k1": 8,\n' + spec_windows + "    }\n    jepa = {\n",
        ),
        (
            "launch.py",
            '            0.2 if arm in {"v4_wsm_gdn8_drop02", "v4_wsm_gdn16_drop02"} else 0.5\n',
            '            0.2 if arm in {"v4_wsm_gdn8_drop02", "v4_wsm_gdn16_drop02", ' + inline + "} else 0.5\n",
        ),
        (
            "launch.py",
            '    if arm in V4_ARM_IDS:\n        spec["protocol"] = "robomme_v4"\n'
            '        spec["new_parameter_subtrees"] = list(V4_NEW_PARAMETER_SUBTREES[arm])\n'
            "    return spec\n",
            '    if arm in V4_ARM_IDS:\n        spec["protocol"] = "robomme_v4"\n'
            '        spec["new_parameter_subtrees"] = list(V4_NEW_PARAMETER_SUBTREES[arm])\n'
            "    if arm in (" + inline + ",):\n"
            "        # RoboMME demo-prefix M-arms (workspace_models/overlays/rmme_arms.py). The window\n"
            "        # split and the non-zero decay-bias init are scientific treatments, recorded\n"
            "        # explicitly so the manifest is self-describing and the identity folds them.\n"
            '        spec["omega_window"] = ' + spec_windows_split + "[arm]\n"
            '        spec["pos_decay_bias_init"] = %s\n'
            % POS_DECAY_BIAS_INIT
            + '        spec["advanced_openpi_capability"] = {\n'
            '            "source_sha256": "%s",\n'
            % D2_OPENPI_SHA
            + '            "base_archive_sha256": "%s",\n' % D2_OPENPI_BASE_SHA
            + '            "requires": ["gated_deltanet", "history_dropout", "pos_decay_bias_init"],\n'
            '            "source_matched_parent": "%s",\n' % SEALED_PARENT + "        }\n"
            "    return spec\n",
        ),
    )


def stage_tree(source, destination, *, overwrite: bool = False) -> dict:
    """Write a PATCHED COPY of `robomme_integration/`. The checkout is never mutated."""
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if not (source / "training" / "arms.py").is_file():
        raise SystemExit("%s is not a robomme_integration tree" % source)
    # `launch.py` imports ABSOLUTELY (`from robomme_integration.training.arms import ...`), so the
    # staged copy has to be importable under that exact package name or the launcher silently picks
    # the SEALED tree back up and rejects the new arms. Enforced rather than documented.
    if destination.name != "robomme_integration":
        raise SystemExit(
            "the staged tree must be named 'robomme_integration' (launch.py imports it by that "
            "absolute package name); pass e.g. %s/robomme_integration" % destination
        )
    if destination.exists():
        if not overwrite:
            raise SystemExit("%s exists; pass --overwrite to replace it" % destination)
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    ids = tuple(arm.arm_id for arm in ARMS)
    applied = []
    for relpath, anchor, replacement in _build_patches(ids):
        path = destination / relpath
        text = path.read_text(encoding="utf-8")
        count = text.count(anchor)
        if count != 1:
            raise SystemExit("anchor for %s occurs %d times, expected exactly 1: %r" % (relpath, count, anchor[:120]))
        path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
        applied.append(relpath)
    for arm_id in ids:
        for relpath in ("training/arms.py", "training/config.py", "launch.py", "eval/workspace_runner.py"):
            if arm_id not in (destination / relpath).read_text(encoding="utf-8"):
                raise SystemExit("%s did not reach %s" % (arm_id, relpath))
    launch_text = (destination / "launch.py").read_text(encoding="utf-8")
    for arm in ARMS:
        if '"%s": %d,' % (arm.arm_id, arm.window) not in launch_text:
            raise SystemExit("%s window did not reach launch._arm_spec" % arm.arm_id)
    if 'spec["pos_decay_bias_init"] = %s' % POS_DECAY_BIAS_INIT not in launch_text:
        raise SystemExit("pos_decay_bias_init did not reach launch._arm_spec")
    if D2_OPENPI_SHA not in (destination / "eval/launch_p5_campaign.py").read_text(encoding="utf-8"):
        raise SystemExit("the D2 archive did not reach eval/launch_p5_campaign.py")
    return {
        "source": str(source),
        "destination": str(destination),
        "files_patched": sorted(set(applied)),
        "arms": list(ids),
        "openpi_sha256": D2_OPENPI_SHA,
        "pos_decay_bias_init": POS_DECAY_BIAS_INIT,
    }


def assert_openpi_archive(openpi_root) -> None:
    """Fail closed if the paired archive predates the D2 kwarg (the archive-pairing lesson)."""
    module = Path(openpi_root).expanduser() / "src/openpi/models/wsm_current_cond.py"
    if not module.is_file():
        raise SystemExit("openpi archive is missing %s" % module)
    text = module.read_text(encoding="utf-8")
    if "_WSM_POS_DECAY_BIAS_INIT = True" not in text or "pos_decay_bias_init" not in text:
        raise SystemExit(
            "the paired openpi archive predates the D2 `pos_decay_bias_init` kwarg; an M-arm would "
            "silently train the ZERO-init conditioner and produce a mechanical null. Pair "
            "%s.tgz." % D2_OPENPI_SHA
        )


def config_diff_table() -> str:
    """The literal one-line diff each arm makes against the sealed parent."""
    rows = [f"{'arm':<40}{'window':>7}{'drop':>6}{'decay_init':>11}  diff vs " + SEALED_PARENT, "-" * 112]
    rows.append(f"{SEALED_PARENT:<40}{SEALED_WINDOW:>7}{SEALED_DROPOUT:>6}{0.0:>11}  (sealed parent)")
    for arm in ARMS:
        rows.append(
            f"{arm.arm_id:<40}{arm.window:>7}{arm.history_dropout:>6}"
            f"{arm.pos_decay_bias_init:>11}  cond_window {SEALED_WINDOW} -> {arm.window} "
            f"[{arm.k_demo} demo ; {arm.k_live} live], omega={arm.omega_cell}"
        )
    return "\n".join(rows)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="install and print the receipt")
    ap.add_argument(
        "--stage-tree",
        default="",
        help="write a PATCHED COPY of robomme_integration/ here and exit; the sealed "
        "checkout is never mutated, so no existing run_id moves",
    )
    ap.add_argument("--source-tree", default="robomme_integration")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--assert-openpi",
        default="",
        help="path to an unpacked openpi archive; fails closed if it predates the D2 pos_decay_bias_init kwarg",
    )
    ap.add_argument(
        "--require-config",
        action="store_true",
        help="fail if robomme_integration.training.config (openpi) is unavailable",
    )
    args = ap.parse_args()
    if args.assert_openpi:
        assert_openpi_archive(args.assert_openpi)
        print(f"[overlay] openpi archive at {args.assert_openpi} carries the D2 kwarg")
    if args.stage_tree:
        receipt = stage_tree(args.source_tree, args.stage_tree, overwrite=args.overwrite)
        print(json.dumps(receipt, indent=1))
        return
    if not args.check:
        raise SystemExit("this module is an overlay; pass --check, --stage-tree or --assert-openpi to exercise it")

    from robomme_integration.eval import workspace_runner as wr
    from robomme_integration.training import arms as A

    before = (len(A.ARM_IDS), len(wr.WORKSPACE_WINDOWS))
    receipt = install(require_config=args.require_config)
    second = install(require_config=args.require_config)  # idempotence
    after = (len(A.ARM_IDS), len(wr.WORKSPACE_WINDOWS))
    assert after == (before[0] + len(ARMS), before[1] + len(ARMS)), (before, after)
    assert second["workspace_runner"]["WORKSPACE_WINDOWS"] == 0, "install is not idempotent"
    assert all(v == 0 for v in second["arms"].values()), "install is not idempotent"
    for arm in ARMS:
        assert arm.arm_id in A.ARM_IDS and arm.arm_id in A.V4_ARM_IDS
        assert arm.arm_id in A.WORKSPACE_ARMS and arm.arm_id in wr.WORKSPACE_STEERING_ARMS
        assert wr.workspace_window_for_arm(arm.arm_id) == arm.window
        assert A.V4_NEW_PARAMETER_SUBTREES[arm.arm_id] == ("wsm_tanh_cond",)
    # the sealed arms must be untouched
    assert wr.WORKSPACE_WINDOWS[SEALED_PARENT] == SEALED_WINDOW
    assert "v4_s0" in A.ARM_IDS and "v4_s0" not in wr.WORKSPACE_STEERING_ARMS
    print(json.dumps(receipt, indent=1, sort_keys=True))
    print()
    print(config_diff_table())
    print("\n[overlay] PASS — 4 arms registered, idempotent, sealed tables extended not rewritten")


if __name__ == "__main__":
    main()

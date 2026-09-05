#!/usr/bin/env python3
"""B6 — H13 strip-export gate: prove an H13 checkpoint serves through the VANILLA pi0.5 stack.

WHY THIS EXISTS. Every H13 arm writes three param subtrees the base checkpoint cannot contain
(``wsm_enc``, ``wsm_dec``, and for R2 ``wsm_h13_jepa_head``). Serving drops them via
``BaseModel.load(remove_extra_params=True)``, which intersects the checkpoint against the vanilla
model's tree and then runs ``check_pytree_equality(..., check_shapes=True)``. If the intersection is
not EXACTLY the vanilla tree — a missing leaf, a renamed subtree, a changed shape — the eval fails at
policy-build time, on a node, after a 60k-step training run has already been paid for. The tree's
gate is therefore to run this against the CANARY checkpoint, before the fulls are submitted.

WHAT IT PROVES, and how cheaply. The whole contract is a function of NAMES and SHAPES, so this reads
the orbax metadata only — the ``_METADATA`` / ``array_metadatas`` / ocdbt manifest objects, ~130 KB —
and never touches the ~12.5 GB of parameter data. Three checks:

  1. STRIP: the H13 checkpoint minus the aux subtrees equals the base anchor checkpoint's tree
     EXACTLY (same leaf paths, same shapes). This is the strongest statement available: the stripped
     artifact is indistinguishable from a baseline post-train checkpoint.
  2. LOAD: the simulated ``remove_extra_params`` intersection against the vanilla ``Pi0`` tree built
     by ``nnx.eval_shape`` reproduces that tree leaf-for-leaf, with matching shapes — i.e. the exact
     assertion ``BaseModel.load`` makes at serve.
  3. AUX ACCOUNTING: the aux subtrees are reported (leaf count + parameter count) so the strip is
     visibly removing what it claims to, and not, say, silently finding nothing to remove.

Usage:
  strip_h13_checkpoint.py --checkpoint s3://.../<run_id>/<step>/params \
      [--base s3://.../s0-c43f076daad4a799/59999/params] [--jepa] [--work DIR]

Exits nonzero with a named failure on any mismatch.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

from launch_guardrails import LONG_CONTEXT_STUDY_S3

#: Everything H13 adds. R1 writes the first two; R2 adds the predictor.
H13_AUX_SUBTREES = ("wsm_enc", "wsm_dec", "wsm_h13_jepa_head", "wsm_lang_head", "wsm_lang_cls")
DEFAULT_BASE = f"{LONG_CONTEXT_STUDY_S3}/checkpoints/pi05/s0/s0-c43f076daad4a799/59999/params"
#: The dnw8 anchor (the sealed 59.9 row). R5-R8 serve with the gated-DeltaNet conditioner KEPT and
#: only the H13 aux stripped, so their stripped tree must equal THIS tree, not the base one — the
#: stripped artifact has to be indistinguishable from an ordinary dnw8 post-train checkpoint.
#: Selecting the wrong base here fails in BOTH directions: against the base anchor a correct gdn8
#: checkpoint looks like it has an extra `wsm_tanh_cond` subtree, and a gdn8 checkpoint that had
#: wrongly lost its conditioner would pass. Use --base-dnw8 for h13e-h13h.
DNW8_BASE = f"{LONG_CONTEXT_STUDY_S3}/checkpoints/pi05/s1/s1-f8e6400ab0e21968/59999/params"
#: The ocdbt data shards are the 12.5 GB this script exists to avoid downloading.
_METADATA_ONLY_EXCLUDE = "ocdbt.process_0/d/*"


def _fetch_metadata(uri: str, dest: pathlib.Path) -> pathlib.Path:
    """Sync only a checkpoint's metadata objects (or accept an already-local path)."""
    if not uri.startswith("s3://"):
        return pathlib.Path(uri)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", uri.rstrip("/"), str(dest), "--exclude", _METADATA_ONLY_EXCLUDE, "--only-show-errors"],
        check=True,
    )
    return dest


def _ckpt_tree(path: pathlib.Path) -> dict[str, tuple]:
    """{'a/b/c': shape} for every leaf of the checkpoint's `params` item, from metadata alone.

    The trailing ``value`` component is dropped exactly as `models.model.restore_params` drops it —
    and under the same all-or-nothing condition. `save_state` writes an ``nnx.State``, so every leaf
    path ends in ``value``; the serve path normalizes that away before comparing against the model's
    pure dict, so a gate that did not would compare two different naming conventions and reject a
    perfectly good checkpoint.
    """
    import jax
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path).item_metadata
    leaves = jax.tree_util.tree_flatten_with_path(metadata["params"])[0]
    paths = [tuple(str(getattr(k, "key", k)) for k in keypath) for keypath, _ in leaves]
    if paths and all(p[-1] == "value" for p in paths):
        paths = [p[:-1] for p in paths]
    return {
        "/".join(path_parts): tuple(getattr(leaf, "shape", ()) or ())
        for path_parts, (_, leaf) in zip(paths, leaves, strict=True)
    }


def _vanilla_tree(
    action_horizon: int,
    max_token_len: int,
    paligemma: str = "gemma_2b",
    expert: str = "gemma_300m",
    *,
    gdn_window: int = 0,
) -> dict[str, tuple]:
    """The serve-side model's param tree — the thing `remove_extra_params` intersects against.

    The variant arguments exist so this gate can be POSITIVE-controlled locally against a `dummy`-size
    checkpoint. A gate that has only ever been shown to reject (the base anchor has no aux subtrees to
    strip) has not been shown to accept, and B6 is what the full runs are gated on.
    """
    import jax
    from flax import nnx
    from openpi.models.pi0_config import Pi0Config

    # `gdn_window > 0` builds the DNW8 SERVE stack (tanh interface, gated-DeltaNet conditioner at
    # that window) instead of the plain vanilla one. R5-R8 serve through that stack with the
    # conditioner KEPT, so checking them against the vanilla tree would verify the wrong contract:
    # it would demand `wsm_tanh_cond` be absent from the intersection when serve in fact requires it.
    extra = {"wsm_tanh": True, "wsm_cond_type": "gated_deltanet", "wsm_cond_window": gdn_window} if gdn_window else {}
    config = Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        paligemma_variant=paligemma,
        action_expert_variant=expert,
        **extra,
    )
    state = jax.eval_shape(lambda r: nnx.state(config.create(r)), jax.random.key(0))
    leaves = jax.tree_util.tree_flatten_with_path(state.to_pure_dict())[0]
    return {"/".join(str(getattr(k, "key", k)) for k in kp): tuple(getattr(v, "shape", ()) or ()) for kp, v in leaves}


def _diff(name: str, want: dict[str, tuple], got: dict[str, tuple]) -> list[str]:
    problems = []
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing:
        problems.append(f"{name}: {len(missing)} leaf/leaves MISSING, e.g. {missing[:5]}")
    if extra:
        problems.append(f"{name}: {len(extra)} UNEXPECTED leaf/leaves, e.g. {extra[:5]}")
    shape_mismatch = [k for k in set(want) & set(got) if want[k] != got[k]]
    if shape_mismatch:
        problems.append(
            f"{name}: {len(shape_mismatch)} shape mismatch(es), e.g. "
            + str([(k, want[k], got[k]) for k in sorted(shape_mismatch)[:3]])
        )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="s3:// or local path to the H13 <step>/params dir")
    ap.add_argument("--base", default=DEFAULT_BASE, help="the base-anchor <step>/params dir to match")
    ap.add_argument(
        "--base-dnw8",
        action="store_true",
        help="compare against the dnw8 anchor instead (h13e-h13h keep the gdn conditioner at serve)",
    )
    ap.add_argument(
        "--gdn-window",
        type=int,
        default=8,
        help="conditioner window used to build the dnw8 SERVE tree for the load check (--base-dnw8)",
    )
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--max-token-len", type=int, default=200)
    ap.add_argument("--paligemma-variant", default="gemma_2b")
    ap.add_argument("--action-expert-variant", default="gemma_300m")
    ap.add_argument("--work", default=None, help="scratch dir for the metadata sync")
    args = ap.parse_args()

    if args.base_dnw8:
        if args.base != DEFAULT_BASE:
            raise SystemExit("--base and --base-dnw8 are alternatives; pass only one")
        args.base = DNW8_BASE
    work = pathlib.Path(args.work) if args.work else pathlib.Path(tempfile.mkdtemp(prefix="h13-strip-"))
    ckpt = _ckpt_tree(_fetch_metadata(args.checkpoint, work / "h13" / "params"))
    base = _ckpt_tree(_fetch_metadata(args.base, work / "base" / "params"))

    aux = {k: v for k, v in ckpt.items() if k.split("/")[0] in H13_AUX_SUBTREES}
    stripped = {k: v for k, v in ckpt.items() if k.split("/")[0] not in H13_AUX_SUBTREES}
    aux_params = sum(int(_prod(s)) for s in aux.values())
    present = sorted({k.split("/")[0] for k in aux})

    print(
        f"[b6] checkpoint leaves={len(ckpt)}  aux leaves={len(aux)} ({present}) "
        f"aux params={aux_params:,}  stripped leaves={len(stripped)}"
    )
    print(f"[b6] base anchor leaves={len(base)}")

    failures = []
    if not aux:
        failures.append(
            "AUX-ABSENT: the checkpoint contains none of "
            f"{H13_AUX_SUBTREES} — this is not an H13 checkpoint, so the strip proves nothing"
        )
    failures += _diff("STRIP vs base anchor", base, stripped)

    vanilla = _vanilla_tree(
        args.action_horizon,
        args.max_token_len,
        args.paligemma_variant,
        args.action_expert_variant,
        gdn_window=args.gdn_window if args.base_dnw8 else 0,
    )
    intersected = {k: v for k, v in ckpt.items() if k in vanilla}
    stack = "dnw8 serve stack" if args.base_dnw8 else "vanilla model"
    failures += _diff(f"LOAD (remove_extra_params intersection) vs {stack}", vanilla, intersected)

    if failures:
        print("\n[b6] FAILED:")
        for problem in failures:
            print(f"  - {problem}")
        sys.exit(1)
    print("[b6] PASS: stripped tree == base anchor tree (names+shapes); vanilla serve stack loads it")


def _prod(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


if __name__ == "__main__":
    main()

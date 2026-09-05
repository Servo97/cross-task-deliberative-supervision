"""Reviewed-patch registry for the AM seam in the official MemoryAttention.

``robomme_integration.training.framesamp_am_jax.require_reviewed_model_patch``
is intentionally fail-closed for the RoboMME lane's sealed-artifact runtime:
``REVIEWED_PATCHED_HISTORY_GEMMA_SHA256`` there is ``None`` and that file is
read-only for this build.  This module supplies the *same* flow for the amkv
lane without touching it:

1. the official base source must hash to the audited value pinned by
   ``OfficialMemoryAttentionAMPatchContract`` (so the patch was derived from
   the source it claims);
2. the vendored patched copy must hash to the value reviewed here;
3. only then may a policy be labelled AM-enabled.

Consequently an amkv result is self-labelling: the pair of hashes below plus
the checkpoint hash identify exactly which computation produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from robomme_integration.training.framesamp_am_jax import (
    OFFICIAL_HISTORY_GEMMA_SHA256,
    OFFICIAL_MEMORY_ATTENTION_CLASS,
    OFFICIAL_POLICY_GIT_SHA,
    PATCH_CONTRACT,
)

PATCHED_MODULE_PATH = Path(__file__).with_name("patched_history_gemma.py")
OFFICIAL_POLICY_TREE_SHA1 = "cb8c33dce3a6f19f731481f30bbab4dca66ee768"

# Reviewed 2026-08-10 against the five patch points of
# OfficialMemoryAttentionAMPatchContract.  Regenerate with
# ``python -m robomme_integration.amkv.patch_contract --record`` after any
# deliberate edit, and re-review the diff before doing so.
REVIEWED_AMKV_PATCH_SHA256 = "7194c30986b0059113ca75dc689b07933b98379b24e1c9bd3b9905a6c131a550"

OFFICIAL_HISTORY_GEMMA_RELATIVE = Path("src/mme_vla_suite/models/integration/history_gemma.py")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_history_gemma_path(policy_source_root: str | Path) -> Path:
    path = Path(policy_source_root) / OFFICIAL_HISTORY_GEMMA_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"official MemoryAttention source is missing under {policy_source_root}: {path}")
    return path


@dataclasses.dataclass(frozen=True)
class ReviewedAMKVPatch:
    """The source identities that make an amkv AM run self-identifying."""

    policy_git_sha: str
    policy_tree_sha1: str
    official_source_sha256: str
    patched_module_sha256: str
    module_class: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def require_reviewed_amkv_patch(policy_source_root: str | Path) -> ReviewedAMKVPatch:
    """Fail closed unless base source and vendored patch both match their pins."""

    source = official_history_gemma_path(policy_source_root)
    PATCH_CONTRACT.validate_unmodified_source(source)
    actual = sha256_file(PATCHED_MODULE_PATH)
    if actual != REVIEWED_AMKV_PATCH_SHA256:
        raise ValueError(
            "vendored amkv MemoryAttention patch drifted from its reviewed hash: "
            f"expected {REVIEWED_AMKV_PATCH_SHA256}, got {actual}"
        )
    return ReviewedAMKVPatch(
        policy_git_sha=OFFICIAL_POLICY_GIT_SHA,
        policy_tree_sha1=OFFICIAL_POLICY_TREE_SHA1,
        official_source_sha256=OFFICIAL_HISTORY_GEMMA_SHA256,
        patched_module_sha256=actual,
        module_class=OFFICIAL_MEMORY_ATTENTION_CLASS,
    )


def install_patched_history_module(model: Any, *, capture: bool = False) -> Any:
    """Swap a constructed ``HistoryPi0``'s linen history module for the patch.

    The nnx bridge holds the linen module on ``.module``; the parameter tree is
    untouched, so this runs after the released checkpoint has been restored and
    is reversible by installing another instance.  Returns the module that was
    replaced so a caller can restore the official path exactly.
    """

    from robomme_integration.amkv.patched_history_gemma import ModuleAM, patched_module_like

    bridge = model.PaliGemma.llm
    current = bridge.module
    if isinstance(current, ModuleAM):
        if current.capture == capture:
            return current
        replacement = ModuleAM(
            configs=current.configs,
            embed_dtype=current.embed_dtype,
            dropout=current.dropout,
            dropout_bdims=current.dropout_bdims,
            adarms=current.adarms,
            integration_type=current.integration_type,
            capture=capture,
        )
    else:
        replacement = patched_module_like(current, capture=capture)
    bridge.module = replacement
    return current


def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="print the current vendored patch hash")
    parser.add_argument("--policy-source-root", help="verify the official base source under this root")
    args = parser.parse_args()
    payload: dict[str, object] = {
        "patched_module_path": str(PATCHED_MODULE_PATH),
        "patched_module_sha256": sha256_file(PATCHED_MODULE_PATH),
        "reviewed_patch_sha256": REVIEWED_AMKV_PATCH_SHA256,
        "official_source_sha256": OFFICIAL_HISTORY_GEMMA_SHA256,
        "policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
        "policy_tree_sha1": OFFICIAL_POLICY_TREE_SHA1,
    }
    if args.policy_source_root:
        PATCH_CONTRACT.validate_unmodified_source(official_history_gemma_path(args.policy_source_root))
        payload["official_source_verified"] = True
    if not args.record:
        payload["matches_review"] = payload["patched_module_sha256"] == REVIEWED_AMKV_PATCH_SHA256
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Serve-ONLY RoboTTT ablation knobs (A1-A5) + A0 probe instrumentation.

The decisive Q0-vs-Q2 eval measured a -3.7 pt cost for online fast weights. These knobs exist to
localize WHERE that cost comes from, by perturbing only the serve-time W lifecycle — never the
trained parameters, never the training recipe, never the model. Everything here is inert unless
``ROBOTTT_ABLATION`` is set, so the decisive/sealed path keeps running the exact code it ran before
(the train/serve parity test is the proof).

Flags (``ROBOTTT_ABLATION``, comma-separated, canonical order freeze,reset,decay,eta,commitfirst):

  ""            unset/empty -> EXACT unmodified behavior (the decisive path; no ACK needed)
  freeze        skip every commit; W stays at its per-episode init for the whole episode (A1)
  reset:N       after every N-th commit, W <- the per-episode init (A2)
  decay:G       after each commit, W <- W0 + G*(W_committed - W0), 0<G<1, elementwise (A3)
  eta:X         scale the EFFECTIVE inner-GD step by X>=0 (A4)
  commitfirst   reorder commit-before-condition (A5) -- a DOCUMENTED NO-OP at serve, see below

`eta:X` is applied without touching any trained parameter and without re-jitting the model. The
inner update is one gradient step at the ENTERING weights,

    commit(W) = W - eta * grad_W L_FW(W)                       (robottt_fast_weights.py Eq 1)

so scaling eta is *exactly* a convex recombination of the entering and committed pytrees:

    W - (X*eta) * grad_W L_FW(W) == W + X * (commit(W) - W)

which is what `apply_post_commit` computes. `softplus(log_inner_lr)` and `base_inner_lr` are read,
never written. eta:0 is therefore an exact no-op step (kept legal on purpose: it is the "the commit
math ran but moved nothing" control, distinct from `freeze`, which never calls commit at all).

commitfirst (A5/G7): at serve the wrapper already commits the executed chunk at the END of the same
`infer`/`infer_batch` call that produced it (`WSMPiInferWrapper._commit_robottt`), strictly before
the next request can be conditioned in `_prepare_batch`. There is never a pending uncommitted chunk
when a new request arrives, so "commit first, then condition" is already the live ordering and the
flag cannot change anything. It is accepted, announced loudly, and recorded in the metadata so the
arm is still self-describing, but it is a no-op by construction.

Safety: an ACTIVE ablation refuses to start without ``ROBOTTT_ABLATION_ACK=smoke``. Ablated servers
advertise `robottt_ablation` in their served metadata, so no result can be mistaken for the sealed
number.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

SPEC_ENV = "ROBOTTT_ABLATION"
ACK_ENV = "ROBOTTT_ABLATION_ACK"
ACK_VALUE = "smoke"
PROBE_ENV = "ROBOTTT_PROBE_LOG"

_KNOWN = ("freeze", "reset", "decay", "eta", "commitfirst")


# ------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RoboTTTAblation:
    """One parsed ``ROBOTTT_ABLATION`` spec. Immutable; parsed exactly once at server startup."""

    spec: str = ""  # canonical re-rendering of the parsed flags ("" == unmodified path)
    raw: str = ""  # exactly what the environment said
    freeze: bool = False
    reset_every: int | None = None
    decay: float | None = None
    eta_scale: float = 1.0
    commit_first: bool = False

    @property
    def active(self) -> bool:
        """True iff anything at all was requested (the ACK gate and the metadata stamp key off this)."""
        return bool(self.spec)

    @property
    def changes_w(self) -> bool:
        """True iff the W trajectory can differ from the unmodified path (commitfirst cannot)."""
        return self.freeze or self.reset_every is not None or self.decay is not None or self.eta_scale != 1.0

    def describe(self) -> str:
        return self.spec or "<none: unmodified serve path>"

    def as_metadata(self) -> dict:
        return {
            "spec": self.spec,
            "raw": self.raw,
            "freeze": self.freeze,
            "reset_every": self.reset_every,
            "decay": self.decay,
            "eta_scale": self.eta_scale,
            "commit_first": self.commit_first,
        }


def _float_arg(name: str, arg: str) -> float:
    try:
        return float(arg)
    except (TypeError, ValueError):
        raise ValueError(f"[robottt-ablation] {name}:<float> expected, got {arg!r}") from None


def parse_ablation(spec: str | None) -> RoboTTTAblation:
    """Parse a ``ROBOTTT_ABLATION`` spec. Unknown/garbage tokens are a HARD failure.

    An empty/whitespace/None spec returns the inert default (the unmodified decisive path).
    """
    raw = "" if spec is None else str(spec)
    stripped = raw.strip()
    if not stripped:
        return RoboTTTAblation(raw=raw)

    freeze = False
    reset_every: int | None = None
    decay: float | None = None
    eta_scale = 1.0
    commit_first = False
    seen: set[str] = set()

    for token_raw in stripped.split(","):
        token = token_raw.strip()
        if not token:
            raise ValueError(f"[robottt-ablation] empty token in {raw!r}; known: {','.join(_KNOWN)}")
        name, sep, arg = token.partition(":")
        name = name.strip().lower()
        arg = arg.strip()
        if name not in _KNOWN:
            raise ValueError(f"[robottt-ablation] unknown token {token!r} in {raw!r}; known: {','.join(_KNOWN)}")
        if name in seen:
            raise ValueError(f"[robottt-ablation] duplicate token {name!r} in {raw!r}")
        seen.add(name)
        if name in ("freeze", "commitfirst"):
            if sep or arg:
                raise ValueError(f"[robottt-ablation] {name} takes no argument, got {token!r}")
            if name == "freeze":
                freeze = True
            else:
                commit_first = True
            continue
        if not sep or not arg:
            raise ValueError(f"[robottt-ablation] {name} requires an argument ({name}:<value>)")
        if name == "reset":
            try:
                reset_every = int(arg)
            except (TypeError, ValueError):
                raise ValueError(f"[robottt-ablation] reset:<int> expected, got {arg!r}") from None
            if reset_every < 1:
                raise ValueError(f"[robottt-ablation] reset:N requires N >= 1, got {reset_every}")
        elif name == "decay":
            decay = _float_arg("decay", arg)
            if not (0.0 < decay < 1.0):
                raise ValueError(f"[robottt-ablation] decay:G requires 0 < G < 1, got {decay}")
        else:  # eta
            eta_scale = _float_arg("eta", arg)
            if not np.isfinite(eta_scale) or eta_scale < 0.0:
                raise ValueError(f"[robottt-ablation] eta:X requires a finite X >= 0, got {eta_scale}")

    # freeze skips the commit entirely, so anything that modifies a commit is contradictory.
    if freeze and (reset_every is not None or decay is not None or eta_scale != 1.0):
        raise ValueError(
            f"[robottt-ablation] freeze cannot be combined with reset/decay/eta (no commit runs "
            f"under freeze); got {raw!r}"
        )

    parts: list[str] = []
    if freeze:
        parts.append("freeze")
    if reset_every is not None:
        parts.append(f"reset:{reset_every}")
    if decay is not None:
        parts.append(f"decay:{decay:g}")
    if eta_scale != 1.0:
        parts.append(f"eta:{eta_scale:g}")
    if commit_first:
        parts.append("commitfirst")
    return RoboTTTAblation(
        spec=",".join(parts),
        raw=raw,
        freeze=freeze,
        reset_every=reset_every,
        decay=decay,
        eta_scale=eta_scale,
        commit_first=commit_first,
    )


def ablation_from_env(env: Mapping[str, str] | None = None) -> RoboTTTAblation:
    """Parse ``ROBOTTT_ABLATION`` once, enforcing the ``ROBOTTT_ABLATION_ACK=smoke`` interlock.

    Any ACTIVE ablation is a smoke-tier arm whose number must never be reported as a sealed result,
    so the server refuses to start without the explicit acknowledgement.
    """
    env = os.environ if env is None else env
    ablation = parse_ablation(env.get(SPEC_ENV, ""))
    if ablation.active and env.get(ACK_ENV, "") != ACK_VALUE:
        raise RuntimeError(
            f"[robottt-ablation] {SPEC_ENV}={ablation.raw!r} is an ablated (smoke-tier) serve path; "
            f"refusing to start without {ACK_ENV}={ACK_VALUE}. Sealed/decisive runs must set NEITHER."
        )
    return ablation


# ------------------------------------------------------------------------------------------------
# pytree helpers (work on JAX pytrees at serve; on plain dict/list/tuple pytrees in tests)
# ------------------------------------------------------------------------------------------------
def _jax():
    try:
        import jax  # noqa: PLC0415

        return jax
    except Exception:  # noqa: BLE001  (jax-free fallback for pure-python fakes)
        return None


def tree_leaves(tree) -> list:
    jax = _jax()
    if jax is not None:
        return list(jax.tree.leaves(tree))
    if isinstance(tree, dict):
        return [leaf for key in sorted(tree) for leaf in tree_leaves(tree[key])]
    if isinstance(tree, (list, tuple)):
        return [leaf for item in tree for leaf in tree_leaves(item)]
    return [tree]


def tree_map2(fn: Callable[[Any, Any], Any], a, b):
    """Elementwise ``fn`` over two structurally identical pytrees."""
    jax = _jax()
    if jax is not None:
        return jax.tree.map(fn, a, b)
    if isinstance(a, dict):
        return {key: tree_map2(fn, a[key], b[key]) for key in a}
    if isinstance(a, (list, tuple)):
        return type(a)(tree_map2(fn, x, y) for x, y in zip(a, b))
    return fn(a, b)


def lerp_tree(base, target, weight: float):
    """base + weight * (target - base), elementwise over the pytree.

    Both the eta rescale (base=entering W, target=committed W) and the decay pull-back
    (base=episode-init W0, target=committed W) are this one operation.
    """
    return tree_map2(lambda t, b: b + weight * (t - b), target, base)


def delta_norm(w, w0) -> float:
    """Global L2 of (W - W_episode_init) over the whole pytree.

    Computed with the leaves' own array ops (device-side under JAX) so exactly ONE scalar crosses
    to the host per call, never the fast-weight tensors themselves.
    """
    leaves_w = tree_leaves(w)
    leaves_0 = tree_leaves(w0)
    if len(leaves_w) != len(leaves_0):
        raise ValueError(f"[robottt-ablation] fast-weight pytree mismatch: {len(leaves_w)} vs {len(leaves_0)} leaves")
    if not leaves_w:
        return 0.0
    jax = _jax()
    if jax is not None:
        import jax.numpy as jnp  # noqa: PLC0415

        total = None
        for a, b in zip(leaves_w, leaves_0):
            part = jnp.sum(jnp.square(jnp.asarray(a) - jnp.asarray(b)))
            total = part if total is None else total + part
        return float(jnp.sqrt(total))
    total = 0.0
    for a, b in zip(leaves_w, leaves_0):
        total += float(np.sum(np.square(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
    return float(np.sqrt(total))


# ------------------------------------------------------------------------------------------------
# the one place the knobs touch W
# ------------------------------------------------------------------------------------------------
def apply_post_commit(ablation: RoboTTTAblation, w_entering, w_committed, w0, commit_idx: int):
    """Post-process one commit under the ablation. Returns ``(w_next, ops_applied)``.

    Order is fixed and load-bearing: eta rescales the STEP (relative to the entering W), decay then
    pulls the result back toward the episode init, and reset is a hard override on its cadence.
    With an inert ablation this returns ``w_committed`` UNCHANGED (same object) — the decisive path
    never touches a leaf.
    """
    ops: list[str] = []
    w_next = w_committed
    if ablation.eta_scale != 1.0:
        w_next = lerp_tree(w_entering, w_next, ablation.eta_scale)
        ops.append(f"eta:{ablation.eta_scale:g}")
    if ablation.decay is not None:
        w_next = lerp_tree(w0, w_next, ablation.decay)
        ops.append(f"decay:{ablation.decay:g}")
    if ablation.reset_every is not None and commit_idx > 0 and commit_idx % ablation.reset_every == 0:
        w_next = w0
        ops.append(f"reset:{ablation.reset_every}")
    return w_next, tuple(ops)


# ------------------------------------------------------------------------------------------------
# A0 probe
# ------------------------------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class ProbeLogger:
    """Append-only JSONL probe. One line per commit and per episode reset, written on the HOST.

    Never called from inside a jitted function: the wrapper calls it after the (already synchronous)
    commit returns, so the only added device traffic is the single `w_delta_norm` scalar.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered: survives a killed server
        self._lock = threading.Lock()
        self.n_lines = 0

    def log(self, kind: str, **fields) -> None:
        record = {"kind": str(kind), "ts": time.time()}
        record.update(fields)
        line = json.dumps(record, default=_json_default, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self.n_lines += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def probe_from_env(env: Mapping[str, str] | None = None) -> ProbeLogger | None:
    """``ROBOTTT_PROBE_LOG=<path>`` -> a ProbeLogger; unset/empty -> None (zero overhead)."""
    env = os.environ if env is None else env
    path = (env.get(PROBE_ENV, "") or "").strip()
    return ProbeLogger(path) if path else None

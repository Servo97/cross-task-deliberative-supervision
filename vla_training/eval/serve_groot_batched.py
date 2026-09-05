#!/usr/bin/env python3
"""BATCHED zmq policy server for GR00T evals (perf-first mandate): ROUTER socket + request gathering.

The stock PolicyServer (gr00t/policy/server_client.py) is a single-thread REP loop — K eval runners
serialize on one GPU forward each. Measured fact: Gr00tSimPolicyWrapper.get_action ACCEPTS batched obs
(stack every flat key along dim 0: video (B,1,H,W,3), state (B,1,D), language list len B) and returns a
batched action dict. So: ROUTER replaces REP (REQ clients — eval_runner_groot.ZmqPolicyClient — work
UNCHANGED: identity frame + empty delimiter + payload), get_action requests are gathered for up to
--gather-ms (since the first queued) or --max-batch, run as ONE wrapped-policy forward, and reply rows
are routed back per zmq identity. ping/reset/kill are answered IMMEDIATELY (never batched).

Modes:
  --mode baseline  Gr00tPolicy + Gr00tSimPolicyWrapper from --model-path (as gr00t/eval/run_gr00t_server.py).
  --mode wsm_cfg   serve_groot_wsm_cfg.py's online-w_t CFG serve, but with PER-CLIENT-IDENTITY conditioner
                   state: each identity owns a WSMEvalConditioner (causal buffer + episode lang); reset
                   clears ONLY that identity. Per gathered batch: ONE batched BackboneTap, each identity's
                   conditioner stepped serially, newest w_t rows stacked -> ah._cfg_eval [B,w_dim]; the
                   head's two-pass CFG runs once batched (cfg_velocities stacks cond+uncond rows on the
                   batch dim — per-sample conditioning is exact, see tests/test_twopass_batched.py).

  python vla_training/eval/serve_groot_batched.py --mode baseline --model-path <ckpt> --port 5555
  python vla_training/eval/serve_groot_batched.py --mode wsm_cfg --finetune-ckpt <ckpt> \
      --encoder-ckpt <wsm_step*.pt> --task-lang-table <task_lang_table.npz> --guidance-scale 1.0

Module import is torch-free (heavy deps enter in main / the install helper) so the batching core is
unit-testable in the sim venv: tests/test_serve_groot_batched.py.
"""

from __future__ import annotations

import argparse
import functools
import time
import traceback

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

_PROPRIO_DIM = {"groot": 1536}


# ---- vendored MsgSerializer subset (gr00t/policy/server_client.py; allow_pickle=False guards kept) ----
def _safe_encode(obj, chain=None):
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        raise TypeError("Refusing to encode object-dtype ndarray (pickle surface).")
    return mnp.encode(obj, chain=chain)


def _safe_decode(obj, chain=None):
    if isinstance(obj, dict):
        nd_val = obj.get(b"nd", obj.get("nd"))
        kind_val = obj.get(b"kind", obj.get("kind"))
        if nd_val and kind_val in (b"O", "O"):
            raise ValueError("Refusing to decode object-dtype ndarray payload (pickle-bearing).")
    return mnp.decode(obj, chain=chain)


def _to_bytes(data):
    return msgpack.packb(data, default=functools.partial(_safe_encode, chain=lambda o: o))


def _from_bytes(data):
    return msgpack.unpackb(data, object_hook=functools.partial(_safe_decode, chain=lambda o: o), raw=False)


# --------------------------------------------------------------------------------------------------
# Batch plumbing (pure numpy; unit-tested).
# --------------------------------------------------------------------------------------------------
def _obs_signature(obs) -> dict:
    """Flat batch-1 sim obs -> {key: (kind, shape, dtype)} signature. The eval-client contract
    (eval_runner_groot.pack_obs): video.<cam> uint8 (1,1,H,W,3), state.<k> float32 (1,1,D), lang [str].
    Raises ValueError on anything that cannot ride a stacked batch (per-request error, not batch-fatal)."""
    if not isinstance(obs, dict) or not obs:
        raise ValueError("observation must be a non-empty dict")
    sig = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            if v.ndim < 1 or v.shape[0] != 1:
                raise ValueError(f"{k}: expected leading batch dim 1, got shape {getattr(v, 'shape', None)}")
            sig[k] = ("nd", tuple(v.shape), str(v.dtype))
        elif isinstance(v, (list, tuple)):
            if len(v) != 1 or not isinstance(v[0], str):
                raise ValueError(f"{k}: expected a length-1 list of str, got {type(v).__name__} len {len(v)}")
            sig[k] = ("strs", 1)
        else:
            raise ValueError(f"{k}: unsupported obs value type {type(v).__name__}")
    return sig


def _stack_rows(obs_list: list[dict]) -> dict:
    """Stack signature-matched batch-1 obs along dim 0: ndarray -> concat axis 0, str-list -> concat."""
    out = {}
    for k in obs_list[0]:
        vals = [o[k] for o in obs_list]
        out[k] = np.concatenate(vals, axis=0) if isinstance(vals[0], np.ndarray) else [s for v in vals for s in v]
    return out


def _split_row(reply: dict, i: int, b: int) -> dict:
    """Row i of a batched reply dict, KEEPING the leading batch dim (client contract: (1,T,D) or (T,D)).
    Values without a length-B leading dim (scalars, strings, shared metadata) pass through unchanged."""
    if not isinstance(reply, dict):
        return reply
    return {
        k: (v[i : i + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == b else v)
        for k, v in reply.items()
    }


class BatchedPolicyServer:
    """ROUTER-socket batched drop-in for PolicyServer. REQ clients work unchanged against ROUTER
    (identity frame + empty delimiter + payload). ping/reset/kill/get_modality_config are answered
    IMMEDIATELY; only get_action enters the batch. Gather rule: block on the first message, drain
    non-blocking, then keep gathering until gather_ms since the FIRST queued get_action OR max_batch.

    Hooks (all optional; the wsm_cfg mode wires them to WSMCfgIdentityStates):
      reset_hook(identity, options) -> reply dict     (default: policy.reset(options=...))
      validate_hook(identity, observation, options)   raise -> error reply to THAT identity only
      batch_hook(identities)                          called just before the ONE batched forward
    """

    def __init__(
        self,
        policy,
        host: str = "*",
        port: int = 5555,
        gather_ms: float = 20.0,
        max_batch: int = 12,
        reset_hook=None,
        validate_hook=None,
        batch_hook=None,
        duty_cycle: float = 1.0,
    ):
        assert max_batch >= 1 and gather_ms >= 0, f"bad gather config: {gather_ms=} {max_batch=}"
        self.policy = policy
        self.gather_s = float(gather_ms) / 1000.0
        self.max_batch = int(max_batch)
        self.duty_cycle = float(duty_cycle)
        self.reset_hook = reset_hook
        self.validate_hook = validate_hook
        self.batch_hook = batch_hook
        self.running = True
        self.batch_sizes: list[int] = []  # per-flush sizes (diagnostics/tests)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        if int(port) == 0:  # tests: ephemeral port
            self.port = self.socket.bind_to_random_port(f"tcp://{host}")
        else:
            self.socket.bind(f"tcp://{host}:{port}")
            self.port = int(port)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def _send(self, ident: bytes, payload) -> None:
        try:
            self.socket.send_multipart([ident, b"", _to_bytes(payload)])
        except Exception as e:  # dead identity etc. — never kill the loop
            print(f"[serve-batched] send to {ident!r} failed: {e}", flush=True)

    def _dispatch(self, frames):
        """One ROUTER message. get_action -> (identity, observation, options) for the batch; everything
        else (ping/reset/kill/...) is replied IMMEDIATELY here and returns None."""
        if len(frames) != 3 or frames[1] != b"":
            self._send(frames[0] if frames else b"", {"error": f"malformed envelope ({len(frames)} frames)"})
            return None
        ident, payload = frames[0], frames[2]
        try:
            request = _from_bytes(payload)
            endpoint = request.get("endpoint", "get_action")
            data = request.get("data") or {}
        except Exception as e:
            self._send(ident, {"error": f"bad request: {e}"})
            return None
        if endpoint == "get_action":
            return ident, data.get("observation"), data.get("options")
        try:
            if endpoint == "ping":
                self._send(ident, {"status": "ok", "message": "Server is running"})
            elif endpoint == "reset":
                options = data.get("options")
                if self.reset_hook is not None:
                    self._send(ident, self.reset_hook(ident, options) or {"status": "ok"})
                else:
                    self._send(ident, self.policy.reset(options=options) or {"status": "ok"})
            elif endpoint == "kill":
                self.running = False
                self._send(ident, {"status": "ok"})
            elif endpoint == "get_modality_config":
                self._send(ident, getattr(self.policy, "get_modality_config", lambda: {})())
            else:
                self._send(ident, {"error": f"Unknown endpoint: {endpoint}"})
        except Exception as e:
            traceback.print_exc()
            self._send(ident, {"error": str(e)})
        return None

    def _flush(self, batch) -> None:
        """Validate + stack the gathered requests, ONE wrapped-policy forward, route reply rows.
        Per-request malformed input -> error reply to that identity ONLY; the rest still run."""
        good, ref = [], None
        for ident, obs, options in batch:
            try:
                sig = _obs_signature(obs)
                if self.validate_hook is not None:
                    self.validate_hook(ident, obs, options)
            except Exception as e:
                self._send(ident, {"error": f"malformed request: {e}"})
                continue
            if ref is None:
                ref = sig
            elif sig != ref:
                self._send(
                    ident,
                    {
                        "error": "observation signature mismatch vs batch "
                        f"(per-key shapes/dtypes must match): {sig} != {ref}"
                    },
                )
                continue
            good.append((ident, obs, options))
        if not good:
            return
        # per-request options cannot ride ONE batched forward — require equality (eval clients send None)
        opts0, kept = good[0][2], []
        for ident, obs, options in good:
            try:
                same = bool(options == opts0)
            except Exception:
                same = False
            if same:
                kept.append((ident, obs, options))
            else:
                self._send(ident, {"error": f"options differ within a batch (unsupported): {options!r} != {opts0!r}"})
        if not kept:
            return
        idents = [k[0] for k in kept]
        stacked = _stack_rows([k[1] for k in kept])
        b = len(kept)
        self.batch_sizes.append(b)
        try:
            if self.batch_hook is not None:
                self.batch_hook(idents)
            t_fwd = time.monotonic()
            action, info = self.policy.get_action(stacked, options=opts0)
            if self.duty_cycle < 1.0:  # GPU duty cap: leave render gaps for the K envs
                try:  # (H100 compute scheduler starves EGL renders behind
                    import torch  # back-to-back forwards; observed 10x env-step

                    if torch.cuda.is_available():  # inflation at K=8, 2026-07-18)
                        torch.cuda.synchronize()
                except Exception:
                    pass
                dt = time.monotonic() - t_fwd
                time.sleep(dt * (1.0 / self.duty_cycle - 1.0))
        except Exception as e:
            traceback.print_exc()
            for ident in idents:
                self._send(ident, {"error": str(e)})
            return
        for i, ident in enumerate(idents):  # (action, info) tuple == wire list, as REP did
            self._send(ident, [_split_row(action, i, b), _split_row(info, i, b)])

    def run(self) -> None:
        print(
            f"[serve-batched] listening on port {self.port} "
            f"(gather_ms={self.gather_s * 1e3:.0f}, max_batch={self.max_batch})",
            flush=True,
        )
        pending, first_ts = [], 0.0
        while self.running:
            if pending:
                budget_ms = (first_ts + self.gather_s - time.monotonic()) * 1000.0
                if len(pending) >= self.max_batch or budget_ms <= 0 or not self.socket.poll(int(budget_ms)):
                    self._flush(pending)
                    pending = []
                    continue
            else:
                self.socket.poll()  # block until the first message
            try:
                while len(pending) < self.max_batch:  # non-blocking drain of everything ready
                    item = self._dispatch(self.socket.recv_multipart(zmq.NOBLOCK))
                    if item is None:
                        continue
                    if any(item[0] == p[0] for p in pending):  # 2nd in-flight from one identity: impossible
                        self._flush(pending)  # for REQ clients; keep replies 1:1 anyway
                        pending = []
                    if not pending:
                        first_ts = time.monotonic()
                    pending.append(item)
            except zmq.Again:
                pass
            if not self.running:
                break
        if pending:  # kill mid-gather: don't strand clients
            self._flush(pending)


# --------------------------------------------------------------------------------------------------
# wsm_cfg mode: per-CLIENT-IDENTITY online-w_t state (torch-free class; heavy deps via the factory).
# --------------------------------------------------------------------------------------------------
class WSMCfgIdentityStates:
    """serve_groot_wsm_cfg's per-step conditioner state, keyed by zmq identity: each client owns a
    WSMEvalConditioner (causal buffer + episode lang). reset(options) clears ONLY that identity;
    conditioner_rows steps each pending identity SERIALLY (each holds its own causal prefix) and returns
    the newest w_t per batch row. Factory-injected so it is unit-testable with fakes (no torch)."""

    def __init__(self, conditioner_factory, task_lang_table: dict):
        self._factory = conditioner_factory
        self._table = task_lang_table
        self._conds: dict[bytes, object] = {}
        self._pending: list[bytes] = []

    def reset(self, ident: bytes, options=None) -> dict:
        task = (options or {}).get("task") if isinstance(options, dict) else None
        if task is None or task not in self._table:
            raise RuntimeError(
                f"[serve-batched] reset needs a known task; got {task!r} (table has {len(self._table)} tasks)"
            )
        cond = self._conds.get(ident)
        if cond is None:
            cond = self._conds[ident] = self._factory()
        cond.reset(self._table[task])
        return {"status": "ok"}

    def validate(self, ident: bytes, observation=None, options=None) -> None:
        """Server validate_hook: fail THIS request (not the batch) if the identity never reset(task)."""
        cond = self._conds.get(ident)
        if cond is None or getattr(cond, "_lang", "set") is None:
            raise RuntimeError(f"get_action before reset(task) for identity {ident!r} — no episode state")

    def on_batch(self, idents: list[bytes]) -> None:
        """Server batch_hook: the ordered identities of the forward about to run (read by the tap patch)."""
        self._pending = list(idents)

    def conditioner_rows(self, patches, proprios) -> list:
        """One (patch, proprio) row per pending identity, batch order -> [newest w_t per row]."""
        assert len(self._pending) == len(patches) == len(proprios), (
            f"[serve-batched] batch/identity mismatch: {len(self._pending)} idents, {len(patches)} rows"
        )
        rows = []
        for ident, patch, proprio in zip(self._pending, patches, proprios):
            cond = self._conds.get(ident)
            if cond is None:  # validate_hook screens this; provenance guard
                raise RuntimeError(f"[serve-batched] no conditioner for identity {ident!r} (reset first)")
            w_window, _lang = cond.step(patch, proprio)  # finite-w guarded inside WSMEvalConditioner
            rows.append(w_window[-1])  # newest causal w_t
        return rows


def install_wsm_cfg_batched(policy, states: WSMCfgIdentityStates) -> None:
    """BATCHED counterpart of _groot_wsm_cfg_eval.install_wsm_cfg_eval: patch the INNER Gr00tPolicy's
    get_action (the sim wrapper converts flat->nested first, which is what BackboneTap needs) to do ONE
    batched tap, step each identity's conditioner serially, and stash the stacked newest-w_t [B,w_dim] on
    ah._cfg_eval. The head's two-pass CFG then runs ONCE batched: cfg_velocities cats the [B,dim]
    cond/uncond rows along the batch dim, so per-sample conditioning is exact in a single forward
    (validated by tests/test_twopass_batched.py + WSM_VERIFY_2PASS in production)."""
    import torch

    from workspace_models.features.backbone_tap import BackboneTap

    inner = getattr(policy, "policy", policy)  # unwrap Gr00tSimPolicyWrapper if present
    tap = BackboneTap(inner)
    ah = inner.model.action_head
    cdt = next(ah.wsm_cfg.parameters()).dtype
    _orig_get_action = inner.get_action

    def get_action(observation, *args, **kwargs):
        r = tap.tap(observation)  # batched TapResult
        b = r.patch_tokens.shape[0]
        patches = [r.patch_tokens[i].float().cpu().numpy() for i in range(b)]  # [P, backbone_dim] each
        proprios = [r.state_emb[i, 0].float().cpu().numpy() for i in range(b)]  # [Dp] each
        w_t = torch.stack(states.conditioner_rows(patches, proprios)).to(dtype=cdt)  # [B, w_dim]
        ah._cfg_eval = (w_t, None)  # POC: no w_next (self-conditioning)
        try:
            return _orig_get_action(observation, *args, **kwargs)
        finally:
            ah._cfg_eval = (None, None)  # never leak a stale batch-B stash

    inner.get_action = get_action
    print(
        "[serve-batched] wsm_cfg installed: per-identity conditioners, batched tap + batched two-pass CFG", flush=True
    )


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "wsm_cfg"], default="baseline")
    ap.add_argument("--model-path", help="baseline: HF ckpt dir (as gr00t/eval/run_gr00t_server.py)")
    # wsm_cfg flags (mirror serve_groot_wsm_cfg.py)
    ap.add_argument("--finetune-ckpt", help="wsm_cfg: CFG finetune model dir (HF checkpoint-XXXX)")
    ap.add_argument("--encoder-ckpt", help="wsm_cfg: FROZEN WorkspaceModel ckpt (wsm_step*.pt)")
    ap.add_argument("--task-lang-table", help="wsm_cfg: task_lang_table.npz (next to the precompute _meta.json)")
    ap.add_argument("--guidance-scale", type=float, default=0.0, help="CFG scale s (0=baseline, 1=cond)")
    ap.add_argument("--w-dim", type=int, default=512)
    ap.add_argument("--p-drop", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=8, help="cache grid stride (== eval exec_steps)")
    # server flags
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--gather-ms", type=float, default=20.0, help="batch window since the first queued get_action")
    ap.add_argument("--max-batch", type=int, default=12, help="flush the batch at this many requests")
    ap.add_argument(
        "--duty-cycle",
        type=float,
        default=1.0,
        help="<1.0: cap the server's GPU-busy fraction (post-forward sleep) so co-resident EGL env renders never starve (H100)",
    )
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--no-sim-wrapper", action="store_true")
    args = ap.parse_args()

    import json
    import os
    from pathlib import Path

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    hooks: dict = {}
    if args.mode == "baseline":
        assert args.model_path, "--model-path required for --mode baseline"
        if args.model_path.startswith("/") and not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
            model_path=args.model_path,
            device=args.device,
            strict=not args.no_strict,
        )
        server_policy = policy if args.no_sim_wrapper else Gr00tSimPolicyWrapper(policy)
    else:  # wsm_cfg — serve_groot_wsm_cfg per identity
        for f in ("finetune_ckpt", "encoder_ckpt", "task_lang_table"):
            assert getattr(args, f), f"--{f.replace('_', '-')} required for --mode wsm_cfg"

        from vla_training.eval._groot_wsm_cfg_eval import load_task_lang_table, restore_cfg_weights
        from vla_training.eval._groot_wsm_eval import WSMEvalConditioner
        from vla_training.train.train_base._groot_wsm_cfg_common import attach_wsm_cfg
        from workspace_models.features.generate_policy_features import load_wsm

        print(f"[serve-batched] loading finetune ckpt {args.finetune_ckpt}", flush=True)
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
            model_path=args.finetune_ckpt,
            device=args.device,
            strict=not args.no_strict,
        )
        # re-attach conditioner + temb wrapper (from_pretrained bypasses the train-path patch), set s, restore
        attach_wsm_cfg(
            policy.model,
            w_dim=args.w_dim,
            p_drop=args.p_drop,
            with_future=False,
            diag_every=1,
            guidance_scale=args.guidance_scale,
        )
        n = restore_cfg_weights(policy.model, args.finetune_ckpt)
        assert n > 0, "no conditioner weights restored — refusing to serve an untrained (≈baseline) conditioner"

        encoder, _meta = load_wsm(args.encoder_ckpt, args.device, proprio_dim=_PROPRIO_DIM["groot"])
        # encoder-provenance guard (same as serve_groot_wsm_cfg): served encoder must match the precompute
        # the conditioner trained on. Finite-w is guarded per step inside WSMEvalConditioner.
        meta_path = Path(args.task_lang_table).expanduser().parent / "_meta.json"
        if meta_path.exists():
            pstep = json.loads(meta_path.read_text()).get("wsm_step")
            estep = _meta.get("step")
            if estep is not None and pstep is not None and int(estep) != int(pstep):
                raise RuntimeError(
                    f"[serve-batched] ENCODER/PRECOMPUTE MISMATCH: encoder step {estep} != "
                    f"precompute wsm_step {pstep} ({meta_path})."
                )
            print(f"[serve-batched] encoder provenance OK: step {estep} == precompute {pstep}", flush=True)
        else:
            print(
                f"[serve-batched] WARNING: no _meta.json at {meta_path}; cannot verify encoder provenance.", flush=True
            )

        table = load_task_lang_table(args.task_lang_table)
        states = WSMCfgIdentityStates(
            lambda: WSMEvalConditioner(encoder, k_window=1, stride=args.stride, device=args.device), table
        )
        server_policy = policy if args.no_sim_wrapper else Gr00tSimPolicyWrapper(policy)
        install_wsm_cfg_batched(server_policy, states)
        hooks = dict(reset_hook=states.reset, validate_hook=states.validate, batch_hook=states.on_batch)
        print(
            f"[serve-batched] wsm_cfg ready: s={args.guidance_scale}, {len(table)} tasks, encoder={args.encoder_ckpt}",
            flush=True,
        )

    BatchedPolicyServer(
        server_policy,
        host=args.host,
        port=args.port,
        gather_ms=args.gather_ms,
        duty_cycle=args.duty_cycle,
        max_batch=args.max_batch,
        **hooks,
    ).run()


if __name__ == "__main__":
    main()

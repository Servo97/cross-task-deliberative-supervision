"""Compute-once data-balancing math; apply-twice in each backbone's native format.

The plan's balancing-ON pretrain arm targets an explicit per-source-group mix in each batch
(e.g. 1/3 mg_atomic, 1/3 human_atomic, 1/3 human_composite). This module turns a partitioned
soup + the YAML ``data_balancing`` knobs into normalized per-GROUP masses, then derives:

  * pi0.5 (openpi)  -> a per-dataset ``dataset_weights`` list (passed to LeRobotRobocasaDataConfig);
  * GR00T (PyTorch) -> one ``GrootGroupSpec`` (dirs, mix_ratio) per group (one SingleDatasetConfig each).

Both realize the same per-GROUP mass; they differ only in the within-group split (pi0.5 = equal per
member dataset because its mixture uses weights as-given with balance_dataset_weights=False; GR00T =
proportional to step count via factory's relative_length*mix_ratio). That asymmetry is documented and
sub-leading to the group-level mix the experiment controls.

IMPORTANT caveats (surfaced by the design review):
  * ``strategy: per_batch_resample`` / ``max_mg_atomic_fraction`` are realized only IN EXPECTATION by
    static weights — neither stack enforces a hard per-batch cap. A 256-batch lands at the target
    fractions ± binomial noise (std ~ sqrt(p(1-p)/256) ≈ 3% for p=1/3).
  * The pi0.5 ON-arm additionally requires a 2-line robocasa_openpi fork fix (the ``dataset_weights``
    branch currently NameErrors); the adapter must assert that patch is present before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.config_schema import DataBalancingView, DataView
from utils.soup import (
    GROUPS,
    REMEMBENCH13_SOUP,
    combined_target_soup,
    dirs_for_group,
    partition_by_group,
    remembench_soup,
    resolve_soup,
)


@dataclass(frozen=True)
class GrootGroupSpec:
    """Framework-neutral precursor of a GR00T SingleDatasetConfig (no gr00t import)."""

    group: str
    dirs: list[str]
    mix_ratio: float


def compute_group_masses(groups: dict[str, list], balancing: DataBalancingView) -> dict[str, float]:
    """Per-source-group expected mass, normalized to sum 1 over PRESENT (non-empty) groups.

    OFF (``enabled=False``) -> {} : signal each backbone to use its NATIVE default mixing
    (pi0.5 size power-law / GR00T size-proportional single spec).
    ON -> base weights from the per-source ``*_weight`` knobs, normalized; then the
    ``max_mg_atomic_fraction`` cap clamps mg_atomic's mass and redistributes the excess to the
    human groups in proportion to their current mass. (Equal 1/3-thirds is simply
    weights=1/1/1 with cap>=0.5.)
    """
    present = [g for g in GROUPS if groups.get(g)]
    if not balancing.enabled or not present:
        return {}

    base = {
        "mg_atomic": balancing.mg_atomic_weight,
        "human_atomic": balancing.human_atomic_weight,
        "human_composite": balancing.human_composite_weight,
    }
    w = {g: float(base[g]) for g in present}
    total = sum(w.values())
    if total <= 0:
        raise ValueError(f"non-positive total balancing weight for groups {present}")
    masses = {g: w[g] / total for g in present}

    cap = balancing.max_mg_atomic_fraction
    if "mg_atomic" in masses and masses["mg_atomic"] > cap:
        excess = masses["mg_atomic"] - cap
        masses["mg_atomic"] = cap
        humans = [g for g in masses if g != "mg_atomic"]
        hsum = sum(masses[g] for g in humans)
        for g in humans:
            share = (masses[g] / hsum) if hsum > 0 else 1.0 / len(humans)
            masses[g] += excess * share
    # Renormalize to sum 1. Handles the mg-only-group + cap case (the capped excess has no human
    # group to absorb it, which would otherwise leave masses summing to <1 and de-normalize the mix).
    s = sum(masses.values())
    if s <= 0:
        raise ValueError(f"non-positive total mass after balancing: {masses}")
    return {g: m / s for g, m in masses.items()}


def pi05_dataset_weights(soup: list[dict], group_masses: dict[str, float]) -> list[float] | None:
    """Per-dataset weights aligned to soup order, for openpi's LeRobotRobocasaDataConfig.

    Returns None when balancing is OFF (group_masses == {}), so the adapter passes
    ``dataset_weights=None`` and openpi's native size power-law / single-ds path runs.

    Each dataset gets raw weight = mass_of_its_group / (#datasets in that group) — equal within
    a group, since openpi builds the mixture with balance_dataset_weights=False (weights used
    as-given, NOT length-multiplied), so per-step probability == normalized weight and each
    group's total probability == its mass. The whole vector is divided by its max so the max
    becomes exactly 1.0 (LeRobotMixtureDataset requires >=1 dataset with weight==1.0 as the
    'primary'; dividing by max is float-exact and order-independent).
    """
    if not group_masses:
        return None
    from utils.soup import source_group_of

    member_groups = [source_group_of(m) for m in soup]
    counts: dict[str, int] = {}
    for g in member_groups:
        counts[g] = counts.get(g, 0) + 1
    raw = np.array([group_masses.get(g, 0.0) / counts[g] for g in member_groups], dtype=np.float64)
    mx = raw.max()
    if mx <= 0:
        raise ValueError("all-zero pi05 dataset weights")
    return (raw / mx).tolist()


def groot_group_specs(groups: dict[str, list], group_masses: dict[str, float]) -> list[GrootGroupSpec]:
    """One spec per present group with mix_ratio == group mass. When balancing is OFF, returns a
    single spec over ALL dirs with mix_ratio 1.0 (GR00T's native size-proportional mixing)."""
    present = [(g, groups[g]) for g in GROUPS if groups.get(g)]
    if not group_masses:
        all_dirs = [d for _, metas in present for d in dirs_for_group(metas)]
        return [GrootGroupSpec(group="all", dirs=all_dirs, mix_ratio=1.0)]
    return [
        GrootGroupSpec(group=g, dirs=dirs_for_group(metas), mix_ratio=group_masses[g])
        for g, metas in present
        if group_masses.get(g, 0.0) > 0
    ]


@dataclass(frozen=True)
class GroupedSoup:
    """The single in-memory DataSpec crossing the framework boundary. Built ONCE, consumed by
    either backbone adapter. ``soup`` order is load-bearing: ``pi05_weights[i]`` aligns to
    ``soup[i]`` (== the data_dirs order openpi receives)."""

    soup: list[dict]
    groups: dict[str, list]
    group_masses: dict[str, float]
    pi05_weights: list[float] | None
    groot_specs: list[GrootGroupSpec]
    balancing_enabled: bool
    source: str

    @classmethod
    def from_soup(cls, soup: list[dict], balancing: DataBalancingView, *, source: str = "all") -> "GroupedSoup":
        """Core constructor from an explicit soup + balancing knobs."""
        groups = partition_by_group(soup)
        masses = compute_group_masses(groups, balancing)
        return cls(
            soup=soup,
            groups=groups,
            group_masses=masses,
            pi05_weights=pi05_dataset_weights(soup, masses),
            groot_specs=groot_group_specs(groups, masses),
            balancing_enabled=balancing.enabled,
            source=source,
        )

    @classmethod
    def from_data_view(cls, data: DataView) -> "GroupedSoup":
        """Resolve the soup from the YAML data block, then build. If a target subsample fraction
        is set (finetune), build the COMBINED target soup (atomic_seen+composite_seen+composite_unseen)
        at that fraction; otherwise resolve the named pretrain soup.

        ``soup: remembench13`` takes an explicit, earlier branch: it is a robocasa-free glob over the
        FLAT ReMemBench layout at NATIVE full mass, so it must never fall into either the target50
        subsample path or the registry lookup. Every other soup name reaches exactly the code it
        reached before this branch existed.
        """
        if data.soup == REMEMBENCH13_SOUP:
            if data.subsample_fraction is not None:
                raise ValueError(
                    f"soup={REMEMBENCH13_SOUP!r} trains on every demo; remove data.subsample "
                    f"(got target_fraction={data.subsample_fraction})"
                )
            soup = remembench_soup()
        elif data.subsample_fraction is not None:
            if data.soup != "target50":
                raise ValueError(f"data.subsample is only defined for soup=target50, got {data.soup!r}")
            soup = combined_target_soup(data.subsample_fraction)
        else:
            soup = resolve_soup(name=data.soup)
        return cls.from_soup(soup, data.balancing, source=data.source)

    def summary(self) -> str:
        lines = [
            f"GroupedSoup: {len(self.soup)} datasets, source={self.source}, "
            f"balancing={'ON' if self.balancing_enabled else 'OFF (native default)'}"
        ]
        for g in GROUPS:
            n = len(self.groups.get(g, []))
            if n:
                mass = self.group_masses.get(g)
                mtxt = f"{mass:.3f}" if mass is not None else "native"
                lines.append(f"  {g:<16} {n:>4} datasets   expected_mass={mtxt}")
        if self.pi05_weights is not None:
            w = np.asarray(self.pi05_weights)
            lines.append(
                f"  pi05_weights: min={w.min():.4f} max={w.max():.4f} (primary==1.0: {bool(np.isclose(w.max(), 1.0))})"
            )
        if self.groot_specs:
            lines.append(
                "  groot_specs: " + ", ".join(f"{s.group}:{s.mix_ratio:.3f}({len(s.dirs)})" for s in self.groot_specs)
            )
        return "\n".join(lines)

"""Single home for the identity, AWS-infrastructure, and machine-path defaults of this repository.

Every value is read from an environment variable named ``WSM_<FIELD>`` and falls back to a documented
default that reproduces the literal the code carried before centralization, so a process with none of
these variables set behaves byte-identically to the pre-settings code. Empty values count as unset.

Two owner notions are deliberately distinct and must not be conflated:

``STUDY_OWNER``
    The S3 storage prefix, ``s3://<RESULTS_BUCKET>/<STUDY_OWNER>/wsm_robocasa``. Every content address
    of the study (run ids, manifests, archives) is minted under it, so it never moves — even when the
    submitting identity changes.
``OWNER_EMAIL``
    The ``tri.owner.email`` SCP tag of the submitting identity (org SCP ``p-ahpdy5vv`` denies
    ``batch:SubmitServiceJob`` unless it and ``tri.project`` are present and non-empty).

Launchers import these through ``scripts/launch/launch_guardrails.py`` (their single import point);
everything else imports this module directly — the repository root is on ``sys.path`` wherever the
``vla_training`` / ``workspace_models`` / ``robomme_integration`` packages are importable. Modules that
run on a cluster node from an isolated bundle that excludes the repository root (the
``robomme_integration/`` package bundle) keep their literals and are listed in the README.

Run ``python wsm_settings.py`` to print the resolved values and where each one came from.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path

ENV_PREFIX = "WSM_"

#: Field -> (environment variable, default description). The description is documentation only; the
#: defaults themselves are computed in :func:`resolve` so that dependent defaults follow overrides.
ENV_VARS: dict[str, tuple[str, str]] = {
    "execution_account": ("WSM_EXECUTION_ACCOUNT", "141701954645 (cam-robotics SageMaker Batch account)"),
    "storage_account": ("WSM_STORAGE_ACCOUNT", "141701954645 (study storage == execution since 2026-07-22)"),
    "legacy_account": ("WSM_LEGACY_ACCOUNT", "124224456861 (original storage account; historical paths only)"),
    "region": ("WSM_REGION", "us-west-2"),
    "results_bucket": ("WSM_RESULTS_BUCKET", "sagemaker-<region>-<storage_account>"),
    "study_owner": ("WSM_STUDY_OWNER", "sarvesh.patil (S3 storage prefix; content addresses live under it)"),
    "owner_email": ("WSM_OWNER_EMAIL", "sarvesh.patil.pi@tri.global (tri.owner.email SCP tag)"),
    "project_tag": ("WSM_PROJECT_TAG", "LONG-CONTEXT-VLA (tri.project SCP tag)"),
    "image_owner": ("WSM_IMAGE_OWNER", "sarvesh.patil (ECR repository owner prefix, <owner>-groot-dexjoco)"),
    "research_root": ("WSM_RESEARCH_ROOT", "~/Research"),
    "tri_root": ("WSM_TRI_ROOT", "<research_root>/TRI"),
    "wsm_data_root": ("WSM_DATA_ROOT", "<tri_root>/wsm_data (local hot set: checkpoints, features, labels)"),
    "internal_training_root": ("WSM_INTERNAL_TRAINING_ROOT", "<tri_root>/internal_training"),
    "robomme_eval_root": ("WSM_ROBOMME_EVAL_ROOT", "<tri_root>/robomme_eval"),
    "robocasa_openpi_root": ("WSM_ROBOCASA_OPENPI_ROOT", "<research_root>/robocasa_openpi (the openpi fork)"),
    "robocasa_root": ("WSM_ROBOCASA_ROOT", "<research_root>/robocasa"),
    "robosuite_root": ("WSM_ROBOSUITE_ROOT", "<research_root>/robosuite"),
    "envs_root": ("WSM_ENVS_ROOT", "<research_root>/envs (uv virtualenvs: sm_launch, ...)"),
}


@dataclasses.dataclass(frozen=True)
class Settings:
    """Resolved settings. Construct with :func:`resolve`; use the module constants for the defaults."""

    execution_account: str
    storage_account: str
    legacy_account: str
    region: str
    results_bucket: str
    study_owner: str
    owner_email: str
    project_tag: str
    image_owner: str
    research_root: Path
    tri_root: Path
    wsm_data_root: Path
    internal_training_root: Path
    robomme_eval_root: Path
    robocasa_openpi_root: Path
    robocasa_root: Path
    robosuite_root: Path
    envs_root: Path
    #: Names of the fields that were taken from the environment rather than defaulted.
    from_env: tuple[str, ...] = ()

    # ---- derived values (never overridable on their own: they follow their inputs) ----
    @property
    def legacy_results_bucket(self) -> str:
        return f"sagemaker-{self.region}-{self.legacy_account}"

    @property
    def ecr_registry(self) -> str:
        return f"{self.execution_account}.dkr.ecr.{self.region}.amazonaws.com"

    @property
    def dexjoco_image_repo(self) -> str:
        """The shared thin DexJoCo runtime image repository (always pinned by digest at the call site)."""
        return f"{self.ecr_registry}/{self.image_owner}-groot-dexjoco"

    @property
    def wsm_robocasa_s3(self) -> str:
        """The study family's storage root: ``s3://<bucket>/<owner>/wsm_robocasa``."""
        return f"s3://{self.results_bucket}/{self.study_owner}/wsm_robocasa"

    @property
    def long_context_study_s3(self) -> str:
        return f"{self.wsm_robocasa_s3}/studies/long_context_v1"

    @property
    def robocasa_openpi_src(self) -> Path:
        return self.robocasa_openpi_root / "src"


def resolve(env: Mapping[str, str] | None = None, *, home: Path | None = None) -> Settings:
    """Resolve settings from ``env`` (default ``os.environ``) anchored at ``home`` (default ``Path.home()``)."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else Path(home)
    from_env: list[str] = []

    def text(field: str, default: str) -> str:
        value = env.get(ENV_VARS[field][0], "").strip()
        if value:
            from_env.append(field)
            return value
        return default

    def path(field: str, default: Path) -> Path:
        value = env.get(ENV_VARS[field][0], "").strip()
        if value:
            from_env.append(field)
            return Path(value).expanduser()
        return default

    execution_account = text("execution_account", "141701954645")
    storage_account = text("storage_account", "141701954645")
    legacy_account = text("legacy_account", "124224456861")
    region = text("region", "us-west-2")
    results_bucket = text("results_bucket", f"sagemaker-{region}-{storage_account}")
    research_root = path("research_root", home / "Research")
    tri_root = path("tri_root", research_root / "TRI")
    return Settings(
        execution_account=execution_account,
        storage_account=storage_account,
        legacy_account=legacy_account,
        region=region,
        results_bucket=results_bucket,
        study_owner=text("study_owner", "sarvesh.patil"),
        owner_email=text("owner_email", "sarvesh.patil.pi@tri.global"),
        project_tag=text("project_tag", "LONG-CONTEXT-VLA"),
        image_owner=text("image_owner", "sarvesh.patil"),
        research_root=research_root,
        tri_root=tri_root,
        wsm_data_root=path("wsm_data_root", tri_root / "wsm_data"),
        internal_training_root=path("internal_training_root", tri_root / "internal_training"),
        robomme_eval_root=path("robomme_eval_root", tri_root / "robomme_eval"),
        robocasa_openpi_root=path("robocasa_openpi_root", research_root / "robocasa_openpi"),
        robocasa_root=path("robocasa_root", research_root / "robocasa"),
        robosuite_root=path("robosuite_root", research_root / "robosuite"),
        envs_root=path("envs_root", research_root / "envs"),
        from_env=tuple(from_env),
    )


def describe(settings: Settings | None = None) -> str:
    """Human-readable table of every field: resolved value, its environment variable, and its origin."""
    settings = SETTINGS if settings is None else settings
    rows = []
    for field in dataclasses.fields(Settings):
        if field.name == "from_env":
            continue
        variable, _ = ENV_VARS[field.name]
        origin = "env" if field.name in settings.from_env else "default"
        rows.append((field.name, str(getattr(settings, field.name)), variable, origin))
    derived = (
        ("legacy_results_bucket", settings.legacy_results_bucket),
        ("ecr_registry", settings.ecr_registry),
        ("dexjoco_image_repo", settings.dexjoco_image_repo),
        ("wsm_robocasa_s3", settings.wsm_robocasa_s3),
        ("long_context_study_s3", settings.long_context_study_s3),
        ("robocasa_openpi_src", str(settings.robocasa_openpi_src)),
    )
    width = max(len(name) for name, *_ in rows + list(derived))
    lines = [f"{name:<{width}}  {value}  [{variable}: {origin}]" for name, value, variable, origin in rows]
    lines.append("derived:")
    lines.extend(f"{name:<{width}}  {value}" for name, value in derived)
    return "\n".join(lines)


SETTINGS = resolve()

EXECUTION_ACCOUNT = SETTINGS.execution_account
STORAGE_ACCOUNT = SETTINGS.storage_account
LEGACY_ACCOUNT = SETTINGS.legacy_account
REGION = SETTINGS.region
RESULTS_BUCKET = SETTINGS.results_bucket
LEGACY_RESULTS_BUCKET = SETTINGS.legacy_results_bucket
STUDY_OWNER = SETTINGS.study_owner
OWNER_EMAIL = SETTINGS.owner_email
PROJECT_TAG = SETTINGS.project_tag
IMAGE_OWNER = SETTINGS.image_owner
ECR_REGISTRY = SETTINGS.ecr_registry
DEXJOCO_IMAGE_REPO = SETTINGS.dexjoco_image_repo
WSM_ROBOCASA_S3 = SETTINGS.wsm_robocasa_s3
LONG_CONTEXT_STUDY_S3 = SETTINGS.long_context_study_s3

RESEARCH_ROOT = SETTINGS.research_root
TRI_ROOT = SETTINGS.tri_root
WSM_DATA_ROOT = SETTINGS.wsm_data_root
INTERNAL_TRAINING_ROOT = SETTINGS.internal_training_root
ROBOMME_EVAL_ROOT = SETTINGS.robomme_eval_root
ROBOCASA_OPENPI_ROOT = SETTINGS.robocasa_openpi_root
ROBOCASA_OPENPI_SRC = SETTINGS.robocasa_openpi_src
ROBOCASA_ROOT = SETTINGS.robocasa_root
ROBOSUITE_ROOT = SETTINGS.robosuite_root
ENVS_ROOT = SETTINGS.envs_root

__all__ = [
    "DEXJOCO_IMAGE_REPO",
    "ECR_REGISTRY",
    "ENVS_ROOT",
    "ENV_VARS",
    "EXECUTION_ACCOUNT",
    "IMAGE_OWNER",
    "INTERNAL_TRAINING_ROOT",
    "LEGACY_ACCOUNT",
    "LEGACY_RESULTS_BUCKET",
    "LONG_CONTEXT_STUDY_S3",
    "OWNER_EMAIL",
    "PROJECT_TAG",
    "REGION",
    "RESEARCH_ROOT",
    "RESULTS_BUCKET",
    "ROBOCASA_OPENPI_ROOT",
    "ROBOCASA_OPENPI_SRC",
    "ROBOCASA_ROOT",
    "ROBOMME_EVAL_ROOT",
    "ROBOSUITE_ROOT",
    "SETTINGS",
    "STORAGE_ACCOUNT",
    "STUDY_OWNER",
    "Settings",
    "TRI_ROOT",
    "WSM_DATA_ROOT",
    "WSM_ROBOCASA_S3",
    "describe",
    "resolve",
]

if __name__ == "__main__":
    print(describe())

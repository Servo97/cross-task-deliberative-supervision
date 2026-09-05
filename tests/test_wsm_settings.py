"""`wsm_settings`: defaults reproduce the historical literals, env overrides work, describe() runs.

Run: PYTHONPATH=. python -m pytest -q tests/test_wsm_settings.py
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import wsm_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path("/home/example")


def test_defaults_resolve_to_the_historical_literals():
    settings = wsm_settings.resolve(env={}, home=HOME)
    assert settings.execution_account == "141701954645"
    assert settings.storage_account == "141701954645"
    assert settings.legacy_account == "124224456861"
    assert settings.region == "us-west-2"
    assert settings.results_bucket == "sagemaker-us-west-2-141701954645"
    assert settings.legacy_results_bucket == "sagemaker-us-west-2-124224456861"
    assert settings.study_owner == "sarvesh.patil"
    assert settings.owner_email == "sarvesh.patil.pi@tri.global"
    assert settings.project_tag == "LONG-CONTEXT-VLA"
    assert settings.image_owner == "sarvesh.patil"
    assert settings.ecr_registry == "141701954645.dkr.ecr.us-west-2.amazonaws.com"
    assert settings.dexjoco_image_repo == "141701954645.dkr.ecr.us-west-2.amazonaws.com/sarvesh.patil-groot-dexjoco"
    assert settings.wsm_robocasa_s3 == "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa"
    assert settings.long_context_study_s3 == (
        "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"
    )
    assert settings.from_env == ()


def test_default_paths_are_home_anchored():
    settings = wsm_settings.resolve(env={}, home=HOME)
    assert settings.research_root == HOME / "Research"
    assert settings.tri_root == HOME / "Research" / "TRI"
    assert settings.wsm_data_root == HOME / "Research" / "TRI" / "wsm_data"
    assert settings.internal_training_root == HOME / "Research" / "TRI" / "internal_training"
    assert settings.robomme_eval_root == HOME / "Research" / "TRI" / "robomme_eval"
    assert settings.robocasa_openpi_root == HOME / "Research" / "robocasa_openpi"
    assert settings.robocasa_openpi_src == HOME / "Research" / "robocasa_openpi" / "src"
    assert settings.robocasa_root == HOME / "Research" / "robocasa"
    assert settings.robosuite_root == HOME / "Research" / "robosuite"
    assert settings.envs_root == HOME / "Research" / "envs"


def test_module_constants_match_a_fresh_resolution_of_the_process_environment():
    settings = wsm_settings.resolve()
    assert wsm_settings.SETTINGS == settings
    assert wsm_settings.EXECUTION_ACCOUNT == settings.execution_account
    assert wsm_settings.STORAGE_ACCOUNT == settings.storage_account
    assert wsm_settings.LEGACY_ACCOUNT == settings.legacy_account
    assert wsm_settings.REGION == settings.region
    assert wsm_settings.RESULTS_BUCKET == settings.results_bucket
    assert wsm_settings.LEGACY_RESULTS_BUCKET == settings.legacy_results_bucket
    assert wsm_settings.STUDY_OWNER == settings.study_owner
    assert wsm_settings.OWNER_EMAIL == settings.owner_email
    assert wsm_settings.PROJECT_TAG == settings.project_tag
    assert wsm_settings.IMAGE_OWNER == settings.image_owner
    assert wsm_settings.ECR_REGISTRY == settings.ecr_registry
    assert wsm_settings.DEXJOCO_IMAGE_REPO == settings.dexjoco_image_repo
    assert wsm_settings.WSM_ROBOCASA_S3 == settings.wsm_robocasa_s3
    assert wsm_settings.LONG_CONTEXT_STUDY_S3 == settings.long_context_study_s3
    assert wsm_settings.RESEARCH_ROOT == settings.research_root
    assert wsm_settings.TRI_ROOT == settings.tri_root
    assert wsm_settings.WSM_DATA_ROOT == settings.wsm_data_root
    assert wsm_settings.INTERNAL_TRAINING_ROOT == settings.internal_training_root
    assert wsm_settings.ROBOMME_EVAL_ROOT == settings.robomme_eval_root
    assert wsm_settings.ROBOCASA_OPENPI_ROOT == settings.robocasa_openpi_root
    assert wsm_settings.ROBOCASA_OPENPI_SRC == settings.robocasa_openpi_src
    assert wsm_settings.ROBOCASA_ROOT == settings.robocasa_root
    assert wsm_settings.ROBOSUITE_ROOT == settings.robosuite_root
    assert wsm_settings.ENVS_ROOT == settings.envs_root


def test_env_overrides_flow_into_derived_values():
    env = {
        "WSM_EXECUTION_ACCOUNT": "111111111111",
        "WSM_STORAGE_ACCOUNT": "000000000000",
        "WSM_REGION": "eu-west-1",
        "WSM_STUDY_OWNER": "other.person",
        "WSM_IMAGE_OWNER": "team",
        "WSM_RESEARCH_ROOT": "/srv/r",
    }
    settings = wsm_settings.resolve(env=env, home=HOME)
    assert settings.results_bucket == "sagemaker-eu-west-1-000000000000"
    assert settings.wsm_robocasa_s3 == "s3://sagemaker-eu-west-1-000000000000/other.person/wsm_robocasa"
    assert settings.long_context_study_s3.endswith("/other.person/wsm_robocasa/studies/long_context_v1")
    assert settings.dexjoco_image_repo == "111111111111.dkr.ecr.eu-west-1.amazonaws.com/team-groot-dexjoco"
    assert settings.legacy_results_bucket == "sagemaker-eu-west-1-124224456861"
    assert settings.tri_root == Path("/srv/r/TRI")
    assert settings.wsm_data_root == Path("/srv/r/TRI/wsm_data")
    assert settings.robocasa_openpi_src == Path("/srv/r/robocasa_openpi/src")
    assert set(settings.from_env) == {
        "execution_account",
        "storage_account",
        "region",
        "study_owner",
        "image_owner",
        "research_root",
    }
    # the storage prefix and the submitting identity are independent knobs
    assert settings.owner_email == "sarvesh.patil.pi@tri.global"


def test_explicit_bucket_wins_and_blank_values_count_as_unset():
    settings = wsm_settings.resolve(env={"WSM_RESULTS_BUCKET": "my-bucket", "WSM_STUDY_OWNER": "   "}, home=HOME)
    assert settings.results_bucket == "my-bucket"
    assert settings.study_owner == "sarvesh.patil"
    assert settings.wsm_robocasa_s3 == "s3://my-bucket/sarvesh.patil/wsm_robocasa"
    assert settings.from_env == ("results_bucket",)


def test_path_overrides_expand_tilde_and_leaf_overrides_do_not_move_their_siblings():
    settings = wsm_settings.resolve(env={"WSM_DATA_ROOT": "~/data", "WSM_TRI_ROOT": "/mnt/tri"}, home=HOME)
    assert settings.wsm_data_root == Path("~/data").expanduser()
    assert settings.tri_root == Path("/mnt/tri")
    assert settings.internal_training_root == Path("/mnt/tri/internal_training")
    assert settings.robomme_eval_root == Path("/mnt/tri/robomme_eval")
    assert settings.research_root == HOME / "Research"
    assert settings.envs_root == HOME / "Research" / "envs"


def test_every_field_has_a_documented_env_var_with_the_prefix():
    fields = {field.name for field in dataclasses.fields(wsm_settings.Settings)} - {"from_env"}
    assert fields == set(wsm_settings.ENV_VARS)
    for variable, _ in wsm_settings.ENV_VARS.values():
        assert variable.startswith(wsm_settings.ENV_PREFIX), variable


def test_describe_lists_every_variable_and_its_origin():
    text = wsm_settings.describe(wsm_settings.resolve(env={"WSM_STUDY_OWNER": "x"}, home=HOME))
    for variable, _ in wsm_settings.ENV_VARS.values():
        assert variable in text
    assert "[WSM_STUDY_OWNER: env]" in text
    assert "[WSM_EXECUTION_ACCOUNT: default]" in text
    assert "s3://sagemaker-us-west-2-141701954645/x/wsm_robocasa" in text
    assert "execution_account" in wsm_settings.describe()


def test_cli_prints_the_resolved_table():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "wsm_settings.py")],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(HOME)},
    )
    assert "141701954645" in result.stdout
    assert str(HOME / "Research" / "TRI" / "wsm_data") in result.stdout


def test_launch_guardrails_re_exports_the_settings():
    launch_dir = str(REPO_ROOT / "scripts" / "launch")
    if launch_dir not in sys.path:
        sys.path.insert(0, launch_dir)
    import launch_guardrails as guardrails

    assert guardrails.wsm_settings is wsm_settings
    assert guardrails.EXECUTION_ACCOUNT == wsm_settings.EXECUTION_ACCOUNT
    assert guardrails.STORAGE_ACCOUNT == wsm_settings.STORAGE_ACCOUNT
    assert guardrails.LEGACY_ACCOUNT == wsm_settings.LEGACY_ACCOUNT
    assert guardrails.REGION == wsm_settings.REGION
    assert guardrails.DEFAULT_RESULTS_BUCKET == wsm_settings.RESULTS_BUCKET
    assert guardrails.STUDY_OWNER == wsm_settings.STUDY_OWNER
    assert guardrails.OWNER_EMAIL == wsm_settings.OWNER_EMAIL
    assert guardrails.PROJECT_TAG == wsm_settings.PROJECT_TAG
    assert guardrails.IMAGE_OWNER == wsm_settings.IMAGE_OWNER
    assert guardrails.DEXJOCO_IMAGE_REPO == wsm_settings.DEXJOCO_IMAGE_REPO
    assert guardrails.WSM_ROBOCASA_S3 == wsm_settings.WSM_ROBOCASA_S3
    assert guardrails.LONG_CONTEXT_STUDY_S3 == wsm_settings.LONG_CONTEXT_STUDY_S3
    assert guardrails.ROLE_ARN == (
        f"arn:aws:iam::{wsm_settings.EXECUTION_ACCOUNT}:role/CAM-Robotics-Sagemaker-role-{wsm_settings.REGION}"
    )

"""Runtime overlays over sealed RoboMME modules.

Nothing here edits `robomme_integration/`. `launch.py:656-658` folds
`sanitized_source_tree_sha256` over that WHOLE tree into `scientific_spec_sha256` -> `run_id` ->
every S3 path and claim URI, so a one-character edit there re-addresses every RoboMME run. These
modules therefore patch the sealed identity TABLES in memory at import time on the node, the same
way `wsm_robocasa_configs.install()` registers openpi configs without mutating the checkout.
"""

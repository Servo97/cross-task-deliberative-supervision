#!/usr/bin/env python3
"""Fully decode every pinned RoboMME Parquet row group after an integrity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .inventory import HF_REPO, HF_REVISION


def validate(
    audit_report,
    episodes_metadata,
    output,
    *,
    workers: int = 16,
) -> dict:
    import pyarrow.parquet as parquet

    audit_path = Path(audit_report).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("hf_repo_id") != HF_REPO
        or audit.get("hf_revision") != HF_REVISION
        or audit.get("summary", {}).get("s3_source_mismatches") != 0
        or len(audit.get("records", [])) != 1600
    ):
        raise ValueError("input is not a successful full RoboMME source/S3 audit")
    snapshot = Path(audit["snapshot"])
    overlay = Path(audit["overlay"]) if audit.get("overlay") else None

    episode_rows = {}
    with Path(episodes_metadata).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                episode_rows[record["episode_index"]] = record["length"]
    if len(episode_rows) != 1600:
        raise ValueError(f"expected 1600 episode records, found {len(episode_rows)}")

    def decode(record: dict) -> dict:
        relative = Path(record["key"])
        path = overlay / relative if record["overlay_used"] else snapshot / relative
        episode_index = int(relative.stem.removeprefix("episode_"))
        expected_rows = episode_rows[episode_index]
        source = parquet.ParquetFile(path)
        if source.metadata.num_rows != expected_rows:
            raise ValueError(
                f"metadata row mismatch for {record['key']}: {source.metadata.num_rows} != {expected_rows}"
            )
        rows = 0
        for row_group in range(source.metadata.num_row_groups):
            table = source.read_row_group(row_group, use_threads=False)
            table.validate(full=True)
            rows += table.num_rows
        if rows != expected_rows:
            raise ValueError(f"decoded row mismatch for {record['key']}: {rows} != {expected_rows}")
        return {
            "key": record["key"],
            "rows": rows,
            "row_groups": source.metadata.num_row_groups,
        }

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(audit["records"]))) as pool:
        futures = {pool.submit(decode, record): record for record in audit["records"]}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(audit["records"]):
                print(f"[decode] verified {index}/{len(audit['records'])}", flush=True)
    results.sort(key=lambda item: item["key"])

    report = {
        "schema_version": 1,
        "hf_repo_id": HF_REPO,
        "hf_revision": HF_REVISION,
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "summary": {
            "parquet_files": len(results),
            "rows": sum(item["rows"] for item in results),
            "row_groups": sum(item["row_groups"] for item in results),
            "fully_decoded": len(results),
        },
        "files": results,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".incomplete")
    temporary.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--episodes-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = validate(
        args.audit_report,
        args.episodes_metadata,
        args.output,
        workers=args.workers,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

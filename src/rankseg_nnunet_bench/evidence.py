from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import yaml

from .aggregate import aggregate_summaries
from .schema import AGGREGATE_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION, migrate_summary, validate_summary


EVIDENCE_MANIFEST_SCHEMA_VERSION = 1
_WORKSPACE_DIRECTORIES = ("artifacts", "configs", "data", "outputs", "registry", "results", "work")
_AGGREGATE_FILES = (
    "aggregate_summary.json",
    "case_paired_deltas.csv",
    "dataset_summary.csv",
    "RESULTS.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sanitize_paths(value: Any, roots: tuple[Path, ...]) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_paths(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item, roots) for item in value]
    if isinstance(value, str):
        result = value
        for root in roots:
            result = result.replace(str(root), "${REPO_ROOT}")
        path = Path(result)
        if path.is_absolute():
            for index, part in enumerate(path.parts):
                if part in _WORKSPACE_DIRECTORIES:
                    result = "${REPO_ROOT}/" + "/".join(path.parts[index:])
                    break
        return result
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_local_paths(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [path for item in value.values() for path in _absolute_local_paths(item)]
    if isinstance(value, list):
        return [path for item in value for path in _absolute_local_paths(item)]
    if isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        return [value]
    return []


def _evidence_readme(dataset_count: int, case_count: int) -> str:
    return f"""# Published Full-16 evidence

This directory is the small, version-controlled evidence package for the RankSEG × nnU-Net benchmark. It contains
no images, labels, model weights, probability arrays, or restricted Task035 artifacts.

- `{dataset_count}` dataset summaries use summary schema v{SUMMARY_SCHEMA_VERSION}.
- `{case_count}` paired case-level Dice rows are retained for regression auditing.
- `full16/` is rebuilt from the dataset summaries and case deltas.
- `MANIFEST.json` records the SHA256 and byte size of every published file.

Verify the checked-in package without access to the large local benchmark workspace:

```bash
python -m rankseg_nnunet_bench.cli verify-evidence evidence
```

Maintainers with the ignored local `outputs/` workspace can regenerate it from `configs/full16_evidence.yaml`:

```bash
python -m rankseg_nnunet_bench.cli publish-evidence configs/full16_evidence.yaml \\
  --output-dir evidence-new
```

Legacy measurements that predate a timing column are represented as JSON `null`; the migration never infers a
device or resource measurement that was not recorded.
"""


def publish_evidence(manifest_path: Path, output_dir: Path) -> Path:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {output_dir}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("summaries"), list):
        raise ValueError(f"Evidence source manifest must contain a summaries list: {manifest_path}")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported evidence source manifest schema: {raw.get('schema_version')!r}")

    repository_root = manifest_path.parent.parent.resolve()
    legacy_roots = tuple(
        dict.fromkeys(
            (
                repository_root,
                *(Path(value).expanduser() for value in raw.get("legacy_repository_roots", [])),
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    published_summaries: list[Path] = []
    dataset_entries: list[dict[str, Any]] = []
    total_cases = 0

    for relative in raw["summaries"]:
        source_summary = (manifest_path.parent / str(relative)).resolve()
        source_dir = source_summary.parent
        timings_path = source_dir / "timings.csv"
        timings = pd.read_csv(timings_path) if timings_path.is_file() else None
        summary = migrate_summary(_read_json(source_summary), timings=timings)
        summary = _sanitize_paths(summary, legacy_roots)
        validate_summary(summary)

        dataset_id = str(summary["dataset"]["id"])
        destination_dir = output_dir / "datasets" / dataset_id
        destination_summary = destination_dir / "summary.json"
        _write_json(destination_summary, summary)

        case_source = source_dir / "case_paired_deltas.csv"
        if not case_source.is_file():
            raise FileNotFoundError(f"Missing paired case deltas for published dataset: {case_source}")
        case_frame = pd.read_csv(case_source)
        if len(case_frame) != int(summary["dataset"]["cases"]):
            raise ValueError(f"Case delta count does not match summary: {case_source}")
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case_source, destination_dir / "case_paired_deltas.csv")

        audit_source = source_dir / "AUDIT.json"
        audit_relative = None
        if audit_source.is_file():
            audit_destination = destination_dir / "AUDIT.json"
            _write_json(audit_destination, _sanitize_paths(_read_json(audit_source), legacy_roots))
            audit_relative = str(audit_destination.relative_to(output_dir))

        published_summaries.append(destination_summary)
        total_cases += int(summary["dataset"]["cases"])
        dataset_entries.append(
            {
                "dataset_id": dataset_id,
                "cases": int(summary["dataset"]["cases"]),
                "summary": str(destination_summary.relative_to(output_dir)),
                "case_paired_deltas": str(
                    (destination_dir / "case_paired_deltas.csv").relative_to(output_dir)
                ),
                "audit": audit_relative,
            }
        )

    if len({entry["dataset_id"] for entry in dataset_entries}) != len(dataset_entries):
        raise ValueError("Evidence source manifest contains duplicate dataset IDs")

    aggregate_dir = output_dir / "full16"
    aggregate_summaries(
        published_summaries,
        aggregate_dir,
        include_overall_tests=bool(raw.get("include_overall_tests", True)),
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(_evidence_readme(len(dataset_entries), total_cases), encoding="utf-8")

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.json":
            continue
        relative = str(path.relative_to(output_dir))
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    publication_manifest = {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_manifest": str(manifest_path.relative_to(repository_root)),
        "include_overall_tests": bool(raw.get("include_overall_tests", True)),
        "datasets": dataset_entries,
        "dataset_count": len(dataset_entries),
        "case_count": total_cases,
        "files": files,
    }
    publication_path = output_dir / "MANIFEST.json"
    _write_json(publication_path, publication_manifest)
    verify_evidence(output_dir)
    return publication_path


def verify_evidence(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EVIDENCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evidence manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Evidence summary schema does not match the installed benchmark")
    if manifest.get("aggregate_schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise ValueError("Evidence aggregate schema does not match the installed benchmark")

    listed = set(manifest.get("files", {}))
    observed = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if listed != observed:
        raise ValueError(
            f"Evidence file listing mismatch; missing={sorted(listed - observed)}, "
            f"unlisted={sorted(observed - listed)}"
        )
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"Evidence file size mismatch: {path}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"Evidence SHA256 mismatch: {path}")

    for json_path in output_dir.rglob("*.json"):
        if json_path == manifest_path:
            continue
        local_paths = _absolute_local_paths(_read_json(json_path))
        if local_paths:
            raise ValueError(
                f"Published JSON contains machine-local absolute paths: {json_path}: {local_paths}"
            )

    dataset_entries = manifest.get("datasets", [])
    if len(dataset_entries) != int(manifest.get("dataset_count", -1)):
        raise ValueError("Evidence dataset count does not match its manifest")
    summary_paths = []
    case_count = 0
    for entry in dataset_entries:
        summary_path = output_dir / entry["summary"]
        summary = _read_json(summary_path)
        validate_summary(summary)
        if summary["dataset"]["id"] != entry["dataset_id"]:
            raise ValueError(f"Evidence dataset ID mismatch: {summary_path}")
        case_path = output_dir / entry["case_paired_deltas"]
        cases = pd.read_csv(case_path)
        if len(cases) != int(entry["cases"]):
            raise ValueError(f"Evidence case count mismatch: {case_path}")
        case_count += len(cases)
        summary_paths.append(summary_path)
    if case_count != int(manifest.get("case_count", -1)):
        raise ValueError("Evidence total case count does not match its manifest")

    with TemporaryDirectory(prefix="rankseg-evidence-") as temporary:
        rebuilt_dir = Path(temporary) / "full16"
        aggregate_summaries(
            summary_paths,
            rebuilt_dir,
            include_overall_tests=bool(manifest.get("include_overall_tests", True)),
        )
        for filename in _AGGREGATE_FILES:
            checked_in = output_dir / "full16" / filename
            rebuilt = rebuilt_dir / filename
            if checked_in.read_bytes() != rebuilt.read_bytes():
                raise ValueError(f"Published aggregate is stale: {checked_in}")
    return manifest_path

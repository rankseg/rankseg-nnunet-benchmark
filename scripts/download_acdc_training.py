#!/usr/bin/env python3
"""Download only the annotated ACDC ED/ES frames from the official public Girder record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
TRAINING_FOLDER_ID = "63721d7073e9f0047faa0525"
FRAME_PATTERN = re.compile(r"^patient[0-9]{3}_frame[0-9]+(?:_gt)?\.nii\.gz$")


def _get_json(url: str, *, attempts: int = 6) -> Any:
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=(30, 120))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(item: dict[str, Any], patient: str, output_dir: Path) -> dict[str, Any]:
    destination_dir = output_dir / patient
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / item["name"]
    expected_size = int(item["size"])
    if not destination.is_file() or destination.stat().st_size != expected_size:
        part = destination.with_name(destination.name + ".part")
        url = f"{BASE_URL}/item/{item['_id']}/download"
        for attempt in range(6):
            try:
                with requests.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with part.open("wb") as handle:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                handle.write(block)
                if part.stat().st_size != expected_size:
                    raise IOError(
                        f"Size mismatch for {item['name']}: {part.stat().st_size} != {expected_size}"
                    )
                os.replace(part, destination)
                break
            except (requests.RequestException, OSError):
                if attempt + 1 == 6:
                    raise
                time.sleep(2**attempt)
    return {
        "patient_id": patient,
        "name": item["name"],
        "item_id": item["_id"],
        "size_bytes": expected_size,
        "sha256": _sha256(destination),
        "download_url": f"{BASE_URL}/item/{item['_id']}/download",
    }


def download(output_dir: Path, workers: int) -> None:
    folders = _get_json(
        f"{BASE_URL}/folder?parentType=folder&parentId={TRAINING_FOLDER_ID}&limit=0"
    )
    folder_by_patient = {folder["name"]: folder for folder in folders}
    expected = {f"patient{index:03d}" for index in range(1, 101)}
    if set(folder_by_patient) != expected:
        raise ValueError(f"Expected patient001..patient100, found {len(folder_by_patient)} folders")

    def list_items(patient: str) -> tuple[str, list[dict[str, Any]]]:
        folder_id = folder_by_patient[patient]["_id"]
        items = _get_json(f"{BASE_URL}/item?folderId={folder_id}&limit=0")
        selected = sorted(
            (item for item in items if FRAME_PATTERN.fullmatch(item["name"])),
            key=lambda item: item["name"],
        )
        if len(selected) != 4:
            raise ValueError(f"Expected two frames and two labels for {patient}, found {len(selected)}")
        return patient, selected

    selected_items: list[tuple[str, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(list_items, patient): patient for patient in sorted(expected)}
        for completed, future in enumerate(as_completed(futures), start=1):
            patient, items = future.result()
            selected_items.extend((patient, item) for item in items)
            if completed % 10 == 0:
                print(f"Listed {completed}/100 patient folders", flush=True)
    expected_bytes = sum(int(item["size"]) for _, item in selected_items)
    print(
        f"Downloading {len(selected_items)} annotated files ({expected_bytes / 2**20:.1f} MiB) "
        f"with {workers} workers",
        flush=True,
    )

    manifest_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, item, patient, output_dir): (patient, item["name"])
            for patient, item in selected_items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            manifest_rows.append(future.result())
            if completed % 20 == 0 or completed == len(futures):
                print(f"Downloaded/verified {completed}/{len(futures)} files", flush=True)

    manifest = {
        "source": "official ACDC public Girder training folder",
        "base_url": BASE_URL,
        "training_folder_id": TRAINING_FOLDER_ID,
        "patients": 100,
        "files": len(manifest_rows),
        "total_size_bytes": sum(row["size_bytes"] for row in manifest_rows),
        "selection": "two annotated ED/ES frame images and their labels; 4D cine excluded",
        "items": sorted(manifest_rows, key=lambda row: (row["patient_id"], row["name"])),
    }
    (output_dir / "official_download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {output_dir / 'official_download_manifest.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download(args.output_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()

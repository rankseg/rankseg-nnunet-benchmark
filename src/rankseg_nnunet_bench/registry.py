from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import time
import zlib
import zipfile

import requests
import yaml


_LOCAL_ZIP_HEADER = struct.Struct("<4s5H3I2H")
_ACTIVE_DOWNLOAD_LOCKS: dict[Path, object] = {}


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict) or not ({"models", "datasets"} & set(registry)):
        raise ValueError(f"Invalid artifact registry: {path}")
    return registry


def model_by_task(registry: dict, task: str) -> dict:
    matches = [model for model in registry["models"] if model["task"] == task]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one registry entry for {task!r}, found {len(matches)}")
    return matches[0]


def checksum(path: Path, algorithm: str = "md5", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s3_multipart_etag(path: Path, part_size: int = 8 * 1024 * 1024) -> str:
    """Reproduce the ETag for an S3/MinIO multipart object with a known part size."""
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    part_digests: list[bytes] = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(part_size), b""):
            part_digests.append(hashlib.md5(chunk).digest())  # noqa: S324 - S3 ETag compatibility
    if not part_digests:
        raise ValueError(f"Cannot compute multipart ETag for empty file: {path}")
    digest = hashlib.md5(b"".join(part_digests)).hexdigest()  # noqa: S324 - S3 ETag compatibility
    return f"{digest}-{len(part_digests)}"


def _validate_parallel_artifact(
    path: Path,
    *,
    expected_md5: str | None,
    expected_s3_etag: str | None,
    s3_part_size: int,
) -> tuple[bool, dict[str, str]]:
    if expected_md5 is None and expected_s3_etag is None:
        raise ValueError("At least one publisher checksum or multipart ETag is required")
    observed: dict[str, str] = {}
    valid = True
    if expected_md5 is not None:
        observed["md5"] = checksum(path)
        valid = valid and observed["md5"] == expected_md5
    if expected_s3_etag is not None:
        observed["s3_multipart_etag"] = s3_multipart_etag(path, s3_part_size)
        valid = valid and observed["s3_multipart_etag"] == expected_s3_etag.strip('"')
    return valid, observed


def download_file(
    *,
    url: str,
    destination: Path,
    expected_md5: str,
    expected_size: int | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size_ok = expected_size is None or destination.stat().st_size == expected_size
        if size_ok and checksum(destination) == expected_md5:
            return destination
        raise ValueError(f"Existing file failed size/checksum validation: {destination}")

    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        if existing and not append:
            existing = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {partial}: {partial.stat().st_size} != {expected_size}"
        )
    observed = checksum(partial)
    if observed != expected_md5:
        raise ValueError(f"Downloaded MD5 mismatch for {partial}: {observed} != {expected_md5}")
    partial.replace(destination)
    return destination


def _partition_byte_ranges(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if start > end:
        return []
    total = end - start + 1
    parts = min(parts, total)
    quotient, remainder = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    position = start
    for index in range(parts):
        length = quotient + (1 if index < remainder else 0)
        ranges.append((position, position + length - 1))
        position += length
    return ranges


def _download_range(
    url: str,
    path: Path,
    start: int,
    end: int,
    chunk_size: int,
    *,
    maximum_attempts: int = 20,
) -> Path:
    expected_size = end - start + 1
    existing = path.stat().st_size if path.exists() else 0
    if existing > expected_size:
        raise ValueError(f"Range partial is too large: {path}")
    if existing == expected_size:
        return path
    attempts = 0
    while (path.stat().st_size if path.exists() else 0) < expected_size:
        existing = path.stat().st_size if path.exists() else 0
        request_start = start + existing
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={request_start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(
                        f"Server did not honor range {request_start}-{end}: HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range")
                if content_range is not None:
                    try:
                        unit, value = content_range.split(" ", 1)
                        bounds, _total = value.split("/", 1)
                        response_start, response_end = (int(value) for value in bounds.split("-", 1))
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Invalid Content-Range response: {content_range!r}") from error
                    if unit.lower() != "bytes" or response_start != request_start or response_end > end:
                        raise ValueError(
                            f"Server returned unexpected range {content_range!r}; requested {request_start}-{end}"
                        )
                remaining = expected_size - existing
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            write_size = min(len(chunk), remaining)
                            handle.write(chunk[:write_size])
                            remaining -= write_size
                            if remaining == 0:
                                break
        except requests.RequestException:
            attempts += 1
            if attempts >= maximum_attempts:
                raise
            time.sleep(min(2 ** (attempts - 1), 30))
    if path.stat().st_size != expected_size:
        raise ValueError(f"Range size mismatch for {path}: {path.stat().st_size} != {expected_size}")
    return path


def _complete_range_in_parallel(
    *,
    urls: list[str],
    path: Path,
    start: int,
    end: int,
    connections: int,
    chunk_size: int,
) -> Path:
    """Subdivide the missing suffix when only one original partition remains."""
    expected_size = end - start + 1
    existing = path.stat().st_size if path.exists() else 0
    if existing > expected_size:
        raise ValueError(f"Range partial is too large: {path}")
    if existing == expected_size:
        return path
    missing_ranges = _partition_byte_ranges(start + existing, end, connections)
    missing_paths = [
        path.with_name(f"{path.name}.resume-{range_start}-{range_end}")
        for range_start, range_end in missing_ranges
    ]
    with ThreadPoolExecutor(max_workers=len(missing_ranges)) as executor:
        futures = [
            executor.submit(
                _download_range,
                urls[index % len(urls)],
                missing_path,
                range_start,
                range_end,
                chunk_size,
            )
            for index, (missing_path, (range_start, range_end)) in enumerate(
                zip(missing_paths, missing_ranges)
            )
        ]
        for future in futures:
            future.result()

    merged = path.with_name(path.name + ".merging")
    with merged.open("wb") as output:
        if path.exists():
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=chunk_size)
        for missing_path in missing_paths:
            with missing_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=chunk_size)
    if merged.stat().st_size != expected_size:
        raise ValueError(f"Merged range size mismatch: {merged.stat().st_size} != {expected_size}")
    merged.replace(path)
    for missing_path in missing_paths:
        missing_path.unlink()
    return path


def _download_file_parallel_locked(
    *,
    urls: list[str],
    destination: Path,
    expected_md5: str | None = None,
    expected_s3_etag: str | None = None,
    s3_part_size: int = 8 * 1024 * 1024,
    expected_size: int,
    connections: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    if connections <= 1:
        if expected_md5 is None:
            raise ValueError("Single-connection downloads currently require a publisher MD5")
        return download_file(
            url=urls[0],
            destination=destination,
            expected_md5=expected_md5,
            expected_size=expected_size,
            chunk_size=chunk_size,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        valid, observed = _validate_parallel_artifact(
            destination,
            expected_md5=expected_md5,
            expected_s3_etag=expected_s3_etag,
            s3_part_size=s3_part_size,
        )
        if destination.stat().st_size == expected_size and valid:
            return destination
        raise ValueError(
            f"Existing file failed size/checksum validation: {destination}; observed={observed}"
        )
    partial = destination.with_name(destination.name + ".part")
    base_size = partial.stat().st_size if partial.exists() else 0
    if base_size > expected_size:
        raise ValueError(f"Existing partial is larger than the expected archive: {partial}")
    if base_size == expected_size:
        valid, observed = _validate_parallel_artifact(
            partial,
            expected_md5=expected_md5,
            expected_s3_etag=expected_s3_etag,
            s3_part_size=s3_part_size,
        )
        if not valid:
            raise ValueError(f"Completed partial checksum mismatch for {partial}: {observed}")
        partial.replace(destination)
        return destination
    ranges = _partition_byte_ranges(base_size, expected_size - 1, connections)
    range_paths = [
        partial.with_name(f"{partial.name}.range-{start}-{end}") for start, end in ranges
    ]
    incomplete = []
    for path, (start, end) in zip(range_paths, ranges):
        expected_range_size = end - start + 1
        observed_range_size = path.stat().st_size if path.exists() else 0
        if observed_range_size > expected_range_size:
            raise ValueError(f"Range partial is too large: {path}")
        if observed_range_size < expected_range_size:
            incomplete.append((path, start, end))
    if len(incomplete) == 1 and connections > 1:
        path, start, end = incomplete[0]
        _complete_range_in_parallel(
            urls=urls,
            path=path,
            start=start,
            end=end,
            connections=connections,
            chunk_size=chunk_size,
        )
    else:
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(
                    _download_range,
                    urls[index % len(urls)],
                    path,
                    start,
                    end,
                    chunk_size,
                )
                for index, (path, (start, end)) in enumerate(zip(range_paths, ranges))
            ]
            for future in futures:
                future.result()

    assembled = destination.with_name(destination.name + ".assembling")
    with assembled.open("wb") as output:
        if partial.exists():
            with partial.open("rb") as source:
                shutil.copyfileobj(source, output, length=chunk_size)
        for path in range_paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=chunk_size)
    if assembled.stat().st_size != expected_size:
        raise ValueError(f"Assembled size mismatch: {assembled.stat().st_size} != {expected_size}")
    valid, observed = _validate_parallel_artifact(
        assembled,
        expected_md5=expected_md5,
        expected_s3_etag=expected_s3_etag,
        s3_part_size=s3_part_size,
    )
    if not valid:
        raise ValueError(f"Assembled checksum mismatch for {assembled}: {observed}")
    assembled.replace(destination)
    if partial.exists():
        partial.unlink()
    for path in range_paths:
        path.unlink()
    return destination


def download_file_parallel(
    *,
    urls: list[str],
    destination: Path,
    expected_size: int,
    connections: int,
    expected_md5: str | None = None,
    expected_s3_etag: str | None = None,
    s3_part_size: int = 8 * 1024 * 1024,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download with an inter-process destination lock that survives interrupted worker joins."""
    lock_key = destination.resolve()
    lock_path = destination.with_name(destination.name + ".download.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_key in _ACTIVE_DOWNLOAD_LOCKS:
        raise RuntimeError(f"Download already active in this process: {destination}")
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"Another process is already downloading to {destination}") from error
    _ACTIVE_DOWNLOAD_LOCKS[lock_key] = lock_handle
    try:
        result = _download_file_parallel_locked(
            urls=urls,
            destination=destination,
            expected_size=expected_size,
            connections=connections,
            expected_md5=expected_md5,
            expected_s3_etag=expected_s3_etag,
            s3_part_size=s3_part_size,
            chunk_size=chunk_size,
        )
    except KeyboardInterrupt:
        # ThreadPoolExecutor workers may still be writing while the main thread unwinds. Keep the
        # descriptor in the process-global mapping so no replacement process can acquire the lock.
        raise
    except BaseException:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        _ACTIVE_DOWNLOAD_LOCKS.pop(lock_key, None)
        raise
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    _ACTIVE_DOWNLOAD_LOCKS.pop(lock_key, None)
    return result


def download_model(model: dict, output_dir: Path, *, connections: int = 1) -> Path:
    urls = [model.get("download_mirror", model["url"])]
    if model["url"] not in urls:
        urls.append(model["url"])
    if connections > 1:
        return download_file_parallel(
            urls=urls,
            destination=output_dir / model["file"],
            expected_md5=model["md5"],
            expected_size=int(model["size_bytes"]),
            connections=connections,
        )
    return download_file(
        url=urls[0],
        destination=output_dir / model["file"],
        expected_md5=model["md5"],
        expected_size=int(model["size_bytes"]),
    )


def dataset_by_task(registry: dict, task: str) -> dict:
    matches = [dataset for dataset in registry["datasets"] if dataset["task"] == task]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one dataset registry entry for {task!r}, found {len(matches)}")
    return matches[0]


def download_dataset(dataset: dict, output_dir: Path, *, connections: int = 1) -> Path:
    filename = dataset.get("file", f"{dataset['task']}.tar")
    if connections > 1:
        if "size_bytes" not in dataset:
            raise ValueError(
                f"Parallel dataset download requires size_bytes in the registry for {dataset['task']}"
            )
        return download_file_parallel(
            urls=[dataset["url"]],
            destination=output_dir / filename,
            expected_md5=dataset.get("md5"),
            expected_s3_etag=dataset.get("s3_multipart_etag"),
            s3_part_size=int(dataset.get("s3_part_size_bytes", 8 * 1024 * 1024)),
            expected_size=int(dataset["size_bytes"]),
            connections=connections,
        )
    if "md5" not in dataset:
        raise ValueError(
            f"Single-connection dataset download requires a registered MD5 for {dataset['task']}"
        )
    return download_file(
        url=dataset["url"],
        destination=output_dir / filename,
        expected_md5=dataset["md5"],
        expected_size=int(dataset["size_bytes"]) if "size_bytes" in dataset else None,
    )


def read_first_zip_member(prefix: bytes) -> tuple[str, bytes]:
    """Read a complete first ZIP member without needing the archive's central directory."""
    if len(prefix) < _LOCAL_ZIP_HEADER.size:
        raise ValueError("ZIP prefix is shorter than the local header")
    (
        signature,
        _extract_version,
        flags,
        compression,
        _modification_time,
        _modification_date,
        expected_crc,
        compressed_size,
        uncompressed_size,
        filename_size,
        extra_size,
    ) = _LOCAL_ZIP_HEADER.unpack_from(prefix)
    if signature != b"PK\x03\x04":
        raise ValueError("ZIP prefix does not start with a local file header")
    if flags & 0x08:
        raise ValueError("First ZIP member uses a data descriptor; size is unavailable in its local header")
    payload_start = _LOCAL_ZIP_HEADER.size + filename_size + extra_size
    payload_end = payload_start + compressed_size
    if len(prefix) < payload_end:
        raise ValueError(f"ZIP prefix needs at least {payload_end} bytes, received {len(prefix)}")
    filename = prefix[_LOCAL_ZIP_HEADER.size : _LOCAL_ZIP_HEADER.size + filename_size].decode("utf-8")
    compressed = prefix[payload_start:payload_end]
    if compression == 0:
        content = compressed
    elif compression == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported ZIP compression method: {compression}")
    if len(content) != uncompressed_size:
        raise ValueError(f"Uncompressed size mismatch for {filename}")
    if zlib.crc32(content) & 0xFFFFFFFF != expected_crc:
        raise ValueError(f"CRC mismatch for {filename}")
    return filename, content


class HTTPRangeReader:
    """Minimal seekable reader that lets zipfile request only needed byte ranges."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        maximum_read: int = 32 * 1024 * 1024,
        maximum_attempts: int = 8,
    ):
        self.url = url
        self.size = size
        self.maximum_read = maximum_read
        self.maximum_attempts = maximum_attempts
        self.position = 0
        self.session = requests.Session()

    def tell(self) -> int:
        return self.position

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the remote file")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > self.maximum_read:
            raise ValueError(f"Refusing an unexpectedly large HTTP range read of {size} bytes")
        end = self.position + size - 1
        content = None
        for attempt in range(self.maximum_attempts):
            try:
                with self.session.get(
                    self.url,
                    headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
                    timeout=(30, 300),
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError(
                            f"Server did not honor ZIP byte-range request: HTTP {response.status_code}"
                        )
                    content = response.content
                break
            except requests.RequestException:
                if attempt + 1 >= self.maximum_attempts:
                    raise
                time.sleep(min(2**attempt, 30))
        if content is None:  # pragma: no cover - loop either succeeds or raises
            raise RuntimeError("HTTP range read ended without a response")
        if len(content) != size:
            raise ValueError(f"HTTP range returned {len(content)} bytes; expected {size}")
        self.position += size
        return content

    def close(self) -> None:
        self.session.close()


def _configuration_from_postprocessing_path(filename: str, task: str) -> str | None:
    parts = filename.split("/")
    try:
        task_index = parts.index(task)
    except ValueError:
        return None
    if task_index == 0 or task_index + 1 >= len(parts):
        return None
    if parts[task_index - 1] == "ensembles":
        return parts[task_index + 1]
    return parts[task_index - 1]


def _model_selections_from_archive(archive: zipfile.ZipFile, task: str) -> list[dict[str, str]]:
    expected = f"ensembles/{task}/summary.csv"
    summary_candidates = [name for name in archive.namelist() if name.endswith("summary.csv")]
    if expected in summary_candidates:
        selected_name = expected
    elif len(summary_candidates) == 1:
        selected_name = summary_candidates[0]
    else:
        selected_name = None

    if selected_name is not None:
        content = archive.read(selected_name)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        if not rows or not {"model", "average"}.issubset(rows[0]):
            raise ValueError(f"Invalid official model-selection summary in {selected_name}")
        return [{**row, "selection_source": "summary.csv"} for row in rows]

    postprocessing_names = [
        name
        for name in archive.namelist()
        if name.endswith("postprocessing.json") and f"/{task}/" in f"/{name}"
    ]
    rows: list[dict[str, str]] = []
    for filename in sorted(postprocessing_names):
        configuration = _configuration_from_postprocessing_path(filename, task)
        if configuration is None:
            continue
        metadata = json.loads(archive.read(filename))
        # ``dc_per_class_pp_per_class`` contains the diagnostic score obtained by
        # trying each connected-component operation in isolation. It is not the
        # score of the released final pipeline when several operations are
        # selected. ``dc_per_class_pp_all`` is the score after applying the full
        # learned postprocessing sequence and is therefore the model-selection
        # quantity represented by ``validation_final``. Older/no-op metadata can
        # omit it, in which case the raw score is the final score.
        score_by_class = metadata.get("dc_per_class_pp_all") or metadata.get("dc_per_class_raw")
        if not isinstance(score_by_class, dict) or not score_by_class:
            raise ValueError(f"Missing official Dice scores in {filename}")
        scores = [float(value) for value in score_by_class.values()]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Non-finite official Dice score in {filename}")
        rows.append(
            {
                "model": configuration,
                "average": str(sum(scores) / len(scores)),
                "selection_source": "postprocessing.json",
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"Could not identify official summary.csv or per-configuration postprocessing.json for {task}; "
            f"summary candidates: {summary_candidates}"
        )
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate model configurations in official postprocessing metadata for {task}")
    return rows


def fetch_model_selection_summary(model: dict) -> list[dict[str, str]]:
    """Fetch official selection metadata using small HTTP byte-range reads.

    A task-level summary.csv takes precedence. Older archives without that file fall back to the
    per-configuration postprocessing.json files, which contain the official cross-validation Dice.
    """
    reader = HTTPRangeReader(model["url"], int(model["size_bytes"]))
    try:
        with zipfile.ZipFile(reader) as archive:
            return _model_selections_from_archive(archive, model["task"])
    finally:
        reader.close()


def inspect_model_selections(registry: dict, output_path: Path, tasks: set[str] | None = None) -> Path:
    rows: list[dict] = []
    for model in registry["models"]:
        if tasks is not None and model["task"] not in tasks:
            continue
        try:
            selections = fetch_model_selection_summary(model)
        except FileNotFoundError as error:
            rows.append(
                {
                    "task": model["task"],
                    "configuration": "",
                    "official_validation_average": "",
                    "selected_primary": "",
                    "status": "summary_unavailable",
                    "note": str(error),
                    "archive_md5": model["md5"],
                    "archive_url": model["url"],
                }
            )
            continue
        best_average = max(float(selection["average"]) for selection in selections)
        for selection in selections:
            average = float(selection["average"])
            source = selection.get("selection_source", "summary.csv")
            rows.append(
                {
                    "task": model["task"],
                    "configuration": selection["model"],
                    "official_validation_average": average,
                    "selected_primary": average == best_average,
                    "status": "available" if source == "summary.csv" else "available_postprocessing_fallback",
                    "note": "" if source == "summary.csv" else (
                        "Official summary.csv is absent; selection derived before RankSEG evaluation from the "
                        "final cross-validation Dice in each official postprocessing.json."
                    ),
                    "archive_md5": model["md5"],
                    "archive_url": model["url"],
                }
            )
    if not rows:
        raise ValueError("No registered models matched the requested tasks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path

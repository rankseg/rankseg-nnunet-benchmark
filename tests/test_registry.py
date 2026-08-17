import io
import hashlib
import json
import zipfile

import pytest
import requests

from rankseg_nnunet_bench.registry import (
    _download_range,
    _model_selections_from_archive,
    _partition_byte_ranges,
    download_file_parallel,
    download_dataset,
    read_first_zip_member,
    s3_multipart_etag,
)


def test_partition_byte_ranges_is_contiguous_and_exact():
    ranges = _partition_byte_ranges(10, 20, 4)
    assert ranges == [(10, 12), (13, 15), (16, 18), (19, 20)]
    assert sum(end - start + 1 for start, end in ranges) == 11


def test_s3_multipart_etag_uses_binary_part_digests(tmp_path):
    path = tmp_path / "object.bin"
    path.write_bytes(b"abcdefghij")
    part_digests = [hashlib.md5(part).digest() for part in (b"abcd", b"efgh", b"ij")]
    expected = f"{hashlib.md5(b''.join(part_digests)).hexdigest()}-3"

    assert s3_multipart_etag(path, part_size=4) == expected


def test_parallel_download_accepts_publisher_s3_multipart_etag(tmp_path, monkeypatch):
    payload = b"abcdefghijkl"
    destination = tmp_path / "archive.bin"

    def fake_download(url, path, start, end, chunk_size):
        path.write_bytes(payload[start : end + 1])
        return path

    monkeypatch.setattr("rankseg_nnunet_bench.registry._download_range", fake_download)
    part_digests = [hashlib.md5(payload[index : index + 4]).digest() for index in range(0, 12, 4)]
    expected_etag = f"{hashlib.md5(b''.join(part_digests)).hexdigest()}-3"

    download_file_parallel(
        urls=["https://example.invalid/archive"],
        destination=destination,
        expected_s3_etag=expected_etag,
        s3_part_size=4,
        expected_size=len(payload),
        connections=3,
        chunk_size=2,
    )

    assert destination.read_bytes() == payload


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_read_first_zip_member_without_central_directory(compression):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("ensembles/Task999_Test/summary.csv", "model,average\n2d,0.7\n3d,0.8\n")
        archive.writestr("large-checkpoint.bin", b"not needed")
    archive_bytes = stream.getvalue()
    central_directory = archive_bytes.index(b"PK\x01\x02")

    filename, content = read_first_zip_member(archive_bytes[:central_directory])

    assert filename == "ensembles/Task999_Test/summary.csv"
    assert content.decode().endswith("3d,0.8\n")


def test_model_selection_prefers_official_summary_csv():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("ensembles/Task999_Test/summary.csv", "model,average\n2d,0.7\n3d,0.8\n")
        archive.writestr(
            "3d/Task999_Test/Trainer/postprocessing.json",
            json.dumps({"for_which_classes": [], "dc_per_class_raw": {"1": 0.9}}),
        )
    with zipfile.ZipFile(stream) as archive:
        rows = _model_selections_from_archive(archive, "Task999_Test")

    assert [(row["model"], row["average"]) for row in rows] == [("2d", "0.7"), ("3d", "0.8")]
    assert {row["selection_source"] for row in rows} == {"summary.csv"}


def test_model_selection_falls_back_to_official_postprocessing_scores():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "3d_fullres/Task999_Test/Trainer/postprocessing.json",
            json.dumps(
                {
                    "for_which_classes": [],
                    "dc_per_class_raw": {"1": 0.7, "2": 0.5},
                    "dc_per_class_pp_per_class": {"1": 0.1, "2": 0.1},
                }
            ),
        )
        archive.writestr(
            "ensembles/Task999_Test/ensemble_2d--3d/postprocessing.json",
            json.dumps(
                {
                    "for_which_classes": [[1]],
                    "dc_per_class_raw": {"1": 0.2, "2": 0.2},
                    "dc_per_class_pp_per_class": {"1": 0.8, "2": 0.6},
                    "dc_per_class_pp_all": {"1": 0.9, "2": 0.7},
                }
            ),
        )
    with zipfile.ZipFile(stream) as archive:
        rows = _model_selections_from_archive(archive, "Task999_Test")

    scores = {row["model"]: float(row["average"]) for row in rows}
    assert scores == {"3d_fullres": 0.6, "ensemble_2d--3d": 0.8}
    assert {row["selection_source"] for row in rows} == {"postprocessing.json"}


def test_parallel_dataset_download_requires_registered_size(tmp_path):
    dataset = {"task": "Task999_Test", "url": "https://example.invalid/data.tar", "md5": "0" * 32}

    with pytest.raises(ValueError, match="requires size_bytes"):
        download_dataset(dataset, tmp_path, connections=2)


def test_range_download_resumes_after_broken_connection(tmp_path, monkeypatch):
    calls = []

    class Response:
        status_code = 206

        def __init__(self, broken, content_range):
            self.broken = broken
            self.headers = {"Content-Range": content_range}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            if self.broken:
                yield b"ab"
                raise requests.exceptions.ChunkedEncodingError("broken")
            yield b"cdef"

    def fake_get(url, *, headers, stream, timeout):
        calls.append(headers["Range"])
        requested = headers["Range"].removeprefix("bytes=")
        return Response(broken=len(calls) == 1, content_range=f"bytes {requested}/6")

    monkeypatch.setattr("rankseg_nnunet_bench.registry.requests.get", fake_get)
    monkeypatch.setattr("rankseg_nnunet_bench.registry.time.sleep", lambda _: None)
    path = tmp_path / "range.part"

    _download_range("https://example.invalid/archive", path, 0, 5, 2)

    assert path.read_bytes() == b"abcdef"
    assert calls == ["bytes=0-5", "bytes=2-5"]


def test_range_download_never_writes_past_requested_end(tmp_path, monkeypatch):
    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-5/10"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"abcdefEXTRA"

    monkeypatch.setattr(
        "rankseg_nnunet_bench.registry.requests.get", lambda *args, **kwargs: Response()
    )
    path = tmp_path / "range.part"

    _download_range("https://example.invalid/archive", path, 0, 5, 2)

    assert path.read_bytes() == b"abcdef"


def test_parallel_resume_subdivides_the_only_incomplete_partition(tmp_path, monkeypatch):
    payload = b"abcdefghijkl"
    destination = tmp_path / "archive.bin"
    first = tmp_path / "archive.bin.part.range-0-5"
    second = tmp_path / "archive.bin.part.range-6-11"
    first.write_bytes(payload[:2])
    second.write_bytes(payload[6:])
    calls = []

    def fake_download(url, path, start, end, chunk_size):
        calls.append((start, end))
        path.write_bytes(payload[start : end + 1])
        return path

    monkeypatch.setattr("rankseg_nnunet_bench.registry._download_range", fake_download)

    download_file_parallel(
        urls=["https://example.invalid/archive"],
        destination=destination,
        expected_md5=hashlib.md5(payload).hexdigest(),
        expected_size=len(payload),
        connections=2,
        chunk_size=2,
    )

    assert destination.read_bytes() == payload
    assert calls == [(2, 3), (4, 5)]

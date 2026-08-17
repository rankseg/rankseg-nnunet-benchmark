from pathlib import Path

from rankseg_nnunet_bench.evidence import verify_evidence


def test_checked_in_full16_evidence_is_self_consistent():
    repository_root = Path(__file__).resolve().parents[1]
    assert verify_evidence(repository_root / "evidence").is_file()

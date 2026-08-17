from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate_summaries
from .config import load_dataset_config
from .evidence import publish_evidence, verify_evidence
from .evaluation import evaluate_dataset
from .oof import prepare_oof_ensemble_v1, prepare_oof_v1
from .registry import (
    dataset_by_task,
    download_dataset,
    download_model,
    inspect_model_selections,
    load_registry,
    model_by_task,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rankseg-nnunet-bench",
        description="Paired argmax versus RankSEG benchmark on cached nnU-Net probabilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one dataset manifest")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument(
        "--case-limit",
        type=int,
        help="Smoke test only. Limited runs are marked non-scientific and cannot be aggregated.",
    )

    aggregate = subparsers.add_parser("aggregate", help="Aggregate completed dataset summaries")
    aggregate.add_argument("summaries", type=Path, nargs="+")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument(
        "--overall-tests",
        action="store_true",
        help="Include dataset-level sign and Wilcoxon tests; enable only for the final main benchmark.",
    )

    inventory = subparsers.add_parser("inventory", help="List registered official nnU-Net models")
    inventory.add_argument("registry", type=Path)

    inspect = subparsers.add_parser(
        "inspect-selections",
        help="Read official validation summaries from model archive prefixes before benchmarking",
    )
    inspect.add_argument("registry", type=Path)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--tasks", nargs="+")

    download = subparsers.add_parser("download-model", help="Download and checksum one registered model")
    download.add_argument("registry", type=Path)
    download.add_argument("task")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--connections", type=int, default=1)

    download_data = subparsers.add_parser("download-dataset", help="Download and checksum one registered dataset")
    download_data.add_argument("registry", type=Path)
    download_data.add_argument("task")
    download_data.add_argument("--output-dir", type=Path, required=True)
    download_data.add_argument("--connections", type=int, default=1)

    oof = subparsers.add_parser("prepare-oof-v1", help="Prepare leakage-free fold-specific v1 inference inputs")
    oof.add_argument("--task", required=True)
    oof.add_argument("--images-dir", type=Path, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    oof.add_argument("--splits-file", type=Path)
    oof.add_argument("--model", default="3d_fullres")
    oof.add_argument("--trainer")
    oof.add_argument("--plans")
    oof.add_argument("--nifti-save-threads", type=int, default=2)

    ensemble_oof = subparsers.add_parser(
        "prepare-oof-ensemble-v1",
        help="Prepare fold-matched probability ensembling for two v1 OOF configurations",
    )
    ensemble_oof.add_argument("--first-oof-dir", type=Path, required=True)
    ensemble_oof.add_argument("--second-oof-dir", type=Path, required=True)
    ensemble_oof.add_argument("--output-dir", type=Path, required=True)
    ensemble_oof.add_argument("--threads", type=int, default=2)
    ensemble_oof.add_argument(
        "--postprocessing-file",
        type=Path,
        help="Official nnU-Net postprocessing.json applied to native masks; saved softmax remains raw",
    )

    publish = subparsers.add_parser(
        "publish-evidence",
        help="Normalize local results into a small, checksummed public evidence package",
    )
    publish.add_argument("manifest", type=Path)
    publish.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-evidence",
        help="Verify checksums, schemas, case counts, and the rebuilt aggregate",
    )
    verify.add_argument("evidence_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        path = evaluate_dataset(
            load_dataset_config(args.manifest),
            device_override=args.device,
            case_limit=args.case_limit,
            output_dir_override=args.output_dir,
        )
    elif args.command == "aggregate":
        path = aggregate_summaries(
            args.summaries, args.output_dir, include_overall_tests=args.overall_tests
        )
    elif args.command == "inventory":
        registry = load_registry(args.registry)
        print(f"source: {registry['source']['doi']} ({registry['source']['nnunet_version']})")
        for model in registry["models"]:
            gib = int(model["size_bytes"]) / 1024**3
            print(f"{model['task']:<43} {gib:>5.2f} GiB  {model['cohort']}")
        return 0
    elif args.command == "inspect-selections":
        path = inspect_model_selections(
            load_registry(args.registry),
            args.output,
            set(args.tasks) if args.tasks else None,
        )
    elif args.command == "download-model":
        registry = load_registry(args.registry)
        path = download_model(
            model_by_task(registry, args.task),
            args.output_dir,
            connections=args.connections,
        )
    elif args.command == "download-dataset":
        registry = load_registry(args.registry)
        path = download_dataset(
            dataset_by_task(registry, args.task),
            args.output_dir,
            connections=args.connections,
        )
    elif args.command == "prepare-oof-v1":
        path = prepare_oof_v1(
            task=args.task,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            splits_file=args.splits_file,
            model=args.model,
            trainer=args.trainer,
            plans=args.plans,
            nifti_save_threads=args.nifti_save_threads,
        )
    elif args.command == "prepare-oof-ensemble-v1":
        path = prepare_oof_ensemble_v1(
            first_oof_dir=args.first_oof_dir,
            second_oof_dir=args.second_oof_dir,
            output_dir=args.output_dir,
            threads=args.threads,
            postprocessing_file=args.postprocessing_file,
        )
    elif args.command == "publish-evidence":
        path = publish_evidence(args.manifest, args.output_dir)
    elif args.command == "verify-evidence":
        path = verify_evidence(args.evidence_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

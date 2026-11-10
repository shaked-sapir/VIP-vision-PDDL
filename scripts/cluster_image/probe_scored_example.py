"""Probe: the one-shot example's assistant turn as a complete scored list.

The system prompt asks for every predicate with a 0/1/2 score, while the
example turn lists only the true facts. This runs the pipeline with the example
rewritten as ``<predicate>: 2`` for the true facts and ``: 0`` for every other
grounded predicate, on a few staged problems, and prints realised rates.

    VIP_LLM_BASE_URL=http://<node>:8000/v1 python -m scripts.cluster_image.probe_scored_example \\
        --vendor vllm blocksworld:problem1 depot:problem0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from benchmark.data_generator import (
    _DOMAIN_REGISTRY, _apply_transform, _cleanup_external_source_files,
    _run_generation_inference, build_frame_classifier,
)
from benchmark.experiment_running_helpers.gt_builder import export_gt_from_problem_dir
from benchmark.vlm_precheck import _rates, _usage
from src.utils.config import load_config

FRAMES = {
    "depot": "src/domains/depot/problems", "gripper": "src/domains/gripper/problems",
}


def _frames_root(domain: str) -> Path:
    return Path(FRAMES.get(domain, f"benchmark/data/{domain}/precheck_frames__seed7/training/trajectories"))


def run(domain: str, problem: str, vendor: str, out_base: Path) -> dict:
    registry = _DOMAIN_REGISTRY[domain]
    config = load_config()
    dcfg = config["domains"][registry["config_key"]]
    domain_file = Path(dcfg["domain_file"])
    source = _frames_root(domain) / problem
    experiment = out_base / domain / f"probe_scored_example__{vendor}"
    target = experiment / "training" / "trajectories" / problem
    gt_root = experiment / "gt_trajectories"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    handler = registry["handler_class"](
        domain_name=dcfg.get("gym_domain_name", registry["config_key"]),
        pddl_domain_file=domain_file, vendor=vendor,
        frame_classifier=build_frame_classifier(1),
    )
    handler._set_seq_idx_format(target)
    handler.init_visual_components(Path(handler.get_image_path_by_index(target, 0)))
    classifier = handler.fluent_classifier
    image, true_facts = classifier.fewshot_examples[0]
    universe = sorted(classifier._generate_all_possible_predicates())
    scored = [f"{p}: {2 if p in set(true_facts) else 0}" for p in universe]
    classifier.fewshot_examples = [(image, scored)]
    print(f"[{domain}/{problem}] example rewritten: {len(true_facts)} true of {len(universe)} predicates", flush=True)

    frames = len(list(target.glob("*.png")))
    started = time.perf_counter()
    if registry["is_external"]:
        handler.run_pipeline(problem_name=problem, images_path=target)
        export_gt_from_problem_dir(target, gt_root, problem, needs_schema_translation=False)
    else:
        _run_generation_inference(handler, problem, target, gt_root)
        _apply_transform(registry["transform_fn"], target)
    _cleanup_external_source_files(target)
    row = {"frames": frames, "seconds": round(time.perf_counter() - started, 1), **_usage(handler)}
    row.update(_rates(target, domain_file, gt_root))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", help="domain:problem")
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--out-base", type=Path, default=Path("benchmark/data"))
    args = parser.parse_args()
    results = {}
    for target in args.targets:
        domain, problem = target.split(":")
        try:
            results[target] = run(domain, problem, args.vendor, args.out_base)
        except Exception as error:
            results[target] = {"error": f"{type(error).__name__}: {error}"}
        print(target, json.dumps(results[target]), flush=True)
    stamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    out = args.out_base / f"probe_scored_example__{args.vendor}__{stamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()

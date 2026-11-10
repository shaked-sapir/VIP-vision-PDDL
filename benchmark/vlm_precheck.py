"""Run one VLM over a staged set of frames and measure what it produced.

The frames are staged once (``data_generator.py --gen-mode generate
--no-inference`` for a PDDLGym domain; ``src/domains/<d>/problems`` for an
external one) so that every model classifies the same images, and the
realised mask and flip rates against the GT trajectories are comparable
across models::

    python -m benchmark.vlm_precheck --domain hanoi --vendor vllm \\
        --frames-root benchmark/data/hanoi/precheck_frames__seed7/training/trajectories
    python -m benchmark.vlm_precheck --domain depot --vendor openai \\
        --frames-root src/domains/depot/problems --problems 5

Output: ``benchmark/data/<domain>/<label>__model=<model>__<ts>/`` in the
corpus layout (``training/trajectories/``, ``gt_trajectories/``) plus
``precheck_summary.json`` with per-problem rates, wall time, frame count and
prompt-cache usage.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.algorithm_adapters.lamanna.realised_rates import (
    gt_trajectory_lookup,
    realised_rates,
)
from benchmark.data_generator import (
    _DOMAIN_REGISTRY,
    _apply_transform,
    _cleanup_external_source_files,
    _example_dir,
    _protocol_tag,
    _run_generation_inference,
    build_frame_classifier,
)
from benchmark.experiment_running_helpers.gt_builder import export_gt_from_problem_dir
from src.utils.config import load_config


def _problem_dirs(frames_root: Path, limit: Optional[int]) -> List[Path]:
    """The problem folders under ``frames_root``, numerically ordered."""
    dirs = [p for p in frames_root.iterdir() if p.is_dir() and list(p.glob("*.pddl"))]
    if not dirs:
        raise FileNotFoundError(f"no problem folders with a .pddl under {frames_root}")
    ordered = sorted(dirs, key=lambda p: int(re.search(r"\d+$", p.name).group()))
    return ordered[:limit] if limit else ordered


def _usage(handler) -> Dict[str, float]:
    """Prompt-cache counters of the classifier's backend, when it keeps them."""
    backend = getattr(getattr(handler, "fluent_classifier", None), "backend", None)
    return {
        "prompt_tokens_total": getattr(backend, "prompt_tokens_total", 0),
        "prompt_tokens_cached": getattr(backend, "prompt_tokens_cached", 0),
        "cache_hit_ratio": getattr(backend, "cache_hit_ratio", 0.0),
    }


def _infer_one(
    registry: dict, domain_file: Path, domain_name: str, vendor: str,
    workers: int, source: Path, target: Path, gt_root: Path,
    example_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Copy one staged problem into ``target`` and classify its frames there."""
    shutil.copytree(source, target, dirs_exist_ok=True)
    handler = registry["handler_class"](
        domain_name=domain_name,
        pddl_domain_file=domain_file,
        vendor=vendor,
        frame_classifier=build_frame_classifier(workers),
        example_dir=example_dir,
    )
    frames = len(list(target.glob("*.png")))
    started = time.perf_counter()
    if registry["is_external"]:
        handler.run_pipeline(problem_name=target.name, images_path=target)
        export_gt_from_problem_dir(target, gt_root, target.name, needs_schema_translation=False)
    else:
        _run_generation_inference(handler, target.name, target, gt_root)
        _apply_transform(registry["transform_fn"], target)
    _cleanup_external_source_files(target)
    elapsed = time.perf_counter() - started
    return {"frames": frames, "seconds": round(elapsed, 1), **_usage(handler)}


def _rates(target: Path, domain_file: Path, gt_root: Path) -> Dict[str, object]:
    """Realised mask and flip rates of one classified problem against its GT."""
    name = target.name
    prepared = [(target / f"{name}.trajectory", target / f"{name}.masking_info", target / f"{name}.pddl")]
    return realised_rates(prepared, domain_file, gt_trajectory_lookup(gt_root.parent)).as_dict()


def run(domain: str, vendor: str, frames_root: Path, problems: Optional[int],
        workers: int, output_base: Path, label: str = "precheck",
        own_frame0: bool = False) -> Path:
    """Classify the staged problems with ``vendor``'s model; return the output dir.

    Args:
        label: Output directory prefix, naming the prompt protocol of the run.
        own_frame0: Use each problem's own frame 0 as the example even when
            config.yaml names a shared ``example_dir`` for the domain.
    """
    registry = _DOMAIN_REGISTRY[domain]
    config = load_config()
    domain_config = config["domains"][registry["config_key"]]
    domain_file = Path(domain_config["domain_file"])
    domain_name = domain_config.get("gym_domain_name", registry["config_key"])
    model = config[vendor].get("fluent_classification_model", {}).get("model_name", vendor)
    example_dir = None if own_frame0 else _example_dir(domain_config)

    stamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    experiment_dir = output_base / domain / f"{label}__model={model}{_protocol_tag(example_dir)}__{stamp}"
    trajectories_dir = experiment_dir / "training" / "trajectories"
    gt_root = experiment_dir / "gt_trajectories"
    trajectories_dir.mkdir(parents=True)

    summary: Dict[str, object] = {
        "domain": domain, "vendor": vendor, "model": model, "label": label,
        "frames_root": str(frames_root), "inference_workers": workers,
        "example_dir": str(example_dir) if example_dir else None, "problems": {},
    }
    for source in _problem_dirs(frames_root, problems):
        target = trajectories_dir / source.name
        print(f"[{domain}/{source.name}] classifying with {model} ...", flush=True)
        try:
            row = _infer_one(registry, domain_file, domain_name, vendor, workers,
                             source, target, gt_root, example_dir)
            row.update(_rates(target, domain_file, gt_root))
        except Exception as error:  # one failed problem must not hide the others
            row = {"error": f"{type(error).__name__}: {error}"}
            print(f"  failed: {row['error']}", flush=True)
        summary["problems"][source.name] = row
        (experiment_dir / "precheck_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  {json.dumps(row)}", flush=True)
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, choices=sorted(_DOMAIN_REGISTRY))
    parser.add_argument("--vendor", required=True, help="a vendor block of config.yaml")
    parser.add_argument("--frames-root", type=Path, required=True,
                        help="directory of staged problem folders (images + GT + .pddl)")
    parser.add_argument("--problems", type=int, default=None, help="first N problems only")
    parser.add_argument("--inference-workers", type=int, default=1)
    parser.add_argument("--output-base", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--label", default="precheck",
                        help="output directory prefix, naming the prompt protocol")
    parser.add_argument("--own-frame0", action="store_true",
                        help="ignore the domain's shared example_dir; the example "
                             "is each problem's own frame 0")
    args = parser.parse_args()
    out = run(args.domain, args.vendor, args.frames_root, args.problems,
              args.inference_workers, args.output_base, args.label, args.own_frame0)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

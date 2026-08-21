"""ROSAME-I runner (ICAPS-24): joint CV state predictor + ROSAME action model.

Unlike plain ROSAME (which encodes VLM-inferred symbolic states), ROSAME-I trains
a randomly-initialised ResNet-18 CV head *from the raw state images* jointly with
the ROSAME action schemas, using only the observed action sequence and the GT
final state as supervision. This is a faithful single-trace port of the reference
``train.py::run`` (branch ``main`` of the user's ROSAME clone); see
``docs/rosame-i-implementation-plan.md`` for the design and the decisions behind
the raw-logit CV head, device handling, and the two training schedules.

Subclasses :class:`PORosame_Runner` to inherit the build-once / re-ground-after
``add_problem`` fix (see plan §8.5). Its ``prepare_rosame_data`` (0.5-masking
encoder) is unused here and harmless.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner


# Paper synth/resnet preprocessing. ICAPS-24 ``train.py:211/226/237`` uses
# ``transforms.Resize(64)`` -- an *int*, which scales the shorter edge and
# **preserves aspect ratio**. ``Resize((64, 64))`` is a different operation: it
# forces a square and distorts (hanoi 630x224 -> 2.81x horizontal squeeze,
# gripper/depot 800x600 -> 1.33x). We follow upstream. See
# ``docs/rosame-i-milp-26-implementation-plan.md`` §4.6.1.
_DEFAULT_RESIZE: int = 64


def build_image_tf(
    resize: Union[int, Sequence[int], None] = _DEFAULT_RESIZE,
) -> transforms.Compose:
    """Build the image transform for a given resize setting.

    Args:
        resize: ``int`` -> ``Resize(n)``, shorter edge, aspect preserved (the
            ICAPS-24 form, and the default). A 2-sequence ``(h, w)`` ->
            ``Resize((h, w))``, forced, aspect distorted -- note torchvision's
            order is **(height, width)**. ``None`` -> no resize, native size.

    Returns:
        A ``transforms.Compose`` ending in ``ToTensor()`` (float in ``[0, 1]``).
    """
    steps = []
    if resize is not None:
        if isinstance(resize, int):
            steps.append(transforms.Resize(resize))
        else:
            try:
                hw = tuple(int(v) for v in resize)
            except TypeError:
                # Guard the sentinel-leaked-through-pickle class of bug: without
                # this the failure surfaces as "'object' object is not iterable"
                # from inside a comprehension, 3 frames from the cause.
                raise TypeError(
                    f"resize must be an int, a 2-sequence or None; got "
                    f"{resize!r} of type {type(resize).__name__}"
                ) from None
            if len(hw) != 2:
                raise ValueError(f"resize must be an int, a 2-sequence or None; got {resize!r}")
            steps.append(transforms.Resize(hw))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    """Autodetect cuda > mps > cpu, or honour an explicit override."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs for a reproducible per-seed model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_cv_model(n_props: int) -> nn.Module:
    """Random-init ResNet-18 + MLP head → raw proposition logits (paper synth path)."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, n_props),
    )
    return model


class _PreparedTrace:
    """Per-trajectory training tensors, all pinned to the model device."""

    def __init__(
        self,
        images: torch.Tensor,          # (T+1, 3, H, W) -- H,W set by ``resize``
        action_indices: List[int],     # length T, ROSAME action indices
        final_state_vec: torch.Tensor, # (n_props,) GT final state, {0,1}
        name: str,
    ) -> None:
        self.images = images
        self.action_indices = action_indices
        self.final_state_vec = final_state_vec
        self.name = name


class RosameI_Runner(PORosame_Runner):
    """Joint CV + ROSAME learner with per-trajectory and pooled training loops."""

    def __init__(
        self,
        domain_file,
        device: Optional[str | torch.device] = None,
        seed: int = 8800,
        lr_schema: float = 1e-3,
        lr_cv: float = 1e-3,
        resize: Union[int, Sequence[int], None] = _DEFAULT_RESIZE,
    ) -> None:
        super().__init__(domain_file)
        self.device = _resolve_device(device)
        self.seed = seed
        self.lr_schema = lr_schema
        self.lr_cv = lr_cv
        self.resize = resize
        self._image_tf = build_image_tf(resize)
        self.cv_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._proposition_signature: Optional[Tuple[str, ...]] = None

    # ------------------------------------------------------------------ grounding

    def ground_union(self, problems: Sequence[object]) -> None:
        """Ground the domain once on the union of every problem's objects.

        Builds the ROSAME schemas from the first problem (inherited build-once
        ``add_problem``), then re-grounds on the union and builds the CV head.
        """
        if not problems:
            raise ValueError("ground_union requires at least one problem")
        self.add_problem(problems[0])
        self.objects = self._union_objects(problems)
        self.rosame.ground_from_dict(self.objects)
        self._ensure_cv_model()

    def _union_objects(self, problems: Sequence[object]) -> Dict[str, List[str]]:
        """Merge each problem's ``{type: [object]}`` map, preserving first-seen order."""
        union: Dict[str, List[str]] = {}
        for problem in problems:
            for type_name, names in self.get_objects(problem.objects).items():
                bucket = union.setdefault(type_name, [])
                for name in names:
                    if name not in bucket:
                        bucket.append(name)
        return union

    def _ensure_cv_model(self) -> None:
        """Create the CV head on the first grounding; assert the grounding is stable."""
        props = tuple(self.rosame.propositions.keys())
        if self.cv_model is None:
            self._proposition_signature = props
            self.cv_model = _build_cv_model(len(props)).to(self.device)
            self.cv_model.train()
            self._build_optimizer()
        elif props != self._proposition_signature:
            raise RuntimeError(
                "the grounding changed after the CV head was built: "
                f"{len(props)} vs {len(self._proposition_signature)} propositions"
            )

    def _build_optimizer(self) -> None:
        """Adam over schema MLP params (CPU) + CV params (device); mixed-device ok."""
        param_groups = [
            {"params": schema.parameters(), "lr": self.lr_schema}
            for schema in self.rosame.action_schemas
        ]
        param_groups.append({"params": self.cv_model.parameters(), "lr": self.lr_cv})
        self.optimizer = torch.optim.Adam(param_groups)

    # ------------------------------------------------------------------ data prep

    def _load_image(self, path: Path) -> torch.Tensor:
        """PNG → (3, H, W) float tensor (PIL; torchvision.io is broken in venv11).

        ``H, W`` follow ``self.resize``; the ResNet head is resolution-agnostic
        (``AdaptiveAvgPool2d`` before ``fc``), so non-square input is fine.
        """
        with Image.open(path) as img:
            return self._image_tf(img.convert("RGB"))

    def _encode_state(self, positive_predicate_strings: Sequence[str]) -> torch.Tensor:
        """Binary vector over ``rosame.propositions`` (matched positives → 1, CWA → 0)."""
        matched = set()
        for pred_str in positive_predicate_strings:
            prop = self.check_predicate(pred_str)
            if prop is not None:
                matched.add(prop)
        vec = [1.0 if p in matched else 0.0 for p in self.rosame.propositions]
        return torch.tensor(vec, dtype=torch.float32)

    def _prepare_trace(
        self,
        image_paths: Sequence[Path],
        action_strings: Sequence[str],
        final_state_predicates: Sequence[str],
        name: str,
    ) -> Optional[_PreparedTrace]:
        """Build a :class:`_PreparedTrace`, or ``None`` (skip) on any misalignment."""
        action_indices: List[int] = []
        for action_str in action_strings:
            idx = self.check_action(action_str)
            if idx is None:
                print(
                    f"  [ROSAME-I] skipping trajectory {name}: action "
                    f"'{action_str}' does not map to the shared grounding"
                )
                return None
            action_indices.append(idx)

        num_actions = len(action_indices)
        if len(image_paths) != num_actions + 1:
            print(
                f"  [ROSAME-I] skipping trajectory {name}: {len(image_paths)} images "
                f"!= T+1={num_actions + 1}"
            )
            return None

        images = torch.stack([self._load_image(p) for p in image_paths]).to(self.device)
        final_vec = self._encode_state(final_state_predicates).to(self.device)
        return _PreparedTrace(images, action_indices, final_vec, name)

    def prepare_traces(
        self,
        prepared_problems: Sequence[Tuple[object, Sequence[Path], Sequence[str], Sequence[str]]],
    ) -> List[_PreparedTrace]:
        """Union-ground on all problems, then build one trace per usable problem.

        Traces are named after their image directory; image-mode problem PDDLs do
        not carry unique ``(problem <name>)`` headers.
        """
        problems = [problem for problem, _images, _actions, _final in prepared_problems]
        self.ground_union(problems)

        traces: List[_PreparedTrace] = []
        for idx, (_problem, image_paths, action_strings, final_preds) in enumerate(
            prepared_problems
        ):
            name = image_paths[0].parent.name if image_paths else f"problem{idx}"
            trace = self._prepare_trace(image_paths, action_strings, final_preds, name)
            if trace is not None:
                traces.append(trace)
        return traces

    # ------------------------------------------------------------------ loss

    def _forward_predictions(
        self, trace: _PreparedTrace, augment: bool
    ) -> torch.Tensor:
        """One CV forward pass over a trace's frames → ``(T+1, n_props)`` raw logits."""
        images = trace.images
        if augment:  # per-image horizontal flip (blocksworld symmetry), prob 0.5
            flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
            if bool(flip_mask.any()):
                images = images.clone()
                images[flip_mask] = torch.flip(images[flip_mask], dims=[-1])
        return self.cv_model(images)

    def _trajectory_loss(
        self, trace: _PreparedTrace, gamma: float, lambda_: float, augment: bool
    ) -> torch.Tensor:
        """Single-trace ROSAME-I loss (faithful port of ``train.py::run``)."""
        preds = self._forward_predictions(trace, augment)
        return self._loss_from_predictions(preds, trace, gamma, lambda_)

    def _loss_from_predictions(
        self,
        preds: torch.Tensor,
        trace: _PreparedTrace,
        gamma: float,
        lambda_: float,
    ) -> torch.Tensor:
        """ROSAME-I loss for already-computed predictions.

        Consistency (t=1..T-1) + gamma-anchored GT final state + applicability
        + lambda-weighted precondition prior. RAW logits, ``reduction='sum'``.
        """
        pre, add, dele = self.rosame.build(trace.action_indices)  # CPU tensors
        pre = pre.to(self.device)
        add = add.to(self.device)
        dele = dele.to(self.device)

        # Predicted next state from each pre-state under the (soft) action model.
        domain_preds = preds[:-1] * (1 - dele) + (1 - preds[:-1]) * add  # (T, n_props)

        if domain_preds.shape[0] > 1:  # consistency: domain_preds[0..T-2] vs preds[1..T-1]
            loss = F.mse_loss(domain_preds[:-1], preds[1:-1], reduction="sum")
        else:
            loss = torch.zeros((), device=self.device)

        # gamma-anchor the LAST predicted state to the GT final state (decision #2).
        loss = loss + gamma * F.mse_loss(
            domain_preds[-1], trace.final_state_vec, reduction="sum"
        )
        # Applicability: a false pre-state fluent must not be a precondition.
        loss = loss + F.mse_loss(
            (1 - preds[:-1]) * pre, torch.zeros_like(pre), reduction="sum"
        )
        # Precondition prior (pushes preconditions toward 1).
        loss = loss + lambda_ * F.mse_loss(pre, torch.ones_like(pre), reduction="sum")
        return loss

    def _total_loss(
        self, traces: Sequence[_PreparedTrace], gamma: float, lambda_: float
    ) -> float:
        """Sum of per-trace losses (no grad, no augment) — the seed-selection metric."""
        self.cv_model.eval()
        total = 0.0
        with torch.no_grad():
            for trace in traces:
                total += self._trajectory_loss(trace, gamma, lambda_, augment=False).item()
        self.cv_model.train()
        return total

    # ------------------------------------------------------------------ loops

    def learn_pooled(
        self,
        traces: Sequence[_PreparedTrace],
        epochs: int,
        gamma: float,
        lambda_: float,
        augment: bool,
        timeout_check: Optional[Callable[[], bool]] = None,
    ) -> float:
        """Each epoch steps over every trace in a fresh random order.

        Mirrors ICAPS-24 ``train.py``, which pools all traces into one dataset
        and iterates ``DataLoader(trainset, args.batch_size, shuffle=True)`` once
        per epoch. One trace per optimizer step rather than a batch of them; the
        ordering is upstream's, the batch dimension is not.
        """
        order = list(range(len(traces)))
        for _ in range(epochs):
            random.shuffle(order)
            for i in order:
                self.optimizer.zero_grad()
                loss = self._trajectory_loss(traces[i], gamma, lambda_, augment)
                loss.backward()
                self.optimizer.step()
            if timeout_check is not None and timeout_check():
                break
        return self._total_loss(traces, gamma, lambda_)

    # ------------------------------------------------------------------ orchestration

    def learn_full(
        self,
        prepared_problems: Sequence[Tuple[object, Sequence[Path], Sequence[str], Sequence[str]]],
        epochs: int,
        gamma: float,
        lambda_: float,
        augment: bool,
        timeout_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[float]:
        """Ground, prepare all traces, and train them pooled.

        ``prepared_problems`` items are
        ``(problem, image_paths, action_strings, final_state_predicates)``.
        Returns the final total training loss, or ``None`` if no trace is usable.
        """
        _set_seed(self.seed)
        traces = self.prepare_traces(prepared_problems)
        if not traces:
            return None
        return self.learn_pooled(traces, epochs, gamma, lambda_, augment, timeout_check)

    def to_pddl(self) -> str:
        """Threshold the learned schemas into a PDDL domain string."""
        return self.rosame_to_pddl()

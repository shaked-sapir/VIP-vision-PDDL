"""Unit tests for :mod:`src.milp.rosame26_training` — the §3 harness override.

    python -m pytest src/milp/test_rosame26_training.py

Three things are worth testing about a training loop that exists to be faithful
to another one.

**The parameters.** §6 pins fifteen values and stars four. A test that reads the
table back is the only thing that keeps a later edit from quietly unpinning one.

**The two gate sites.** §3: "they are one condition evaluated twice, and both
must move together". A recording stub repairer makes the epochs at which each
fires observable, so drift between them is a failure rather than a subtlety.

**What was deliberately dropped.** The augmentation block cannot be caught by a
behavioural test — on our data it raises rather than corrupts, except for the one
``gripper`` resize that silently permutes (§3). It is caught by asserting the
classes are not in the module's namespace at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp import rosame26_training
from src.milp.domain_assets import write_grounding_assets
from src.milp.rosame26_training import (
    DEVIATING_PARAMETERS,
    UPSTREAM_PARAMETERS,
    Rosame26Trainer,
    build_loader,
    default_parameters,
    is_mip_epoch,
    mip_epoch_count,
    resolve_batch_size,
    select_device,
)
from src.milp.trace_tensors import PaddedTraces

from dl.util.dataset import TraceDataset

DOMAIN = "blocksworld"
OBJECTS = {"block": ["a", "b", "c"]}
C, H, W = 3, 64, 64
T_MAX = 3
LENGTHS = torch.tensor([3, 1, 2])


# --------------------------------------------------------------------------- #
# parameters                                                                   #
# --------------------------------------------------------------------------- #


def test_upstream_parameters_are_train_commons_table() -> None:
    """§6, verbatim, with the sweep lists unwrapped."""
    assert UPSTREAM_PARAMETERS == {
        "aae_width": 1000,
        "aae_depth": 3,
        "feature_dim": 512,
        "hidden_dim": 256,
        "lambda": 0.2,
        "gamma": 10,
        "beta_pred": 1,
        "beta_app": 1,
        "beta_reconst": 0,
        "optimizer": "Adam",
        "lr": 1e-4,
        "mip_interval": 1,
        "mip_traces": 3,
        "pseudo_weight_decay": 0.99,
        "mip_time_limit": 60,
        "DL_to_MIP": ["state", "action", "model"],
    }


def test_upstream_parameters_hold_none_of_the_starred_four() -> None:
    """The four §6 stars are decided per run, never inherited by accident."""
    assert set(DEVIATING_PARAMETERS) == {
        "epoch",
        "batch_size",
        "MIP_to_DL",
        "cp_type",
    }
    assert not set(DEVIATING_PARAMETERS) & set(UPSTREAM_PARAMETERS)


def test_default_parameters_carry_the_pinned_table_through(tmp_path: Path) -> None:
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=tmp_path, epoch=10
    )
    for key, value in UPSTREAM_PARAMETERS.items():
        assert parameters[key] == value, key


def test_mip_to_dl_drops_action_while_dl_to_mip_keeps_it(tmp_path: Path) -> None:
    """Option B's asymmetry (§0): observed actions flow to the solver, not back."""
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=tmp_path, epoch=10
    )
    assert parameters["MIP_to_DL"] == ["state", "model"]
    assert parameters["DL_to_MIP"] == ["state", "action", "model"]


def test_default_parameters_omit_what_build_writes(tmp_path: Path) -> None:
    """``prop_dim`` / ``action_dim`` come from the domain model, not from us."""
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=tmp_path, epoch=10
    )
    assert "prop_dim" not in parameters
    assert "action_dim" not in parameters


def test_cp_type_is_cp_sat(tmp_path: Path) -> None:
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=tmp_path, epoch=10
    )
    assert parameters["cp_type"] == "cp-sat"


def test_select_device_excludes_mps() -> None:
    """§1.3: cuda else cpu. MPS has already produced one silent-crash class here."""
    assert select_device() in {"cuda", "cpu"}


def test_default_device_is_the_selected_one(tmp_path: Path) -> None:
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=tmp_path, epoch=10
    )
    assert parameters["device"] == select_device()


# --------------------------------------------------------------------------- #
# the gate and the loader                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "epoch, pre_mip, interval, expected",
    [
        (0, 0, 1, True),
        (0, 1, 1, False),
        (4, 2, 2, True),
        (5, 2, 2, False),
        (2, 2, 3, False),
        (3, 2, 3, True),
    ],
)
def test_is_mip_epoch_reads_both_conditions(
    epoch: int, pre_mip: int, interval: int, expected: bool
) -> None:
    parameters = {"pre_mip_epoch": pre_mip, "mip_interval": interval}
    assert is_mip_epoch(epoch, parameters) is expected


def test_pre_mip_epoch_at_the_budget_disables_the_mip() -> None:
    """§3's free bonus: this one parameter is the whole DL-only arm."""
    assert mip_epoch_count({"epoch": 50, "pre_mip_epoch": 50, "mip_interval": 1}) == 0
    assert mip_epoch_count({"epoch": 51, "pre_mip_epoch": 50, "mip_interval": 1}) == 1


def test_mip_epoch_count_counts_the_interval() -> None:
    assert mip_epoch_count({"epoch": 10, "pre_mip_epoch": 4, "mip_interval": 2}) == 3


@pytest.mark.parametrize("requested, n_traces, expected", [(128, 9, 9), (2, 9, 2)])
def test_resolve_batch_size_caps_at_the_fold(
    requested: int, n_traces: int, expected: int
) -> None:
    assert resolve_batch_size(requested, n_traces) == expected


def test_resolve_batch_size_rejects_an_empty_fold() -> None:
    with pytest.raises(ValueError, match="no traces"):
        resolve_batch_size(128, 0)


def _order(seed: int, n: int = 8) -> List[int]:
    data = (torch.arange(n).float(), torch.arange(n).float(), torch.arange(n).float())
    loader = build_loader(TraceDataset(data), batch_size=2, num_workers=0, seed=seed)
    return [int(index) for _, _, _, batch in loader for index in batch]


def test_the_shuffle_is_reproducible_from_the_seed() -> None:
    """The FIFO ``TraceSelector`` fills from the first batches, so order matters."""
    assert _order(0) == _order(0)


def test_a_different_seed_gives_a_different_order() -> None:
    assert _order(0) != _order(1)


def test_no_augmentation_transform_is_reachable() -> None:
    """§3 / deviation 5: the classes are not imported, so none can be selected."""
    for name in (
        "ColumnPermute",
        "ItemPermute",
        "RoomBallPermute",
        "BlocksworldPilePermute",
        "TraceResize",
    ):
        assert not hasattr(rosame26_training, name), name


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class RecordingRepairer:
    """A :class:`MipRepairer` that records the epoch of every call.

    ``clear`` runs once at the head of each epoch, before anything else, so the
    number of clears so far minus one *is* the current epoch. That is how the
    stub knows which epoch it is in without the loop telling it.
    """

    distance: Optional[float] = None
    pseudo_labels: Any = None
    clears: int = 0
    updates: List[int] = field(default_factory=list)
    solves: List[int] = field(default_factory=list)

    @property
    def epoch(self) -> int:
        return self.clears - 1

    def clear(self) -> None:
        self.clears += 1

    def update(self, indices, z, a, inits, goals) -> None:
        self.updates.append(self.epoch)

    def run_fixer(self, time_limit: float) -> Optional[float]:
        self.solves.append(self.epoch)
        return self.distance


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rosame26_training_assets")
    write_grounding_assets(DOMAIN, OBJECTS, root)
    return root


def make_trainer(
    assets: Path,
    name: str,
    *,
    epoch: int = 2,
    pre_mip_epoch: Optional[int] = None,
    seed: int = 0,
    repairer: Optional[RecordingRepairer] = None,
) -> Rosame26Trainer:
    """A trainer on ``assets``, DL-only unless a repairer is given."""
    parameters = default_parameters(
        domain=DOMAIN,
        domain_assets_root=assets,
        epoch=epoch,
        pre_mip_epoch=epoch if pre_mip_epoch is None else pre_mip_epoch,
        device="cpu",
        seed=seed,
    )
    return Rosame26Trainer(assets / name, parameters, mip_repairer=repairer)


def make_fold(n_props: int, n_actions: int, seed: int = 0) -> PaddedTraces:
    """A right-padded three-trace fold with lengths ``[3, 1, 2]``."""
    generator = torch.Generator().manual_seed(seed)
    batch = len(LENGTHS)
    images = torch.rand(batch, T_MAX, C, H, W, generator=generator)
    states = torch.randint(
        0, 2, (batch, T_MAX + 2, n_props), generator=generator
    ).float()
    actions = torch.zeros(batch, T_MAX + 1, n_actions)
    for row, length in enumerate(LENGTHS.tolist()):
        for step in range(length + 1):
            actions[row, step, (row + step) % n_actions] = 1.0
    return PaddedTraces(images, states, actions, LENGTHS.clone())


@pytest.fixture(scope="module")
def fold(assets: Path) -> PaddedTraces:
    parameters = default_parameters(
        domain=DOMAIN, domain_assets_root=assets, epoch=1, pre_mip_epoch=1, device="cpu"
    )
    probe = Rosame26Trainer(assets / "dims", parameters)
    probe.build([T_MAX, C, H, W])
    return make_fold(probe.parameters["prop_dim"], probe.parameters["action_dim"])


def test_a_dl_only_run_needs_no_repairer(assets: Path, fold: PaddedTraces) -> None:
    trainer = make_trainer(assets, "dl_only")
    history = trainer.train(fold)

    assert [record["epoch"] for record in history] == [0, 1]
    assert set(history[0]) == {
        "total_loss",
        "loss_pred",
        "loss_app",
        "loss_reconst",
        "loss_prior",
        "epoch",
    }


def test_a_solving_schedule_without_a_repairer_raises(
    assets: Path, fold: PaddedTraces
) -> None:
    """And it raises before the first epoch, not at ``pre_mip_epoch``."""
    trainer = make_trainer(assets, "no_repairer", epoch=2, pre_mip_epoch=0)

    with pytest.raises(ValueError, match="no mip_repairer was given"):
        trainer.train(fold)

    assert trainer.history == []


def test_both_gate_sites_fire_on_the_same_epochs(
    assets: Path, fold: PaddedTraces
) -> None:
    """§3: one condition read twice. Selector update and solve cannot drift."""
    repairer = RecordingRepairer()
    trainer = make_trainer(
        assets, "gates", epoch=3, pre_mip_epoch=1, repairer=repairer
    )
    trainer.train(fold)

    assert repairer.updates == [1, 2]
    assert repairer.solves == repairer.updates


def test_clear_runs_once_per_epoch(assets: Path, fold: PaddedTraces) -> None:
    """Upstream clears unconditionally at ``network.py:256``, not only when solving."""
    repairer = RecordingRepairer()
    trainer = make_trainer(
        assets, "clears", epoch=3, pre_mip_epoch=2, repairer=repairer
    )
    trainer.train(fold)

    assert repairer.clears == 3
    assert len(repairer.solves) == 1


def test_mip_gt_dist_is_recorded_only_on_solving_epochs(
    assets: Path, fold: PaddedTraces
) -> None:
    repairer = RecordingRepairer(distance=1.25)
    trainer = make_trainer(
        assets, "dist", epoch=2, pre_mip_epoch=1, repairer=repairer
    )
    history = trainer.train(fold)

    assert "mip_gt_dist" not in history[0]
    assert history[1]["mip_gt_dist"] == pytest.approx(1.25)


def test_a_repairer_returning_none_records_no_distance(
    assets: Path, fold: PaddedTraces
) -> None:
    repairer = RecordingRepairer(distance=None)
    trainer = make_trainer(
        assets, "nodist", epoch=2, pre_mip_epoch=1, repairer=repairer
    )
    history = trainer.train(fold)

    assert "mip_gt_dist" not in history[1]


def test_the_dataset_is_built_without_a_transform(
    assets: Path, fold: PaddedTraces, monkeypatch
) -> None:
    """The images the loop trains on are the images it was handed."""
    seen: List[Any] = []
    original = rosame26_training.TraceDataset

    def capture(data, transforms=None):
        seen.append(transforms)
        return original(data, transforms=transforms)

    monkeypatch.setattr(rosame26_training, "TraceDataset", capture)
    make_trainer(assets, "notransform").train(fold)

    assert seen == [None]


def test_each_batch_gets_its_own_rows_lengths(
    assets: Path, fold: PaddedTraces
) -> None:
    """``lengths[indices]``, not ``lengths`` — the loader shuffles."""
    trainer = make_trainer(assets, "lengths")
    recorded: List[Tuple[torch.Tensor, torch.Tensor]] = []
    original = trainer.loss

    def spy(outputs, targets, indices=(), mip_pseudo_labels=None, lengths=None):
        recorded.append((indices.clone(), lengths.clone()))
        return original(outputs, targets, indices, mip_pseudo_labels, lengths)

    trainer.loss = spy
    trainer.train(fold)

    assert recorded
    for indices, lengths in recorded:
        assert torch.equal(lengths, fold.lengths[indices])


def test_the_whole_fold_is_one_batch_at_the_pinned_batch_size(
    assets: Path, fold: PaddedTraces
) -> None:
    """§1.1: ``min(batch_size, N)`` makes a 3-trace fold a single batch."""
    trainer = make_trainer(assets, "onebatch", epoch=2)
    calls: List[int] = []
    original = trainer.loss

    def spy(outputs, targets, indices=(), mip_pseudo_labels=None, lengths=None):
        calls.append(len(indices))
        return original(outputs, targets, indices, mip_pseudo_labels, lengths)

    trainer.loss = spy
    trainer.train(fold)

    assert calls == [len(LENGTHS), len(LENGTHS)]


def test_the_same_seed_reproduces_the_run(assets: Path, fold: PaddedTraces) -> None:
    """§1.3: weight init and shuffle both hang off ``seed``."""
    first = make_trainer(assets, "seed_a", seed=7).train(fold)
    second = make_trainer(assets, "seed_b", seed=7).train(fold)

    assert [row["total_loss"] for row in first] == [
        row["total_loss"] for row in second
    ]


def test_a_different_seed_gives_a_different_run(
    assets: Path, fold: PaddedTraces
) -> None:
    first = make_trainer(assets, "seed_c", seed=7).train(fold)
    second = make_trainer(assets, "seed_d", seed=8).train(fold)

    assert first[0]["total_loss"] != second[0]["total_loss"]


def test_metrics_stay_registered_but_are_not_reported(
    assets: Path, fold: PaddedTraces
) -> None:
    """``state_acc`` scores filler and ``action_acc`` is 1.0 under option B (§4.3)."""
    trainer = make_trainer(assets, "metrics")
    history = trainer.train(fold)

    assert {"state_acc", "action_acc"} <= set(trainer.metrics)
    assert "state_acc" not in history[0]
    assert "action_acc" not in history[0]


def test_the_epoch_counter_ends_at_the_budget(
    assets: Path, fold: PaddedTraces
) -> None:
    """Upstream's ``self.epoch += epoch`` accumulates triangularly; this counts."""
    trainer = make_trainer(assets, "counter", epoch=3)
    trainer.train(fold)

    assert trainer.epoch == 3


def test_resume_is_refused(assets: Path) -> None:
    trainer = make_trainer(assets, "resume")
    with pytest.raises(NotImplementedError, match="resume"):
        trainer.resume(None, None, {})

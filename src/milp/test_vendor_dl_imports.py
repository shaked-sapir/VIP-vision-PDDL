"""Import-only smoke test over the vendored ROSAME DL + convertor tree.

    python -m pytest src/milp/test_vendor_dl_imports.py

The vendored tree uses upstream's absolute imports (``from planning_structs...``,
``from util.model_perm...``), which only resolve after ``src.milp`` has inserted
``vendor/`` on ``sys.path``. Nothing here exercises behaviour; it fails when a
module stops importing or when one of the recorded vendor modifications is
reverted or silently widened.

Covers:
  1. Every vendored module imports, and the two concrete entry points
     (``dl.network.Network``, ``dl.model.ROSAMEGoal``) are reachable.
  2. ``src.milp`` is the precondition for a bare ``import dl`` (checked in a
     subprocess, since this process has already paid for it).
  3. The four vendor modifications hold: the guarded ``SummaryWriter``, the
     guarded ``lifted_pddl`` ``Parser``, the ``dl/util/tuning.py`` stub, and the
     stripped CLI re-exports in ``dl/__init__`` and ``dl/main/__init__``.
  4. The deliberately un-vendored CLI/plot modules are absent, not stubbed.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)

_VENDOR = Path(src.milp.__file__).parent / "vendor"
_REPO_ROOT = Path(src.milp.__file__).parent.parent.parent

# Every module under vendor/, as an importable dotted name.
_VENDORED_MODULES = [
    "constraint_opt",
    "constraint_opt.cp_sat",
    "constraint_opt.factory",
    "constraint_opt.util",
    "convertor",
    "convertor.convertor",
    "convertor.pseudo_label",
    "convertor.selector",
    "convertor.translator",
    "dl",
    "dl.main",
    "dl.main.normalization",
    "dl.mixins",
    "dl.mixins.action",
    "dl.mixins.action_model",
    "dl.mixins.encoder_decoder",
    "dl.mixins.output",
    "dl.mixins.pair",
    "dl.mixins.plot",
    "dl.model",
    "dl.network",
    "dl.util",
    "dl.util.ROSAME",
    "dl.util.ROSAME.rosame",
    "dl.util.dataset",
    "dl.util.layers",
    "dl.util.plot",
    "dl.util.tuning",
    "dl.util.util",
    "planning_structs",
    "planning_structs.domain",
    "planning_structs.instance",
    "planning_structs.traces",
    "planning_structs.util",
    "util",
    "util.model_perm",
    "util.pddl_parsing",
]

# Upstream modules we chose not to vendor (plan §2.1). `constraint_opt.mip_gurobi`
# is vendored but needs a licence, so it is excluded from the import sweep.
_NOT_VENDORED = [
    "dl.main.common",
    "dl.main.rosame_full",
    "dl.util.stacktrace",
    "util.ablation_plot",
    "util.mip_objective_plot",
]


@pytest.mark.parametrize("name", _VENDORED_MODULES)
def test_vendored_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_module_list_matches_the_tree() -> None:
    """The parametrised list above is the whole tree, so nothing escapes the sweep."""
    on_disk = set()
    for path in _VENDOR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_VENDOR)
        parts = list(rel.parts[:-1]) + ([] if rel.stem == "__init__" else [rel.stem])
        on_disk.add(".".join(parts))
    assert on_disk == set(_VENDORED_MODULES) | {"constraint_opt.mip_gurobi"}


def test_entry_points_are_reachable() -> None:
    import dl.model
    import dl.network

    assert isinstance(dl.network.Network, type)
    assert issubclass(dl.model.ROSAMEGoal, dl.network.Network)


@pytest.mark.parametrize("name", _NOT_VENDORED)
def test_unvendored_modules_are_absent(name: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(name)


def test_src_milp_is_the_precondition_for_a_bare_vendor_import() -> None:
    """Without `import src.milp`, `import dl` must fail — vendor/ is not a package."""
    probe = "import dl"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode != 0
    assert "No module named 'dl'" in result.stderr


def test_tensorboard_import_is_guarded() -> None:
    """`SummaryWriter` is bound either way; it must never raise at module scope."""
    import dl.network

    assert hasattr(dl.network, "SummaryWriter")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        assert dl.network.SummaryWriter is None
    else:
        assert dl.network.SummaryWriter is SummaryWriter


def test_lifted_pddl_import_is_guarded_and_fails_loudly_when_called() -> None:
    import util.pddl_parsing as pp

    assert hasattr(pp, "Parser")
    if pp.Parser is not None:
        pytest.skip("lifted_pddl is installed; the guard is inert")
    with pytest.raises(ImportError, match="lifted_pddl"):
        pp.parse_pddl_domain(object(), Path("/nonexistent/domain.pddl"))


def test_tuning_stub_exposes_only_the_reachable_symbol() -> None:
    """`dl.main.normalization` uses `parameters` as an image mean/std cache."""
    import dl.util.tuning

    assert isinstance(dl.util.tuning.parameters, dict)
    assert not hasattr(dl.util.tuning, "grid_search")
    assert not hasattr(dl.util.tuning, "simple_genetic_search")


def test_cli_re_exports_are_stripped() -> None:
    """Upstream's `__init__` files pull in the argparse layer we did not vendor."""
    import dl
    import dl.main

    assert not hasattr(dl, "main_")
    assert not hasattr(dl.main, "main")
    assert not hasattr(dl.main, "common")
    assert not hasattr(dl.main, "rosame_full")
    assert hasattr(dl, "util")
    assert hasattr(dl, "model")

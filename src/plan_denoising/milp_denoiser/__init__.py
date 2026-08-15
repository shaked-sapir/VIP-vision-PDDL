"""CDPS's MILP denoiser: repair the trajectories in one solve, then learn.

The same repair problem ``conflict_search.py`` attacks by search, handed to
CP-SAT instead. A solve produces repaired trajectories T', and PI-SAM learns the
returned model from them. See ``docs/pisam-milp-denoiser-design.md``.

The encoder itself is not here — it lives in ``src/milp/``, because the
``rosame_milp*`` baselines drive the identical encoding toward a different end.
What is here is everything that only means something once the caller is CDPS:

  config.py                — the validated ``pisam_milp`` config block
  trajectory_extraction.py — solved MILP -> repaired (re-masked) observations T'
  single_round.py          — the ``pisam_milp_single_round`` driver
  loop.py                  — the ``pisam_milp_loop`` driver (rounds + selection)
  model_prior.py           — a learned model -> the encoder's reference channel
"""

from __future__ import annotations

# Modules in this package import the vendored ``planning_structs`` /
# ``constraint_opt`` by their upstream top-level names. Those names resolve only
# after src.milp has put its vendor directory on sys.path; importing it here
# makes that hold for every module in the package, rather than depending on the
# order in which each one happens to list its imports.
import src.milp  # noqa: F401

"""Deciding which observations to distrust, before a learner is asked to trust them.

Trajectories arriving from a noisy classifier contradict each other and
contradict any single action model. This package finds a minimal set of edits
that makes them mutually consistent, and hands the repaired trajectories on. It
does not learn action models — it calls a learner as a subroutine and judges the
result.

Two denoisers solve that same problem by different means:

  conflict_search.py — CDPS, an explicit search over patches, guided by the
                       conflicts a learner reports.
  milp_denoiser/     — the same repair posed as one CP-SAT program.

Shared by both:
  evaluator.py         — ``observations_reconstruction_score``, the
                         ground-truth-free model score used for online
                         selection. Offline reporting uses a different,
                         GT-aware metric that must never be used here.
  patch_accounting.py  — the single rule for counting *realized* fluent edits.
                         Patches are toggles, so magnitudes only compare
                         correctly under odd parity over normalized keys.
  frontier.py          — search-shape strategies for CDPS.

Nothing here is imported by the learners it drives; the dependency runs one way.
"""

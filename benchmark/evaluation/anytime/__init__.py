"""Anytime performance profiles for the denoising arms.

An anytime curve is a claim about what a learner had on the table at each point
in time. Every arm in this project already writes intermediate models, but each
writes them in its own shape and none of them scores them. This package supplies
the two missing halves: a reader that normalises the shapes into one checkpoint
stream, and an offline scorer that gives every checkpoint a comparable number.

Scoring is deliberately post-hoc. If each arm scored its own checkpoints inline,
the evaluation time would land on the x-axis of the very curve it is measuring,
and arms that checkpoint more densely would be punished for measuring themselves
more carefully.
"""

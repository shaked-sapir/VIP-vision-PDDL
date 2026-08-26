# Endpoint anchoring across the seven ROSAME arms

How each arm uses the ground-truth **initial** and **final** states of a trace.
Validated against the code, not against plan documents. Every claim below cites
the file and line it was read from.

**The decision this records:** we stay faithful to each upstream. The 24 and 26
imaged arms treat the initial state differently, that difference is not
equalised, and any 24-vs-26 comparison carries the caveat in §3.

---

## 1. The table

| arm | initial state | final state | paper | input | uses MILP |
|---|---|---|---|---|---|
| `rosame_24` | — (symbolic states given) | GT target | ICAPS-24 | symbolic | no |
| `rosame_milp_24` | hard row (problem `:init`) | hard row (GT trajectory) | ICAPS-24 | symbolic | yes |
| `rosame_milp_24_tag` | hard row (problem `:init`) | hard row (GT trajectory) | ICAPS-24 | symbolic | yes |
| `rosame_i_24` | **CV-predicted**, rollout start | GT target, weighted γ | ICAPS-24 | imaged | no |
| `rosame_i_milp_24` | hard row (problem `:init`) | hard row (GT trajectory) | ICAPS-24 | imaged | yes |
| `rosame_i_26` | **GT, fed as rollout input** | GT target, weighted γ | ICAPS-26 | imaged | no |
| `rosame_i_milp_26` | hard row (problem `:init`) | hard row (GT trajectory) | ICAPS-26 | imaged | yes |

The only asymmetry is between the two **DL-only imaged** arms, `rosame_i_24` and
`rosame_i_26`. The MILP arms all take both endpoints as hard constraints
regardless of which network feeds them, so the MILP pair is the clean 24-vs-26
comparison.

---

## 2. Where each row comes from

### `rosame_i_24` — CV-predicted init, GT final

`benchmark/algorithm_adapters/rosame_i_runner.py:304-326`, `_loss_from_predictions`:

```python
domain_preds = preds[:-1] * (1 - dele) + (1 - preds[:-1]) * add       # 310
loss = F.mse_loss(domain_preds[:-1], preds[1:-1], reduction="sum")     # 313  consistency
loss = loss + gamma * F.mse_loss(
    domain_preds[-1], trace.final_state_vec, reduction="sum")          # 318  GT final
loss = loss + F.mse_loss(
    (1 - preds[:-1]) * pre, torch.zeros_like(pre), reduction="sum")    # 322  applicability
loss = loss + lambda_ * F.mse_loss(pre, torch.ones_like(pre), ...)     # 326  precondition prior
```

The CV net predicts **every** state including the first, so `preds[0]` is a
*prediction* of the initial state and the rollout starts from it. Ground truth
enters at exactly one site: `trace.final_state_vec` on line 319.

The init is dropped deliberately at the fold walk,
`benchmark/baselines/rosame_i_runner.py:280-283`:

> "The GT *init* state that `resolve_fold_inputs` also returns is dropped: the
> ICAPS-24 network takes only a goal anchor."

`as_prepared_problem` (line 296) passes `trace.gt_final_predicates` and nothing
else; `gt_init` appears nowhere else in that file.

### `rosame_i_26` — GT init as rollout input, GT final as target

`src/milp/vendor/dl/model.py:109-138`, the forward pass:

```python
z = torch.sigmoid(dapply(features, self.symbol_net))   # 109  [B, T, symbol_dim] — INTERIOR frames only
...
z_ext = torch.cat([i.unsqueeze(1), z], dim=1)          # 130  GT init prepended as a real row
z_unsqueezed = z_ext.unsqueeze(2)
z_suc_aae = z_unsqueezed * (1 - all_deleffs) + (1 - z_unsqueezed) * all_addeffs   # 132
p_applicable = 1 - all_precons * (1 - z_unsqueezed)    # 138
```

`i` (init) and `g` (goal) are **inputs to the forward pass**, not targets in a
loss. The CV net never sees the endpoint frames — which is why
`src/milp/trace_tensors.py` drops both endpoints and keeps only interior frames
(`T = N-1`, `interior_frame_count`).

Both endpoints reach the batch through
`benchmark/baselines/rosame26_data.py:425-426`:

```python
init_predicates=[canonical(p) for p in trace.gt_init_predicates],
goal_predicates=[canonical(p) for p in trace.gt_final_predicates],
```

and are written into `state_traces` by `src/milp/trace_tensors.py:140-142` —
*"GT init, zero filler, then the GT goal held to the end."*

---

## 3. The one thing that is easy to get wrong

`loss_app` does **not** anchor the initial state. Read it,
`src/milp/vendor/dl/model.py:186-195`:

```python
p_applicable_suffix = p_applicable[:, 1:, ...]
loss_suffix = (a_suffix @ F.mse_loss(
    p_applicable_suffix, torch.ones_like(p_applicable_suffix), reduction='none')).sum(...)
loss_first = (a[:, 0, :] @ F.mse_loss(
    p_applicable[:, 0, :], torch.ones_like(p_applicable[:, 0, :]), reduction='none')).sum(...)
return self.parameters["gamma"] * loss_first + loss_suffix
```

The target is `torch.ones_like(p_applicable)` — the **constant 1**, not the init
state. This is the applicability constraint: the taken action's preconditions
must hold in the state it was taken from. It is the same constraint as ICAPS-24's
line 322-324. The γ on `loss_first` upweights applicability *at the first
timestep*, because that timestep's state is GT rather than CV-predicted and so is
trustworthy enough to weight more.

Only `loss_pred` (`model.py:178-184`) has a GT state as its target, and that
target is `goals`:

```python
target_last = goals.unsqueeze(1).expand_as(z_suc_aae[:, -1, ...])
mse_last = F.mse_loss(z_suc_aae[:, -1, ...], target_last, reduction='none')
loss_last = (a[:, -1, :] @ mse_last).sum(dim=(1, 2))
return loss_prefix + self.parameters["gamma"] * loss_last
```

**So both papers supervise against the GT final state only.** The difference is
structural, not a second loss term:

> ICAPS-26 starts its rollout from a **known** state.
> ICAPS-24 starts its rollout from a **predicted** one.

Stating it as "the 26 arm gets an extra loss anchor" is wrong about the
mechanism, even though the direction of the advantage is arguably the same.

---

## 4. Why we are not equalising it

The 26 arm's use of the init cannot be removed by deleting a loss term, because
there is no init loss term. It is structural: `z_ext = torch.cat([i, z])`
(`model.py:130`), and since the network never encodes the endpoint images, `z` is
one row short of a full trace. Removing `i` leaves the rollout with no starting
state.

Equalising would require feeding a CV-predicted init instead of the GT one —
re-adding the endpoint frames to the tensor builder and changing
`trace_tensors.py`'s `T = N-1` arithmetic. That is a real deviation from ICAPS-26
upstream.

**Decision: accept the caveat, stay faithful to both upstreams.** The
consequence for reporting:

- `rosame_i_24` vs `rosame_i_26` (DL-only) is **contaminated** — the 26 arm's
  rollout begins from ground truth and its CV net is never asked to read the
  endpoint frames. Carry this caveat wherever those two are compared.
- `rosame_i_milp_24` vs `rosame_i_milp_26` is **clean** — both take both
  endpoints as hard rows.

### Interaction with the trace-length sweep

The advantage does not stay constant as trace length grows. At `L=3` the GT init
is one of a few states; at `L=11` the unanchored interior is much larger for both
arms while the 26 arm's rollout still begins from a known state. A 24-vs-26 trend
across `L` should not be read as an algorithmic effect without accounting for
this.

---

## 5. Related

- `docs/rosame-i-26-failure-suggestions.md` §3a — the earlier write-up of this
  confound. It describes the difference as "extra supervision the 26 arm gets
  for free," which is right about the direction but wrong about the mechanism;
  §3 above supersedes that description.
- `rosame_i_26_fixing_PROCESS.md` §4 — the confound table this is the ninth row of.

# Imaged experiments: which VLM, which example protocol

*Decision note for Roni and Yarin, 2026-11-11. Full tables: `docs/image-experiments-week1-plan.md`, the section "Three models under three example protocols".*

## What was measured

Three models classified the same 331 frames, five problems in each of the five
domains. Each output was compared with the ground-truth states, giving two
numbers per cell: the share of atoms the model left uncertain (**mask**) and
the share of the remaining atoms it got wrong (**flip**).

- **gpt-5.2**, no reasoning.
- **Qwen3.8-27B, thinking off**, on one 48 GB cluster GPU (Yarin's model).
- **Qwen3.8-27B, thinking on**, same card.

## Result under the protocol we intend to use

The one-shot example comes from a held-out problem, so frame 0 of every problem
is classified like any other frame and no ground truth leaks into the input.
The example's answer is written as the full scored list the prompt asks for.

| domain | gpt-5.2 | Qwen, thinking off | Qwen, thinking on |
|---|---|---|---|
| blocksworld | 0.1 % / 0.3 % | 1.0 % / 0.6 % | 0.0 % / 0.3 % |
| depot | 0.0 % / 1.8 % | 0.0 % / 3.6 % | 0.0 % / 2.1 % |
| gripper | 3.4 % / 5.4 % | 0.0 % / 4.6 % | 0.0 % / 0.0 % |
| hanoi | 5.2 % / 6.0 % | 2.6 % / 10.8 % | 0.6 % / 4.3 % |
| npuzzle | 0.2 % / 0.6 % | 22.6 % / 3.8 % | 0.0 % / 0.0 % |
| seconds per frame | 4 to 27 | 10 to 143 | 28 to 131 |

Cells are mask / flip.

- Thinking-on Qwen is the most accurate of the three. It is at or below
  gpt-5.2's flip rate on every domain and exact on gripper and npuzzle.
- It is also 5 to 13 times slower per frame than gpt-5.2 on the shared card.
- Thinking-off Qwen is close to gpt-5.2 on three domains, flips about twice as
  often on depot and hanoi, and still masks a fifth of npuzzle.

## The one result that needs a judgement call

Writing the example as a scored list removes most masking. On hanoi it does so
by turning masks into flips, for every model:

| hanoi | facts-only example | scored example |
|---|---|---|
| gpt-5.2 | 16.2 % / 2.0 % | 5.2 % / 6.0 % |
| Qwen, thinking off | 34.2 % / 3.6 % | 2.6 % / 10.8 % |
| Qwen, thinking on | 33.1 % / 1.7 % | 0.6 % / 4.3 % |

The scored example shows only "true" and "false" scores, never "uncertain", so
the model stops abstaining. For our learner a masked atom can be ignored, while
a flipped one is noise that must be repaired. The scored example is therefore
the harder input for NPI-SAM on hanoi, even though it looks cleaner. On the
other four domains it removes masking at little or no cost in flips.

One gap: the facts-only example was measured only with the leaking protocol,
where the example is the problem's own frame 0. Facts-only with the held-out
example was not run.

## Decisions needed

**1. The model for the main results.**

- *gpt-5.2.* Fastest, the model behind every existing image result, closed.
- *Qwen, thinking on.* Most accurate, open and reproducible, 5 to 13 times
  slower. The five large corpora hold about 193,000 frames, which at the
  measured latency is thousands of GPU-hours. Throughput under real batching
  was not measured.
- *Qwen, thinking off.* Dominated: slower than gpt-5.2 and less accurate.

Recommendation: gpt-5.2 for the main corpora, and thinking-on Qwen on the five
small corpora as an open-model replication. That is about 650 frames, roughly
half a day on one card.

**2. The example protocol.**

The held-out example is not in question, since it removes the leak. The open
point is scored or facts-only.

Recommendation: keep the scored example. It is the version in which the example
agrees with what the prompt asks for, and that reason does not depend on which
variant helps our learner. We then report the realised mask and flip rates per
domain beside every image result, so hanoi's harder input is visible. As a
sensitivity check, run the facts-only held-out variant on hanoi alone, which is
73 frames.

**3. The size of the large image corpus.** 500, 1,000 or 2,500 problems per
domain, and standard or batch API. This follows from decision 1 and is costed
in `docs/image-experiments-plan.md`, section 6.

## Not affected by any of this

The pixel baselines read only frames, actions and the two ground-truth endpoint
states. Their runs on the small corpora are already under way on the cluster.

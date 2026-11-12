# Open VLM on the cluster

`install_vllm.sbatch` builds the `vip_vllm` conda env once; `serve_vlm.sbatch`
serves a Hugging Face VLM through vLLM's OpenAI-compatible endpoint on one GPU
node. The pipeline reaches it from a laptop through an SSH tunnel and the
`vllm` vendor block of `config.yaml`:

```bash
sbatch --constraint rtx_6000 scripts/cluster_image/serve_vlm.sbatch
cat logs/vlm_endpoint-<jobid>.txt            # node:port, once the model is loaded
ssh -N -L 8000:<node>:8000 bgu &            # on the laptop
python benchmark/data_generator.py --domain blocksworld --vendor vllm ...
```

`vllm.fluent_classification_model.model_name` must equal `VIP_VLM_NAME`
(`--served-model-name`); it also names the output directory, so it carries no
`/`. Cancel the serving job when the run ends: `scancel <jobid>`.

# Pixel arms on a GPU

The ROSAME-I arms (`rosame_i_24`, `rosame_i_milp_24`, `rosame_i_26`,
`rosame_i_milp_26`) read only frames, observed actions and the two GT endpoint
states, so they run on the existing small image experiments whatever VLM the
symbolic arms use.

```bash
python -m scripts.cluster_image.make_pixel_manifest        # 25 rows: domain x fold
scripts/cluster_image/deploy_pinned.sh HEAD projects/VIP-pixel bgu
```

`deploy_pinned.sh` builds `~/projects/VIP-pixel` from `git archive <commit>`,
so the jobs read one commit and nothing from a working tree; `PINNED_COMMIT`
records it. It also copies the manifest's data folders and a slim copy of each
image experiment into `benchmark/running_results_image_src/`. Keep this folder
apart from `~/projects/VIP-image`, which the VLM jobs read: a sync into a
folder with queued jobs changes the code under them.

On the cluster, from `~/projects/VIP-pixel`:

```bash
VIP_PX_BASELINES="rosame_i_24 rosame_i_milp_24 rosame_i_26 rosame_i_milp_26" VIP_PX_ARGS="--learn-timeout 300" sbatch --constraint rtx_4090 --array=0-24 scripts/cluster_image/pixel_fold.sbatch
```

Add `--exclude cs-4090-09`: on 2026-11-11 that node reported an RTX 4090 to
`nvidia-smi` but failed CUDA initialisation in torch, while other rtx_4090 and
rtx_6000 nodes worked with the same environment. A task that lands on such a
node fails in its first seconds (`--device cuda` is checked at start-up).

`--epochs` reaches every pixel arm, so the 24 arms (paper epoch counts, no
`--epochs`) and the 26 arms (`--epochs 5000 --budget-mode fixed`, stopped by
the fold's wall clock) go in two arrays with two out roots.

One task is one (domain, fold) over the fold's training-size sweep, the arms in
sequence. Rows land under `$VIP_PX_OUT` (default
`benchmark/running_results_image_pixel`), mirrored as
`<domain>/<experiment>/testing/<cell>/`; `benchmark/merge_result_roots.py`
brings them into the main tree. Each task writes
`logs/vip-px-gpu-<jobid>.csv` (GPU memory and utilisation every 15 s) and
prints the peak. To run two arrays over the same cells at once, give each its
own `VIP_PX_OUT`.

# Image experiments on the cluster

Two experiments, each a symbolic part on CPU nodes and a pixel part on GPUs:

| | run configs | manifests (`scripts/cluster_image/`) | experiment |
|---|---|---|---|
| A, small corpora `__fs=shared` | `benchmark/run_config_small_image.yaml` (blocksworld, hanoi, npuzzle), `benchmark/run_config_small_image_n9.yaml` (depot, gripper: 9 problems, sizes 3..7) | `image_small_{symbolic,pixel}_manifest.csv`, 25 rows | `TO=600__image-small__fs=shared` |
| B, large corpora, N=100 | `benchmark/run_config_large_image.yaml` (blocksworld, hanoi) | `image_large_{symbolic,pixel}_manifest.csv`, 10 rows | `large-corpora__image-mini__N=100` |

A manifest row is one (domain, fold); a domain is a block of five rows
(`make_image_manifests.py` prints the ranges). Regenerate after editing a config:

```bash
python -m scripts.cluster_image.make_image_manifests --name image_small --config benchmark/run_config_small_image.yaml benchmark/run_config_small_image_n9.yaml
python -m scripts.cluster_image.make_image_manifests --name image_large --config benchmark/run_config_large_image.yaml
```

## 1. Deploy (laptop)

Commit first: the deploy reads `git archive <commit>`.

```bash
scripts/cluster_image/deploy_image_experiments.sh HEAD projects/VIP-imgexp bgu
```

It copies, per data folder of the two pixel manifests, the image pool (frames,
pddl, GT json, inferred files), `gt_trajectories/` and the top-level files.
One experiment only: append its pixel manifest as a fourth argument.

## 2. Setup pass (cluster, from `~/projects/VIP-imgexp`)

```bash
sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_small_image.yaml
sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_small_image_n9.yaml
sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_large_image.yaml blocksworld
sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_large_image.yaml hanoi
```

One job prepares a domain's pool (`*_gtrate0_frame_axioms.*`, written into the
shared image pool) and writes every cell's `fold_info.json`,
`domain_reference.pddl`, `original_observations/` and the fold's test states
under `benchmark/running_results_image_src/<domain>/<experiment>/`. The fold
jobs of one domain all write those pool files when they are missing, so this
job runs alone on a domain and ends before that domain's arrays start. It is
also the only place a cell's `original_observations/` is written at the cell
root: a run without the `cdps` arm writes it under the MILP arm's subfolder,
after the baselines.

## 3. Symbolic arrays (CPU), one per domain

```bash
M=scripts/cluster_image/image_small_symbolic_manifest.csv
for R in 0-4 5-9 10-14 15-19 20-24; do sbatch --time 0-08:00:00 --array=$R scripts/cluster_image/symbolic_fold.sbatch $M; done
M=scripts/cluster_image/image_large_symbolic_manifest.csv
for R in 0-4 5-9; do sbatch --array=$R scripts/cluster_image/symbolic_fold.sbatch $M; done
```

Arms: `PISAM_MILP_LOOP__schema=pre_and_add` (`__m=4` on B), `PISAM`,
`ROSAME_24`, `ROSAME_MILP_24`. Rows land in
`benchmark/running_results/<domain>/<experiment>/`. A task resumes: a
re-submitted row skips the cells that hold `fold_result.json`.

## 4. Pixel arrays (GPU), 24 arms and 26 arms apart

They read the setup pass's tree, not the running experiment, so they can be
submitted as soon as step 2 has ended. `M` is the pixel manifest of the
experiment; `T` is 600 for A and 3600 for B (add `--time 1-00:00:00` on B).

```bash
# A: blocksworld 0-4, hanoi 5-9, npuzzle 10-14 at the default 64 px
VIP_PX_BASELINES="rosame_i_24 rosame_i_milp_24" VIP_PX_OUT=benchmark/running_results_imgexp_pixel_24 VIP_PX_ARGS="--learn-timeout $T" \
  sbatch --constraint rtx_4090 --exclude cs-4090-09 --array=0-4 scripts/cluster_image/pixel_fold.sbatch $M
VIP_PX_BASELINES="rosame_i_26 rosame_i_milp_26" VIP_PX_OUT=benchmark/running_results_imgexp_pixel_26 VIP_PX_ARGS="--epochs 5000 --budget-mode fixed --learn-timeout $T" \
  sbatch --constraint rtx_4090 --exclude cs-4090-09 --array=0-4 scripts/cluster_image/pixel_fold.sbatch $M

# A: depot 15-19, gripper 20-24 at native resolution
VIP_PX_BASELINES="rosame_i_24 rosame_i_milp_24" VIP_PX_OUT=benchmark/running_results_imgexp_pixel_24 VIP_PX_ARGS="--learn-timeout $T --resize native" \
  sbatch --constraint rtx_4090 --exclude cs-4090-09 --array=15-19 scripts/cluster_image/pixel_fold.sbatch $M
VIP_PX_BASELINES="rosame_i_26 rosame_i_milp_26" VIP_PX_OUT=benchmark/running_results_imgexp_pixel_26 VIP_PX_ARGS="--epochs 5000 --budget-mode fixed --learn-timeout $T --resize native" \
  sbatch --constraint rtx_pro_6000 --mem 48G --array=15-19 scripts/cluster_image/pixel_fold.sbatch $M
```

B uses the first two commands with its own manifest (blocksworld 0-4, hanoi
5-9); a task selects `--cells fold<F>_numtrajs`, which on B is the one cell
`fold<F>_numtrajs100_gtrate0`. One array per domain; the 24 and the 26 arrays
of a domain share cells, so each keeps its own `VIP_PX_OUT`.

The 26 arms standardise frames by a statistic over every frame of the corpus,
cached beside the pool (`.rosame26_norm__<domain>__res=<r>.pt`) by the first
task that needs it. On B that is all 2500 problems' frames: submit one task of
a domain (`--array=0`), and the other four once its log shows training.

## 5. Results (laptop)

```bash
E='TO=600__image-small__fs=shared'      # or large-corpora__image-mini__N=100
for D in blocksworld hanoi npuzzle depot gripper; do
  rsync -a "bgu:projects/VIP-imgexp/benchmark/running_results/$D/$E/" "benchmark/running_results/$D/$E/"
done
rsync -a bgu:projects/VIP-imgexp/benchmark/running_results_imgexp_pixel_24/ benchmark/running_results_imgexp_pixel_24/
rsync -a bgu:projects/VIP-imgexp/benchmark/running_results_imgexp_pixel_26/ benchmark/running_results_imgexp_pixel_26/
```

`benchmark/merge_result_roots.py --config <sources yaml>` then brings the pixel
rows into the main tree; the sources file lists the two pixel roots and the
labels to take. A pulled experiment records the cluster's `data_dir`; pass
`--data-dir` to a local backfill of it.

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

One task is one (domain, fold) over the fold's training-size sweep, the arms in
sequence. Rows land under `$VIP_PX_OUT` (default
`benchmark/running_results_image_pixel`), mirrored as
`<domain>/<experiment>/testing/<cell>/`; `benchmark/merge_result_roots.py`
brings them into the main tree. Each task writes
`logs/vip-px-gpu-<jobid>.csv` (GPU memory and utilisation every 15 s) and
prints the peak. To run two arrays over the same cells at once, give each its
own `VIP_PX_OUT`.

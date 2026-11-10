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

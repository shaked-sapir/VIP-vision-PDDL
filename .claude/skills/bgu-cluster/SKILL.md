---
name: bgu-cluster
description: Step-by-step guidance for working with the Ben-Gurion University (BGU) ISE/CS/DT 2024 SLURM HPC cluster — university VPN setup (CheckPoint + ComSign OTP), SSH connection to slurm.bgu.ac.il, conda environment management, sbatch job files, interactive jobs (sinteractive/sjupyter), GPU/CPU resource requests, job arrays, Golden Tickets (QoS), local SSD scratch, file transfer in both directions (rsync/scp/tar pulling results back to a laptop), IDE integration (PyCharm/VS Code), Docker (Apptainer/UDOCKER), Tensorboard, and SLURM job control. Use when the user mentions BGU cluster, slurm.bgu.ac.il, vpn.bgu.ac.il, sbatch, sinteractive, sjupyter, MobaXterm, ComSign authenticator, requesting compute jobs, GPUs (gtx1080 / rtx2080 / rtx3090 / rtx4090 / rtx6000 / titan_rtx / tesla_p100), submitting batch jobs at Ben-Gurion University, copying/downloading/pulling results or files from the cluster to a local machine (rsync, scp), or troubleshooting SLURM/CUDA errors.
---

# BGU SLURM Cluster Workflow

Use this skill when the user is working on the Ben-Gurion University HPC cluster (`slurm.bgu.ac.il`). The cluster is SLURM-based: a Manager Node (login node) dispatches jobs to GPU/CPU compute nodes from a priority queue. Each user gets a Linux home directory at `/home/{username}/` on shared storage accessible from all nodes.

> **Hard prerequisite:** every interaction with the cluster requires being on the BGU network — either on-campus Wi-Fi or the **university VPN**. Without it, nothing reaches `slurm.bgu.ac.il`.

> **Critical:** the Manager Node is a shared resource for launching, monitoring, and controlling jobs — **NEVER** use it for computational purposes.

> **Video tutorials:** Moodle course "HPC הדרכת קלסטר" at `https://moodle.bgu.ac.il/moodle/course/view.php?id=60163` (password: `cluster20252`).

---

## 1. One-time VPN setup (skip if already done)

Required only the first time, or after replacing the phone.

1. **Install the ComSign Authenticator** app on the phone (App Store / Play Store).
2. Open it → menu (≡, top-right) → **Device ID** → save the unique device code that's issued.
3. On a computer browser, log in to the BGU OTP portal with university credentials (same as Moodle: username, password, ID number).
4. In the "Fill Details" form enter:
   - **Device ID code** — the string from step 2
   - **PIN code** — a 4-digit PIN of your choice (call it `XXXX`); needed every time you open ComSign to mint a one-time password
5. Wait for the **ComSign Trust** confirmation email.
   - **iPhone:** open the attached `token.ctrust` file → Share → ComSign → "Token accepted successfully".
   - **Android:** open the link next to "For Android click here" in the email.
6. Install the **CheckPoint Endpoint Security VPN** client (download from the official CheckPoint site → Remote Access VPN Products → Remote Access Client). During install pick **Endpoint Security VPN**.
7. Open CheckPoint (lock icon next to the clock) → right-click → **Connect** → Yes.
   - **Server Address:** `vpn.bgu.ac.il`
   - **Display Name:** leave unchecked
   - On the security warning: **Trust and Continue**
   - Authentication: **Standard (Default)** + **Username and Password**

---

## 2. Connecting to the VPN (every session)

1. Right-click the CheckPoint icon next to the clock → **Connect**.
2. **Username:** `<email-prefix>@vpn`  (e.g. `israel@vpn` for `israel@post.bgu.ac.il`)
3. **Password:** the one-time password currently shown in the ComSign app (unlock the app with the `XXXX` PIN to reveal it).
4. A green dot on the CheckPoint tray icon = connected.

If the connection fails repeatedly, it is usually network load on CheckPoint's side, not a setup issue.

---

## 3. SSH to the cluster

The recommended client on Windows is **MobaXterm Home Edition (installer edition)** — download the zip from the official site, extract, run `installer.exe` (or `.msi`). MobaXterm supports X11 forwarding out of the box.

To connect:

1. Open MobaXterm → **Session** → **SSH**
2. **Remote host:** `slurm.bgu.ac.il`
3. **Username:** university email prefix (no `@vpn` suffix here — that's only for VPN). E.g. `israel`.
4. First connection only: accept the host fingerprint (**Accept**).
5. Password: the regular university password (same as Moodle). The field is silent — keystrokes/paste are accepted but nothing is rendered. Hit Enter.
6. When asked to save the password → **No** (avoids breakage on the next org-wide password change).
7. Success = the `SLURM HPC @ BGU` banner appears. The shell is bash on Linux — `pwd`, `ls`, etc.

> Linux/Mac users can equivalently `ssh <username>@slurm.bgu.ac.il` from any terminal, after VPN.

---

## 4. Home directory & recommended layout

Every user gets `/home/{username}/`, password-protected, accessible from every node (the storage is shared). System files there (`.bashrc`, `.bash_profile`, `.bash_logout`, `.bash_history`) — leave them alone.

Recommended structure:

```
/home/{username}/
├── projects/   # one subfolder per project
├── scripts/    # .sbatch files
├── data/       # (optional) cross-project data
└── misc/       # everything else
```

Create with `mkdir projects scripts data misc` from the home directory, or use MobaXterm's "create new directory" button in the left file panel.

If you copy files, remember file permissions: `chmod +x <path>` for files that need execution permissions.

---

## 5. File transfer

> Options A-D below are mainly **laptop → cluster**. For the reverse direction
> (**cluster → laptop**, e.g. pulling results after a sweep) see **§5b**, which
> documents several failure modes that will silently waste your time otherwise.

**A. MobaXterm drag-and-drop (visual)** — connect, navigate to the target folder in the left file panel, drag files in. Works in both directions.

**B. WinSCP** — dedicated file transfer tool for Windows.

**C. GitHub via terminal** — for projects already in a repo:

```bash
cd ~/projects
git clone <repo-url>
```

If the repo is private, GitHub will prompt for credentials. The password **must be a Personal Access Token** — the actual GitHub password does not work over the terminal.

**D. Download public files via `wget`:**

```bash
# from AWS S3
wget --no-check-certificate --no-proxy 'https://<bucket>.s3.amazonaws.com/<path>'

# from Google Drive
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=<FILE_ID>' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=<FILE_ID>"
```

Do **not** transfer: `.git/`, `.idea/`, `__pycache__/`.

---

## 5b. Pulling results back to a laptop (rsync over SSH) — read this before transferring

This is the cluster→laptop direction, used after a sweep finishes. It has several
non-obvious failure modes that were hit in practice. Follow the recipe rather than
improvising.

### Prerequisites

1. **VPN must be up.** Verify: `ifconfig | grep 132.72` or `ssh bgu 'hostname'`.
2. **SSH alias** in `~/.ssh/config` (avoids retyping host/user and enables key auth):
   ```
   Host bgu bgu-cluster
       HostName slurm.bgu.ac.il
       User <username>
       IdentityFile ~/.ssh/id_ed25519
       IdentitiesOnly yes
   ```
3. **Destination directories must already exist locally.** rsync will not create
   missing intermediate parents for a trailing-slash destination.

### The four gotchas

**① macOS ships rsync 2.6.9 (from 2006).** Check with `rsync --version`. It does
**not** support modern flags. Specifically:

| Flag | rsync 2.6.9 (macOS) | Notes |
|---|---|---|
| `--info=progress2` | ❌ **unsupported — hard error** | Use `--progress` instead |
| `--stats` | ✅ supported | Safe |
| `--partial` | ✅ supported | Safe, enables resume |
| `-a -z -v` | ✅ supported | Safe |
| `--outbuf` | ❌ unsupported | Avoid |

Using `--info=progress2` makes **every** transfer fail instantly. If you want a
modern rsync: `brew install rsync` then use `/opt/homebrew/bin/rsync`.

**② Never end a transfer loop with a bare `echo`.** This masks failures:

```bash
# WRONG — exits 0 even if all five rsyncs failed
for d in a b c; do rsync ... ; done; echo "=== TRANSFER DONE ==="
```

The `echo` is the last command, so `$?` reflects the `echo`, not rsync. A "DONE"
banner then appears over a completely failed transfer. Use `&&`-chaining so the
success marker is only reachable when every step actually succeeded (see recipe).

**③ Shell-variable expansion is unreliable in some non-interactive/eval wrappers.**
Symptoms: headers print as `=== ===`, and the remote path collapses (e.g.
`$d/results*` becomes `results*`), giving rsync error code 23. Multi-line `for`
loops can also die with `parse error near 'do'`.

**Fix: use literal paths, one rsync per line, no loop variables.** It is verbose
but it works every time.

**④ `du -sh` on the cluster massively overstates the transfer size.** `du` counts
*allocated blocks*. With tens of thousands of small JSON files, block padding
inflates the figure ~4×. A real case: `du` reported **2.8 G**, the actual byte
payload was **0.72 G**. Measure real bytes before deciding anything:

```bash
ssh bgu "cd <remote_dir> && python3 -c \"
from pathlib import Path
fs=[f for f in Path('.').rglob('*') if f.is_file()]
print('files:',len(fs))
print('bytes: %.2f GB'%(sum(f.stat().st_size for f in fs)/1024**3))
\""
```

### Recommended recipe

Run it in the **background** (large transfers exceed the 2-minute default tool
timeout), with literal paths and `&&`-chaining:

```bash
mkdir -p local/dest/a local/dest/b local/dest/c

rsync -az --partial "bgu:~/projects/MyProj/results/a/run-name*" local/dest/a/ && echo "a OK" && \
rsync -az --partial "bgu:~/projects/MyProj/results/b/run-name*" local/dest/b/ && echo "b OK" && \
rsync -az --partial "bgu:~/projects/MyProj/results/c/run-name*" local/dest/c/ && echo "c OK" && \
echo "=== ALL TRANSFERRED ==="
```

Flag rationale:
- `-a` archive (recursive, preserves times/perms/symlinks)
- `-z` compress on the wire — a large win for JSON/text-heavy results
- `--partial` keep partial files so a re-run resumes instead of restarting
- quote the remote glob so the **remote** shell expands it, not the local one
- add `-n` (dry run) first if unsure what will be pulled

### Verify after transfer — never trust the banner alone

Compare **file counts and byte totals**, not `du` output:

```bash
# remote
ssh bgu "cd <remote_dir> && python3 -c \"
from pathlib import Path
for d in ['a','b','c']:
    fs=[f for f in (Path('.')/d).rglob('*') if f.is_file()]
    print('%-12s files=%6d bytes=%.2f GB'%(d,len(fs),sum(f.stat().st_size for f in fs)/1024**3))
\""

# local — same script against the local destination, then diff the numbers
```

For SLURM sweeps, also verify the **domain-level result markers** (e.g. count
`fold_result.json` files) and that each parses as valid JSON with a non-empty
record list. A transfer can be byte-complete yet still reveal that some jobs
wrote empty results.

### Sanity checks worth running

```bash
# no stray carriage returns in directory names (see §24 on CRLF manifests)
python3 -c "from pathlib import Path; print(sum(1 for p in Path('local/dest').rglob('*') if chr(13) in p.name))"
```

### Alternatives

- **`scp -r`** — simpler but no resume, no compression by default, and far slower
  for many small files. Prefer rsync.
- **`tar` + single stream** — fastest for very many tiny files, since it avoids
  per-file round trips:
  ```bash
  ssh bgu "cd ~/projects/MyProj/results && tar czf - run-name*" | tar xzf - -C local/dest/
  ```
  No resume capability, so best for one-shot transfers under a few GB.

---

## 6. GitHub SSH key setup on the cluster

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# accept default location, set a passphrase
cat ~/.ssh/id_ed25519.pub
# copy the key to https://github.com/settings/keys → "New SSH key"
ssh -T git@github.com   # test (enter passphrase when prompted)
```

---

## 7. Conda environments (mandatory)

Anaconda is pre-installed on every cluster account — **do not reinstall it**. Each project gets its own conda env to avoid library conflicts.

```bash
conda create -n my_env python=3.10
conda activate my_env
pip install numpy
pip install -r requirements.txt
conda list                       # packages in current env
conda env list                   # all envs
conda list -n other_env          # inspect a non-active env
conda env remove --name my_env   # delete env
conda create -n new_env --clone my_env   # clone
conda update conda               # update conda itself
conda deactivate
```

**Compare two environments:** `python /storage/conda_compare.py <env1> <env2>`

> The env isolates libraries, **not files** — files in `/home/...` are visible from any env.

Special install notes:
- **tensorflow-gpu:** use `pip install 'tensorflow[and-cuda]'` (not `pip install tensorflow`)
- **PyTorch:** use `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

---

## 8. Submitting jobs

### Batch jobs (`sbatch`)

Fire-and-forget — persists after disconnect. Submit while **no conda env is active** (`conda deactivate` first):

```bash
sbatch my_job.sbatch
```

An example `.sbatch` file is at `/storage/example.sbatch` on the cluster.

### Interactive jobs (`sinteractive`)

Requires the SSH session to stay open. Used for Jupyter, IDEs, and debugging.

```bash
sinteractive                          # default values
sinteractive --time 0-5:00:00 --gpu 1 # 5 hours, 1 GPU
sinteractive --help                   # all options
```

For course users: `sinteractive --qos course --part course --gpu 1`

To SSH into the compute node of an existing job: `srun --jobid=<your-jobid> --pty bash`

**Always cancel when done:** `scancel <job_id>`

### `.sbatch` file format

Lines:
- `#SBATCH ...` → directive to SLURM
- `##SBATCH ...` → SLURM comment (ignored — like commenting out a directive)
- `### ...` → bash comment (documentation only)
- regular bash → executed on the compute node

### Key parameters

| Flag | Meaning | Notes |
|------|---------|-------|
| `--partition` | Compute node group | Default `main`. Other partitions (`rtx3090`, `rtx2080`, `gtx1080`, `rtx6000`) require QoS. |
| `--time` | Max runtime | Format `D-HH:MM:SS` (e.g. `0-10:30:00`). Max 7 days. Job killed if exceeded. |
| `--job-name` | Friendly name | Used in queue listings and `%x` in output filenames. |
| `--output` | stdout/stderr log path | Use `%J` for job-ID. For arrays: `%A` (master ID), `%a` (task ID). |
| `--mail-user`, `--mail-type` | Email notifications | Events: `ALL`, `BEGIN`, `END`, `FAIL`, `REQUEUE`, `ARRAY_TASKS`, `NONE`. |
| `--mem` | RAM | Default `24G` per GPU. Hard cap **60G per single job**; more requires IT. |
| `--gpus` | GPU count + (optional) type | `--gpus=1`, `--gpus=rtx_3090:1`. Set to `0` for CPU-only. More than 1 GPU requires IT permission. |
| `--cpus-per-task` | CPU cores | Override default CPU allocation (4-6 CPUs auto-allocated per GPU). |
| `--tasks` | Processes | Default 1. Use >1 for MPI or concurrent `srun` programs. |
| `--constraint` | Node feature | Select by CPU/GPU type: `cpu`, `gpu`, `cpu128`, `cpu256`, `gtx_1080`, `rtx_2080`, `rtx_3090`, `rtx_4090`, `rtx_6000`, `titan_rtx`, `tesla_p100`. |
| `--exclude` | Exclude nodes | E.g. `--exclude=dt-1080-01,ise-1080-02`. |
| `--nodelist` | Specific node | E.g. `--nodelist=dt-1080-01`. |
| `--tmp` | Local SSD space | E.g. `--tmp=100G` for `/scratch` allocation. |

### Available GPUs

| GPU | VRAM | Constraint value |
|-----|------|-----------------|
| `gtx1080` | ~8 GB | `gtx_1080` |
| `rtx2080` | ~8 GB | `rtx_2080` |
| `rtx3090` | ~24 GB | `rtx_3090` |
| `rtx4090` | ~24 GB | `rtx_4090` |
| `rtx6000` | ~48 GB | `rtx_6000` |
| `titan_rtx` | ~24 GB | `titan_rtx` |
| `tesla_p100` | ~16 GB | `tesla_p100` |

### Example `.sbatch`

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-10:30:00
#SBATCH --job-name my_job
#SBATCH --output my_job-id-%J.out
#SBATCH --mail-user=user@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
##SBATCH --mem=48G

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

module load anaconda
source activate my_env
python mycode.py my_arg
```

---

## 9. SLURM job management commands

| Command | Purpose |
|---------|---------|
| `sbatch job.sbatch` | Submit batch job |
| `squeue --me` | List own jobs (`ST=PD` waiting, `ST=R` running) |
| `sinfo -Nel` | Cluster node info (NODELIST, sockets:cores:threads) |
| `sres` | Cluster-wide free resources |
| `scancel <job_id>` | Cancel by ID |
| `scancel --name <job_name>` | Cancel by name |
| `scancel -t PENDING -u <username>` | Cancel all pending jobs |
| `scontrol show job <job_id>` | Full info on running job |
| `sstat -j <job_id> --format=MaxRSS,MaxVMSize` | Live memory usage |
| `sacct -j <jobid> --format=JobName,MaxRSS,AllocTRES,State,Elapsed,Start,ExitCode,DerivedExitcode,Comment` | Complete post-run report |
| `nvidia-smi -L` | Which GPU was allocated (run in sbatch or on compute node) |

---

## 10. Allocating resources wisely

- Use **1 GPU per job**. More than 1 requires IT permission.
- **RAM:** default 24G per GPU. Maximum 60G (contact IT for more). ⚠️ The usual advice to check `sacct -j <jobid> --format=JobName,MaxRSS` **does not work on this cluster** — memory accounting is disabled, so `MaxRSS` is always empty (see §24). Always write units: `--mem 24G`, not `--mem 24` (which means 24 MB).
- **GPU VRAM:** most GPUs hold ~11G, advanced ones ~24G. If your code needs >60G RAM, revise it.
- **CPUs:** 4-6 CPUs auto-allocated per GPU — do not override unless CPU-only.
- **Wall time:** max 7 days per job.

---

## 11. Jupyter Lab

### Installation (once per env)

```bash
conda activate my_env
conda install jupyterlab
python -m ipykernel install --user --name my_env --display-name "my best env"
```

### Launch

```bash
sjupyter                              # default values
sjupyter --time 0-5:00:00 --gpu 1     # 5 hours, 1 GPU
sjupyter --help                       # all options
```

For course users: `sjupyter --qos course --part course --gpu 1`

Wait for resources → the script prints a URL → copy and paste into browser → ignore SSL warning → proceed.

### Release resources from within Jupyter

Add at the end of your notebook code:

```python
import os
job_cancel_str = "scancel " + os.environ['SLURM_JOBID']
os.system(job_cancel_str)
```

### Working with notebooks

If you close the browser tab while a cell runs, it keeps running but **you lose the output**. Solutions:
- Write results to a file instead of relying on cell output.
- Run the code as a `.py` script via `sbatch` instead.
- Use `%%capture cap_out` cell magic as the first line, then `cap_out.show()` later or save with `open('cap_output.txt', 'w').write(cap_out.stdout)`.

---

## 12. Tensorboard

**Without Jupyter:**

1. Run your program and generate logs in `my_log_dir`
2. Wait for the run to end
3. SSH to the compute node
4. `conda activate my_environment`
5. `tensorboard --bind_all --logdir=my_log_dir`
6. Copy-paste the link to your browser

**In Jupyter:**

```python
%load_ext tensorboard
# ... run cells that generate logs ...
!tensorboard --bind_all --logdir=my_log_dir
# copy-paste link to browser
```

---

## 13. Job arrays

Run identical scripts with different environment variables (parameter tuning, seed averaging):

```bash
#SBATCH --array=1-10                  # 10 parallel runs
#SBATCH --output=file_name_%A_%a.out  # %A = master job ID, %a = task ID
```

Each task gets its own resources and a unique `SLURM_ARRAY_TASK_ID`.

**Access in Python:**
```python
import os
task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
```

**Access in R:**
```r
task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID")
```

**Send as argument:** `python my_code.py $SLURM_ARRAY_TASK_ID`

**Map to input files:**
```bash
file=$(ls *.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)
myscript -in $file
```

**Read a line from input list:**
```bash
SAMPLE_LIST=($(<input.list))
SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}
```

**Email per task:** `#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS`

**Limit concurrent tasks:** `#SBATCH --array=0-15%4` (max 4 at a time from 16 total)

---

## 14. High Priority Jobs (Golden Tickets)

Some research groups have rights to preempt other jobs when resources are scarce. Golden Tickets are **disabled on `main` partition** — use the partition matching your group's GPU type rights.

```bash
sbatch --partition=gtx1080 --qos=our_qos my_awesome.sbatch
```

- `--qos` is usually the instructor's username.
- Available QoS partitions: `gtx1080`, `rtx2080`, `rtx3090`, `rtx6000`.
- If group's prioritized resources are exhausted, the job pends even if other resources are free.
- **Rule:** when no QoS needed → use `main`. When using QoS → do NOT use `main`.

### Prioritize your own jobs

Lower a job's priority so your other jobs run first:

```bash
scontrol update JobId=<my-job-id> Nice=500   # higher value = lower priority
```

---

## 15. Sending arguments to sbatch

```bash
sbatch --export=ALL,var1='1',var2='hello' my_sbatch_file.sbatch
```

In the sbatch file, access with `$var1`, `$var2`.

---

## 16. Job dependencies

Chain jobs into pipelines:

```bash
sbatch --dependency=after:<jobid> job.sbatch          # start after other_job starts
sbatch --dependency=afterok:<jobid> job.sbatch         # start after other_job succeeds
sbatch --dependency=afterok:77:79 job.sbatch           # start after both 77 and 79 succeed
sbatch --dependency=singleton job.sbatch                # start after all previous jobs with same name+user finish
```

---

## 17. Local SSD scratch storage

Use `/scratch` on the compute node for fast I/O. **All data is erased when the job ends/fails/is cancelled.**

```bash
#SBATCH --tmp=100G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
cp /storage/*.img $SLURM_SCRATCH_DIR          # copy data TO local SSD
# ... your computation using $SLURM_SCRATCH_DIR ...
cp -r $SLURM_SCRATCH_DIR $SLURM_SUBMIT_DIR    # copy results BACK before job ends
```

---

## 18. CUDA version selection

Add **before** `module load anaconda` in the `.sbatch`:

```bash
module load cuda/11.8
```

Available CUDA modules: `7.0, 7.5, 8.0, 9.0, 9.1, 9.2, 10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8, 12.9, 13.0, 13.1`.

---

## 19. IDE integration

### PyCharm Professional

1. SSH to Slurm → `sinteractive --gpu 1` → note the compute node hostname and job ID.
2. In PyCharm: Settings → Project → Project Interpreter → Add Interpreter → **On SSH**.
3. Enter the **compute node** hostname (NOT the manager node) and your BGU username/password.
4. Choose **System Interpreter** → set path to `~/.conda/envs/<your_env>/bin/python`.
5. Wait for file upload to complete (status bar).
6. To avoid re-uploading each time: map the sync folder to your cluster home directory.
7. Remote file tree: `ALT+F1` → Select "Remote Host".

**Keep a script running after disconnecting PyCharm:**

```python
import os, sys
os.system("nohup bash -c '" + sys.executable + " train.py --size 192 >result.txt' &")
```

### Visual Studio Code

1. Install the **Remote – SSH** extension pack and the Python extension.
2. SSH to Slurm → `sinteractive --gpu 1` → note the compute node hostname.
3. Press the green `><` button (bottom-left) → "Remote – SSH: Connect to host…" → enter `<user>@<compute_node>` (NOT the manager node).
4. Enter BGU password → `Ctrl+Shift+P` → "Python: Select Interpreter" → `~/.conda/envs/<env>/bin/python`.
5. For notebook cell support: `Ctrl+Shift+P` → "Preferences: Open Workspace Settings" → Python → set Conda Path to `/storage/modules/packages/anaconda/lib/python3.11/venv/scripts/common/activate`.

**Run/Debug with arguments:** create `launch.json`:
```json
{
  "configurations": [{
    "name": "Python: Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "cwd": "${fileDirname}",
    "args": ["cuda", "100", "exit"]
  }]
}
```

**Jupyter in VS Code SSL fix:**
1. Settings → search "cert" → tick "Jupyter: Allow Unauthorized Remote Connection".
2. Edit `~/.vscode/settings.json` → add `"http.systemCertificates": true`.
3. Add to `~/.bashrc`: `export NODE_TLS_REJECT_UNAUTHORIZED='0'`.

> **VS Code glibc errors:** versions 1.86+ require glibc not available on the cluster. Install VS Code 1.85 from `update.code.visualstudio.com/1.85.2/win32-x64-user/stable` and disable auto-update (File → Settings → "Update: Mode" → "none").

---

## 20. Docker containers

### Apptainer (preferred)

```bash
# create interactive job
sinteractive --gpu 1
# SSH to the compute node, then:
apptainer build --force my_container.sif docker://my_container_page/my_container
apptainer exec my_container.sif /bin/bash
# with GPU access and directory binding:
apptainer exec --nv --bind cluster_dir:container_dir my_container.sif /bin/bash
```

**From local Docker image:**
```bash
# on your machine:
docker save -o my-image.tar my-image:latest
# copy tar to cluster, then:
apptainer build my-image.sif docker-archive://my-image.tar
```

### UDOCKER

Install in a conda environment:

```bash
conda create -n udocker_env python=3.8
conda activate udocker_env
conda install configparser
pip install udocker
```

Usage:
```bash
udocker pull tensorflow/tensorflow:2.8.0rc0-gpu-jupyter
udocker create --name=tf_gpu_jup28 tensorflow/tensorflow:2.8.0rc0-gpu-jupyter
udocker setup --nvidia tf_gpu_jup28
udocker run tf_gpu_jup28 nvidia-smi
udocker run -v /home/my_user/my_code_dir:/home tf_gpu_jup28 python3 /home/my_code.py
```

Key commands: `udocker ps` (list containers), `udocker images` (list images), `udocker rm <name>`, `udocker rmi <id>`.

**Jupyter Lab in UDOCKER:** copy `/storage/udocker_jup.sbatch` and modify environment/image names.

---

## 21. Other languages

### Matlab

```bash
module load matlab
# GUI (requires X11 forwarding):
srun --x11 --nodes=1 --mem=24G --cpus-per-task=4 --gpus=1 --partition=main matlab -nosoftwareopengl -desktop -sd ~
# Headless batch:
srun --nodes=1 --mem=24G --cpus-per-task=4 --gpus=1 --partition=main matlab -nosplash -nodisplay -nodesktop -sd ~ -batch "my_matlab_script"
```

### Julia

```bash
julia -e 'using Pkg;Pkg.add("IJulia")'
# Julia kernel becomes available in Jupyter
```

### R

```bash
conda create -n r_env r-essentials r-base
# R in Jupyter:
conda create -n r_jupyter python=3.9 jupyterlab r-essentials r-base
conda install -c conda-forge r-irkernel
R -e "IRkernel::installspec()"
```

**RStudio via Apptainer:** copy `/storage/scripts/apptainer/rstudio/*` → `sbatch rstudio.sbatch` → open `132.72.X.Y:port` in browser.

### C# (.NET)

```bash
conda install -c conda-forge dotnet-sdk
dotnet new console -o myApp && cd myApp && dotnet run
```

---

## 22. SLURM job management commands (quick reference)

| Command | Purpose |
|---------|---------|
| `sbatch job.sbatch` | Submit batch job |
| `sinteractive [options]` | Launch interactive job |
| `sjupyter [options]` | Launch Jupyter Lab job |
| `squeue --me` | List own jobs |
| `sinfo -Nel` | Cluster node info |
| `sres` | Cluster-wide free resources |
| `scancel <job_id>` | Cancel by ID |
| `scancel --name <name>` | Cancel by name |
| `scancel -t PENDING -u <user>` | Cancel all pending |
| `scontrol show job <id>` | Full running job info |
| `sstat -j <id> --format=MaxRSS,MaxVMSize` | Live memory usage |
| `sacct -j <id> --format=JobName,MaxRSS,AllocTRES,State,Elapsed,Start,ExitCode,DerivedExitcode,Comment` | Post-run report |
| `nvidia-smi -L` | Show allocated GPU |

---

## 23. FAQ & Troubleshooting

### Job pending reasons

| Reason | Meaning |
|--------|---------|
| `PartitionTimeLimit` | `--time` exceeds partition max (usually 7 days) |
| `Resources` | Cluster has insufficient free resources |
| `Priority` | Queued behind higher-priority jobs |
| `QOSMaxJobsPerUserLimit` | Max concurrent jobs for requested partition reached |
| `MaxGRESPerAccount` | Golden Ticket GPU limit exceeded for your account |

### Common errors

**CUDA out of memory:** reduce batch size. For TensorFlow, it grabs 95% of GPU RAM by default — use `tf.config.experimental.set_memory_growth(physical_devices[0], True)`. For PyTorch, call `del variables; gc.collect(); torch.cuda.empty_cache()`.

**`CUDA error: no kernel image is available`:** CUDA code not compiled for your GPU architecture. The 1080 GPU is outdated — try a newer GPU type, or update PyTorch.

**`RuntimeError: CUDA error: device-side assert triggered`:** run on CPU with `CUDA_LAUNCH_BLOCKING=1` to find the actual error location (usually out-of-range indexing).

**TensorFlow doesn't recognize GPU:** ensure `pip install 'tensorflow[and-cuda]'` and matching CUDA/cuDNN versions. Use `module load cuda/xx.x` for the right version.

**PyTorch 3090 incompatibility:** upgrade PyTorch: `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

**TensorFlow 3090 incompatibility:** requires tensorflow > 2.2.

**`OSError: No space left on device`:** `/dev/shm` (tmpfs) is full. Solutions: reduce dataset size, set `num_workers=0`, or redirect temp files: `export TMPDIR=/scratch/...` and `mp.set_sharing_strategy('file_system')`.

**Unbuffered Python output:** `python -u my_app.py` or add `export PYTHONUNBUFFERED=TRUE` to sbatch (has a performance cost).

**DDP hangs on RTX6000:** add `export NCCL_P2P_DISABLE=1` to sbatch (server topology doesn't support P2P).

**`libstdc++.so.6: GLIBCXX_3.4.26 not found`:** add to sbatch: `export LD_LIBRARY_PATH=/home/<user>/.conda/envs/<env>/lib:$LD_LIBRARY_PATH`

**`Segmentation fault (core dumped)` with Python:** conda environment is corrupted — create a new one.

**Java OOM:** JVM auto-sets MaxHeapSize to 32GB — manually set max heap to fit allocated memory.

**Two interactive jobs on same node:** SSH session uses the first job's resources. Connect to the second with `srun --jobid=<2nd-jobid> --pty bash`.

**VS Code: "Remote host key has changed":** delete/rename files in `C:\Users\<user>\.ssh\` — Windows will create new ones.

**VS Code: "Cannot find kernels" after reinstall:** File → Preferences → Profile → Show Profile Content → inspect for corruption, or create a new profile.

**RTX6000 `--gpus=rtx_6000:2` error:** add `#SBATCH --cpus-per-gpu=8` (system erratum).

---

## 24. Things that bite people

- **Editing a queued job's resources is not possible** — `scancel` it and resubmit.
- A running job can be **pre-empted** for a higher-priority user; it appears as `failed` and does NOT auto-resume.
- Only `--partition main` for regular accounts; other partitions require QoS/Golden Tickets.
- Personal Access Token, **not the GitHub password**, for `git clone` of private repos.
- The MobaXterm password prompt is **silent** — typing works but nothing is rendered.
- **Always release resources** when done — even during a few-hours break from interactive sessions.
- Clean up unused files and datasets from the shared storage.
- Anaconda is pre-installed — do NOT reinstall it.

### Transfer & tooling traps (see §5b for the full recipe)

- **macOS rsync is 2.6.9 (2006)** — `--info=progress2` is a hard error. Use `--progress`/`--stats`.
- **A trailing `echo` masks transfer failures** — the loop exits 0 regardless. Use `&&`-chaining.
- **`du -sh` overstates size ~4×** for many small files (block padding). Sum real bytes instead.
- **Loop variables may not expand** in non-interactive/eval shells — use literal paths.

### CRLF in generated manifests silently corrupts output directory names

If an array job reads its parameters from a CSV written by Python's `csv.writer`,
the **default `excel` dialect emits CRLF**. Combined with `sed` + `IFS=, read` in
the sbatch template, the trailing `\r` stays glued to the last field and leaks
into every directory the job creates.

Symptom: `ls -la` of the parent shows the directory, but `cd`/`find` on the
literal path reports **"No such file or directory"**. Diagnose with:

```bash
ls -1 | cat -A     # a trailing ^M reveals the carriage return
```

Two-sided fix:
```python
csv.writer(f, lineterminator="\n")   # producer
```
```bash
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"   # consumer
```

Repairing already-created dirs requires **renaming the directory AND stripping
`\r` from any JSON that recorded the name** (e.g. a `run_params.json` used for
resume checks) — otherwise every resume aborts on a parameter mismatch.

### Memory accounting is disabled cluster-wide

`sacct` returns **empty `MaxRSS`/`AveRSS`** because `JobAcctGatherType = (null)`
(`scontrol show config | grep JobAcctGather`). `seff` is also broken (missing
Perl `Sys::Hostname`). **You cannot measure real memory usage on this cluster** —
`--mem` can only be calibrated by observing whether a job OOMs, which is evidence,
not measurement. Ignore §10's advice to check `MaxRSS`; it will not work here.

### `--mem` bare integers are megabytes, not gigabytes

`#SBATCH --mem 24` requests **24 MB**, not 24 GB. Always write the unit: `--mem 24G`.

### Runtime does not scale monotonically with input size

In a 270-task sweep, the five slowest tasks were all `num_trajectories=3`
(up to 35 min), while `num_trajectories=8` cells finished in ~11 min — an inverse
relationship, because search difficulty (not data volume) dominated. **Do not
extrapolate `--time` from the "biggest" configuration.** Sample several corners
of the grid, then add generous headroom.

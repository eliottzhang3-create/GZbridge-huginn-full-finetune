# GUIZHOU_codex
Just for convenience, want to refine the code frome the remote server via codex.

# Huginn Full Finetuning Sync Repo

## Purpose

This repository is a **code-sync workspace** for Huginn full-parameter finetuning experiments on GSM8K.

It is **not** intended to store:

- model weights
- checkpoints
- output directories
- cached datasets
- long training logs

Its main purpose is:

1. let local Codex edit code comfortably
2. let the user `git push` locally
3. let the remote HPC machine `git pull`
4. keep experiment code, scripts, and debugging state synchronized

---

## Remote Environment

Primary remote paths currently in use:

- Model code directory:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125`
- Training code directory:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main`

Important remote files currently under active modification:

- `/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125/raven_modeling_minimal.py`
- `/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/finetuning_test_gsm8k_fsdp.py`
- `/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/run_train_huginn_full_gsm8k_fsdp.sh`
- `/hpc_stor03/sjtu_home/jinwei.zhang/code/recurrent-pretraining-main/local_scripts/train_huginn_full_gsm8k_fsdp.sh`

---

## Recommended Repo Layout

Mirror only the code and small config structure that matters:

```text
repo-root/
  README_HUGINN_FULL_FINETUNE.md
  .gitignore
  models/
    huginn-0125/
      raven_modeling_minimal.py
      raven_config_minimal.py
      config.json
      README.md
  code/
    recurrent-pretraining-main/
      finetuning_test_gsm8k_fsdp.py
      run_train_huginn_full_gsm8k_fsdp.sh
      local_scripts/
        train_huginn_full_gsm8k_fsdp.sh
```

If additional small Python/config files are modified, add them too.

Do **not** add the large weight shards from `models/huginn-0125/`.

---

## What `.gitignore` Is

`.gitignore` is a file that tells Git:

> "These files/directories should not be tracked or committed."

For this project, it is critical because the remote Huginn model directory may contain:

- `model-00001-of-00004.safetensors`
- other `*.safetensors`
- checkpoints
- logs
- generated outputs

Without a proper `.gitignore`, these can accidentally be committed to GitHub.

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo

# Logs
*.log
logs/

# Outputs / checkpoints
outputs/
checkpoints/
**/outputs/
**/checkpoints/

# Dataset caches
dataset/
datasets/
cache/
**/dataset/

# Torch / numpy / pickle
*.pt
*.pth
*.bin
*.npy
*.npz
*.pkl

# Hugging Face / model weights
*.safetensors
model-*.safetensors

# OS/editor
.DS_Store
Thumbs.db
.idea/
.vscode/
```

If the repo mirrors the Huginn directory structure, this is usually enough to keep all large weights out of Git.

---

## Project Background

The training target is **Huginn**, a recurrent language model architecture with three main structural parts:

- `prelude`
- `core_block`
- `coda`

Unlike a standard decoder-only Transformer with a simple one-pass stack of blocks, Huginn uses recurrent computation in the `core_block` section.

This matters a lot for distributed training.

---

## Goal

Run **full-parameter finetuning** of Huginn on **GSM8K** using **8x V100** with FSDP, while preserving Huginn's training characteristic of **random long-tail recurrent iteration counts**.

There is also a LoRA line of work, but this repository and debugging context are focused on the **full finetuning path**.

---

## Current Training Strategy

### 1. Manual Fine-Grained FSDP

The current working wrapping strategy is:

- wrap every real block inside:
  - `transformer.prelude`
  - `transformer.core_block`
  - `transformer.coda`
- then wrap the whole model once more with an outer/root FSDP

This was chosen instead of `auto_wrap_policy`.

Reason:

- Huginn uses `ModuleList` containers plus recurrent execution
- naive auto-wrap caused deadlocks and container-related wrap confusion
- manual wrapping is more controllable

### 2. Cross-Rank Shared Recurrent Step Sampling

Fine-grained FSDP only became stable after synchronizing recurrent step counts across ranks.

Current idea:

- rank 0 samples `(num_steps_no_grad, num_steps_with_grad)` once per global step
- this pair is broadcast to all ranks
- every rank runs the same recurrence depth for that step

This preserves:

- randomness **between** training steps

while enforcing:

- consistency **within** a distributed step

This was necessary to avoid FSDP collective mismatches and hangs.

### 3. Current Label Construction

The current script uses manual next-token shifting:

- `input_ids = raw[:, :-1]`
- `labels = masked raw[:, 1:]`

The Huginn forward path does **not** apply another internal shift before CE loss.

Debugging so far strongly suggests:

- the current full-finetuning path is **not** suffering from the earlier LoRA-style double-shift issue
- the supervised region starts at the Huginn answer tokens, not the system/user prompt

---

## Important Debugging History (Up to date)

### Phase 1: Root-Only FSDP

Root-only FSDP could run, but memory pressure remained high.

### Phase 2: Auto-Wrap Attempt

An `auto_wrap_policy` approach was tried and failed.

Main issues:

- Huginn `core_block` is a `ModuleList`
- recurrent execution means different ranks can enter child FSDP modules different numbers of times if recurrence is sampled independently
- this caused deadlocks / hangs

### Phase 3: Manual Fine-Grained FSDP + Shared Step Counts

This combination resolved the earlier distributed hang problem.

### Phase 4: Current Main Issue = Numerical Instability

The current blocker is no longer FSDP deadlock.

The current blocker is:

- first real optimizer update triggers non-finite grad norm
- often at `data_step=5`, `optimizer_step=0`

This means:

- gradients blow up during or before the first true update
- parameters themselves usually remain finite at that moment

---

## What the Current Debug Logs Suggest

Current evidence indicates:

- parameters are usually still finite
- gradients become non-finite before optimizer step
- the problem appears before or around the first true update
- suspicious layers frequently include:
  - `transformer.adapter.weight`
  - `transformer.core_block.*.mlp.proj.weight`
  - sometimes `transformer.wte.weight`

The working hypothesis is:

- the issue is a **numerical stability problem in the forward/backward path**
- not primarily a label-shift bug
- not primarily an optimizer-step corruption bug

Current next-step debugging direction:

- inspect activations around:
  - adapter input/output
  - recurrent core block outputs
  - especially MLP projection path

---

## Current User Requirements

The user explicitly wants to preserve Huginn's defining training property:

- random long-tail recurrent training depth

Temporary fixed-depth experiments were only used for diagnosis, not as a final training design.

So any future solution should aim to keep:

- random recurrence across steps
- synchronized recurrence across ranks within each step

---

## How Codex Should Help In Future Chats

When a new Codex thread starts, it should understand:

1. This project is about Huginn **full finetuning**, not only LoRA.
2. The key active files are:
   - `models/huginn-0125/raven_modeling_minimal.py`
   - `code/recurrent-pretraining-main/finetuning_test_gsm8k_fsdp.py`
   - training shell scripts
3. Fine-grained FSDP must be **manual**, not naive auto-wrap.
4. Recurrent step counts must be shared across ranks per global step.
5. Current main blocker is **non-finite gradients near the first optimizer update**.
6. Current debugging focus is **activation / numerical stability**, especially around adapter and recurrent MLP projection paths.

### Operating Rules For Future Codex Chats

Codex should follow these rules strictly in future chats:

1. **Codex is local-only and cannot directly operate on the remote HPC server.**
   - Codex runs on the local machine. Codex can only modify or execute commands or programs on the local machine and of course in the sandbox.
   - Any remote command must be executed by the user.
   - The user will copy command output back into the chat. 
   - Therefore, Codex must give commands in a form the user can run directly on the remote machine.

2. **The remote machine is Linux, while the local machine is Windows.**
   - When discussing paths or commands, Codex must not mix Windows and Linux paths.
   - Remote runtime paths use Linux style, such as:
     - `/hpc_stor03/sjtu_home/jinwei.zhang/...`
   - Local editing paths use Windows style.
   - Codex must always be explicit about which machine a path or command belongs to.

3. **Codex must not pretend to know the remote state.**
   - Codex cannot see the live remote filesystem, installed packages, job queue, container state, logs, or current file contents unless the user provides them.
   - Before making a strong claim or giving a precise remote fix, Codex must first confirm missing remote facts by asking for:
     - command output
     - file snippets
     - logs
     - directory listings
     - scheduler status
   - Codex must not guess remote details when those details are important for correctness.

4. **Remote jobs must never be launched directly.**
   - Directly running long training tasks on the remote server is strictly forbidden.
   - The required workflow is:
     1. `vc info` to inspect available resources
     2. `bash <submit_script>.sh` or the approved `vc submit` submission flow
     3. `vc list` to inspect job status
     4. `tail -f <log_file>` to monitor runtime logs
   - Codex must not suggest directly launching training with raw `python`, `torchrun`, or similar commands on the remote server unless the user explicitly says it is for a short, safe, non-training diagnostic and it is allowed.

5. **Container choice is fixed and should not be changed casually.**
   - The user already has Miniconda locally.
   - The training container currently used on the remote side is fixed:
     - `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`
   - Codex should assume this container remains unchanged unless the user explicitly says otherwise.
   - The focus should be environment setup and code behavior, not replacing the container.

6. **The current repository is a sync repository, not the full runtime environment.**
   - The sync repo is mainly used to move code between local editing and remote execution.
   - It should contain:
     - source code
     - shell scripts
     - small configs
     - documentation
   - It should not contain:
     - model weights
     - checkpoints
     - outputs
     - dataset caches
     - large logs

7. **Codex should help the user by producing precise, minimal, executable remote instructions.**
   - Good outputs include:
     - exact shell commands
     - exact file snippets to replace
     - exact grep/sed/nl commands to inspect code
     - exact interpretation of returned logs
   - Bad outputs include:
     - vague speculation
     - long generic theory with no next action
     - instructions that assume remote access Codex does not have

8. **Codex should preserve Huginn’s core training property.**
   - Huginn’s random long-tail recurrent depth is important and should not be removed as a final solution.
   - Temporary simplifications such as fixed recurrence depth are allowed only as diagnostics.
   - Final recommendations should preserve:
     - random recurrence across global training steps
     - synchronized recurrence across ranks within a step

9. **Codex should be careful with file synchronization advice.**
   - The user may have:
     - the remote runtime copy of the code
     - the local sync repository copy
     - GitHub as the transport layer
   - Codex must be explicit about the source of truth for each step:
     - whether the user is syncing from remote to repo
     - from repo to local
     - or from local back to remote
   - Codex should not casually suggest overwriting remote runtime files unless the user confirms that the repository version is the desired newer version.

10. **Codex should always optimize for reducing user copy-paste burden.**
    - The user is using Git/GitHub precisely to avoid fragile manual copy-paste editing between local and remote.
    - Codex should favor workflows that let the user:
      - edit locally
      - commit/push cleanly
      - pull or sync predictably
    - If a suggested workflow would increase manual duplication, Codex should reconsider it.

---

## Suggested Local-to-Remote Workflow

1. Keep this repository locally.
2. Use Codex locally to edit code files.
3. Commit and push to GitHub.
4. On the remote server, pull the repo.
5. Copy synced code files into the real runtime paths if needed.
6. Submit jobs remotely through the existing training workflow.

If possible, keep the repo directory structure close to remote paths to make syncing simpler.

---

## Last Known Practical Reminder

Before remote training runs:

- run `python -m py_compile` on modified Python files
- verify duplicated debug helper functions have not been accidentally introduced
- verify `.gitignore` is keeping out large artifacts

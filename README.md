# GUIZHOU_codex

Just for convenience, this repo is used to refine code locally with Codex and sync it to the remote HPC side through GitHub.

# Huginn Full Finetuning Sync Repo

## Purpose

This repository is a **code-sync workspace**, not the full runtime environment.

It is used to:

1. edit code comfortably on the local Windows machine with Codex
2. `git push` the code to GitHub
3. `git pull` on the remote Linux HPC machine
4. keep experiment code, shell scripts, small configs, and documentation synchronized

This repository is **not** intended to store:

- model weights
- checkpoints
- output directories
- cached datasets
- large logs
- remote-only temporary artifacts

---

## Project Scope

This repo now contains **two active experiment lines**:

1. **Huginn full-parameter GSM8K finetuning**
   - based on FSDP
   - mainly for multi-GPU remote training
   - currently adapted for **8x RTX 5090**

2. **Huginn audio-modality experiment branch**
   - based on the **original Huginn backbone**, not the GSM8K-finetuned checkpoint
   - architecture: **Whisper-small encoder + temporal compressor + audio projector + Huginn text backbone**
   - current focus is **audio-to-text understanding/alignment**, not speech generation

---

## Local / GitHub / Remote Workflow

This project uses **GitHub as the transport layer** between local editing and remote execution.

### Machines

- **Local machine**
  - Windows
  - Codex edits code here
  - local paths use Windows style

- **Remote machine**
  - Linux HPC cluster
  - actual training / evaluation jobs run here
  - remote paths use Linux style

### Required workflow

1. edit locally in this repo
2. run local static checks when needed, mainly `python -m py_compile`
3. `git add/commit/push`
4. on remote, `git pull`
5. submit jobs remotely with the provided submit scripts
6. inspect remote logs and paste important output back into chat when debugging

Codex **cannot directly operate on the remote server**. Any remote command must be executed by the user.

---

## Remote Environment

### Main remote code roots currently in use

- Remote sync repo code root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/recurrent-pretraining-main`

- Remote model root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125`

- Remote audio experiment model root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1`

- Remote Whisper encoder root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-small`

### Main remote conda environments

- Training / most evaluation:
  - `swift_huginn`

- AAC caption metric evaluation:
  - `audio_eval`
  - used for `aac_metrics`-based caption benchmark scripts

### Fixed remote container

- `docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1`

Do not casually change the container unless the user explicitly asks.

---

## Queue / Submission Constraints

The current queue in use is:

- `pdgpu-5090`

Important queue rule from the user:

- for **1 GPU** jobs:
  - CPU cores must be `<= 8`
  - memory must be `<= 32G`

Therefore the standard single-GPU submit shape is:

- `-c 8 -m 32G -g 1 -n 1`

For **8 GPU** jobs, the current full-training submit script uses:

- `-c 32 -m 256G -g 8 -n 1`

which satisfies the per-GPU rule.

Remote jobs should be launched through the provided `vc submit` shell scripts, not by directly starting long training commands manually.

---

## Repository Layout

Current important structure:

```text
repo-root/
  README.md
  .gitignore
  models/
    huginn-0125/
      raven_modeling_minimal.py
      raven_config_minimal.py
      config.json
      ...
    huginn-audio-whisper-v1/
      _base.py
      raven_config_minimal.py
      raven_modeling_minimal.py
      config.json
      __init__.py
  code/
    recurrent-pretraining-main/
      finetuning_test_gsm8k_fsdp.py
      finetuning_test_gsm8k_fsdp_5090.py
      finetuning_audio_whisper_smoke.py
      finetuning_audio_whisper_tiny_overfit.py
      finetuning_audio_whisper_clotho_aqa.py
      finetuning_audio_whisper_clotho_caption.py
      prepare_clotho_caption_expand.py
      analyze_audio_whisper_clotho_aqa.py
      audio_alignment_eval_common.py
      eval_vocab_retrieval.py
      eval_audio_text_retrieval.py
      eval_visualization.py
      eval_audio_whisper_clotho_caption_aac_metrics.py
      run_*.sh
      local_scripts/
        train_*.sh
        eval_*.sh
```

---

## `.gitignore` Policy

This repo should track:

- source code
- shell scripts
- small config files
- documentation

This repo should not track:

- model shards
- `outputs/`
- checkpoints
- cached datasets
- temporary logs

The current `.gitignore` already excludes the main large artifacts such as:

- `outputs/`
- `*.pt`
- `*.pth`
- `*.bin`
- `*.npy`
- `*.pkl`
- `*.safetensors`
- `model-*.safetensors`

If a new dataset-preprocessing step creates local artifacts, check whether they should also be ignored before committing.

---

## Huginn Background

The training target is **Huginn**, a recurrent language model architecture with three main structural parts:

- `prelude`
- `core_block`
- `coda`

Unlike a standard decoder-only Transformer, Huginn uses recurrent computation inside `core_block`. This affects:

- distributed wrapping strategy
- recurrence sampling logic
- numerical stability debugging
- masking behavior
- multimodal prefix injection design

---

## GSM8K Full-Finetuning Line

### Main goal

Run **full-parameter finetuning** of Huginn on **GSM8K** with FSDP, preserving Huginn's characteristic of **random long-tail recurrent iteration counts**.

### Important scripts

- Main 5090 training script:
  - `code/recurrent-pretraining-main/finetuning_test_gsm8k_fsdp_5090.py`

- Main 5090 submit script:
  - `code/recurrent-pretraining-main/run_train_huginn_full_gsm8k_fsdp_5090.sh`

- Main 5090 local script:
  - `code/recurrent-pretraining-main/local_scripts/train_huginn_full_gsm8k_fsdp_5090.sh`

- GSM8K evaluation without system prompt:
  - `code/recurrent-pretraining-main/eval_huginn_full_checkpoint_gsm8k_5090.sh`

- GSM8K evaluation with system prompt:
  - `code/recurrent-pretraining-main/eval_huginn_full_checkpoint_gsm8k_5090_with_sys.sh`

### Known design choices

1. **Manual fine-grained FSDP**
   - wrap real blocks in `prelude`, `core_block`, `coda`
   - then root-wrap the whole model
   - avoid naive `auto_wrap_policy`

2. **Shared recurrent step counts across ranks**
   - rank 0 samples recurrence settings
   - broadcast to all ranks
   - preserves randomness across steps while preventing cross-rank mismatches within one distributed step

3. **8x5090 adaptation**
   - dedicated 5090 script exists
   - queue name is already wired to `pdgpu-5090`

4. **Checkpoint behavior**
   - intermediate checkpoint frequency was adjusted during debugging
   - training scripts and save logic should always be checked before restarting long runs

### Current status

- The 8x5090 full-training path has already been brought to a runnable state.
- Earlier issues included:
  - dynamic module import problems
  - Huginn remote-code file mismatch
  - torchvision `VideoReader` import issue from dataset formatting
  - non-finite grad norm
  - OOM / precision-related instability
- A complete run was later achieved successfully.

This means the GSM8K line is **no longer at the "cannot run" stage**; the current repo already contains the stabilized code path that got the job to finish.

---

## Huginn Audio Experiment Line

### High-level objective

Build an **independent audio experiment branch on top of the original Huginn backbone**, without modifying the already GSM8K-finetuned model.

### V1 architecture

Current experiment branch:

- audio encoder:
  - **Whisper-small**
- temporal compressor:
  - Conv1d downsampling + normalization + activation + adaptive pooling
- audio projector:
  - project Whisper-side features into Huginn text hidden space
- Huginn text backbone:
  - frozen in V1

Current V1 training policy:

- freeze **Huginn backbone**
- freeze **Whisper encoder**
- train only:
  - `temporal_compressor`
  - `audio_projector`
  - optional `audio_bos`
  - optional `audio_eos`

### Important model files

- `models/huginn-audio-whisper-v1/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-v1/raven_config_minimal.py`
- `models/huginn-audio-whisper-v1/_base.py`

### Important training scripts

- smoke test:
  - `code/recurrent-pretraining-main/finetuning_audio_whisper_smoke.py`

- tiny overfit:
  - `code/recurrent-pretraining-main/finetuning_audio_whisper_tiny_overfit.py`

- full ClothoAQA training:
  - `code/recurrent-pretraining-main/finetuning_audio_whisper_clotho_aqa.py`

- Clotho caption continuation training:
  - `code/recurrent-pretraining-main/finetuning_audio_whisper_clotho_caption.py`

### Current data roots used on remote

- ClothoAQA:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn`

- tiny ClothoAQA subset:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_huginn_tiny_train32`

- Clotho caption:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn`

### Current data assumptions

#### ClothoAQA-style training data

Each record contains:

- `audio_path`
- question / instruction text
- answer text

#### Clotho caption training data

The caption data was expanded so that one audio with multiple references becomes multiple training samples.

Current helper script:

- `code/recurrent-pretraining-main/prepare_clotho_caption_expand.py`

Current caption training manifest:

- `train_expand.json`

Current evaluation manifest convention:

- `test_expand.jsonl`

### Current progress

The audio branch has already passed several stages:

1. **smoke test passed**
   - random / synthetic path can forward + backward + save

2. **tiny overfit passed**
   - tiny ClothoAQA subset can train and save checkpoints

3. **full ClothoAQA training completed**
   - current notable checkpoint lineage includes:
     - `huginn-audio-whisper-clotho-aqa-v2/checkpoint-7029`

4. **Clotho caption continuation training implemented and run**
   - initialized from the ClothoAQA adapter checkpoint
   - uses expanded caption training manifest

5. **audio alignment analysis tooling implemented**
   - retrieval / visualization / vocab probing scripts are now in repo

This means the audio line is already beyond the "just wire projector and pray" phase; there is now an actual trainable branch, checkpoints, and post-training analysis tooling.

---

## Current Audio Training Defaults

### `finetuning_audio_whisper_clotho_aqa.py`

- run name:
  - `huginn-audio-whisper-clotho-aqa-v2`
- dataset:
  - `clotho_aqa_huginn`
- micro batch size:
  - `3`
- optimizer:
  - `AdamW`
- learning rate:
  - `1e-4`
- scheduler:
  - `cosine with warmup`
- warmup ratio:
  - `0.05`

### `finetuning_audio_whisper_clotho_caption.py`

- run name:
  - `huginn-audio-whisper-clotho-caption-v1`
- init checkpoint:
  - from ClothoAQA adapter checkpoint
- dataset:
  - `clotho_caption_huginn`
- train manifest:
  - `train_expand.json`
- micro batch size:
  - `5`
- optimizer:
  - `AdamW`
- learning rate:
  - `5e-5`
- scheduler:
  - `cosine with warmup`
- warmup ratio:
  - `0.05`

Note:

- current audio training is **single-GPU**, not distributed
- `optimizer_update_every=1_micro_step` in the current scripts

---

## Audio Evaluation / Analysis Tooling

There are now **two different audio-eval directions** in this repo.

### 1. Caption benchmark evaluation

Script:

- `code/recurrent-pretraining-main/eval_audio_whisper_clotho_caption_aac_metrics.py`

Purpose:

- generate captions on Clotho test
- evaluate caption metrics such as CIDEr / SPICE / SPIDEr through `aac_metrics`

Environment notes:

- typically intended for the remote `audio_eval` env
- may require extra handling because `torchmetrics` can trigger implicit `torchvision` imports

### 2. Modality-alignment evaluation suite

Shared helper:

- `code/recurrent-pretraining-main/audio_alignment_eval_common.py`

Scripts:

- `code/recurrent-pretraining-main/eval_vocab_retrieval.py`
- `code/recurrent-pretraining-main/eval_audio_text_retrieval.py`
- `code/recurrent-pretraining-main/eval_visualization.py`

Purpose:

1. **Vocabulary retrieval**
   - inspect which text tokens the pooled audio embedding is closest to

2. **Audio-text retrieval**
   - quantify whether the adapter pulls matched audio and caption embeddings closer in text embedding space
   - supports comparing checkpoints

3. **Visualization**
   - UMAP-based 2D projection of audio embeddings and caption embeddings

Current important convention:

- these scripts use **Clotho `test_expand.jsonl`**
- references are grouped by `audio_path`
- checkpoint input means a directory containing:
  - `trainable_state.pt`

---

## Important Masking / Attention Fix Already Landed

One important bugfix already made in this repo:

- `models/huginn-0125/raven_modeling_minimal.py` was updated so external `attention_mask` is actually compiled into Huginn's real attention path instead of being ignored.

This matters because previously:

- labels were masking pad positions correctly
- but Huginn self-attention was not using the external mask

Current fix:

- `compile_mask(...)` supports 2D and 3D masks
- causal masking and external valid-token masking are both respected
- Huginn `forward(...)` now uses the compiled mask instead of forcing `prepared_attn_mask = None`

This is important context if later training behavior changes after the mask fix.

---

## Current Active Files

If a new Codex / AI agent chat needs to start working immediately, the most relevant files are usually:

### Backbone / model logic

- `models/huginn-0125/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-v1/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-v1/raven_config_minimal.py`
- `models/huginn-audio-whisper-v1/_base.py`

### GSM8K full finetuning

- `code/recurrent-pretraining-main/finetuning_test_gsm8k_fsdp_5090.py`
- `code/recurrent-pretraining-main/local_scripts/train_huginn_full_gsm8k_fsdp_5090.sh`
- `code/recurrent-pretraining-main/run_train_huginn_full_gsm8k_fsdp_5090.sh`

### Audio training

- `code/recurrent-pretraining-main/finetuning_audio_whisper_smoke.py`
- `code/recurrent-pretraining-main/finetuning_audio_whisper_tiny_overfit.py`
- `code/recurrent-pretraining-main/finetuning_audio_whisper_clotho_aqa.py`
- `code/recurrent-pretraining-main/finetuning_audio_whisper_clotho_caption.py`
- `code/recurrent-pretraining-main/prepare_clotho_caption_expand.py`

### Audio evaluation

- `code/recurrent-pretraining-main/audio_alignment_eval_common.py`
- `code/recurrent-pretraining-main/eval_vocab_retrieval.py`
- `code/recurrent-pretraining-main/eval_audio_text_retrieval.py`
- `code/recurrent-pretraining-main/eval_visualization.py`
- `code/recurrent-pretraining-main/eval_audio_whisper_clotho_caption_aac_metrics.py`

---

## How New Codex / AI Chats Should Behave

Any new chat should assume the following:

1. This repo is a **sync repo**, not the full remote runtime filesystem.
2. Codex is **local-only** unless the user explicitly pastes remote outputs back.
3. Remote Linux facts must not be guessed if they are important.
4. Long remote jobs should be launched through the existing `run_*.sh` submit scripts.
5. Windows local paths and Linux remote paths must never be mixed up.
6. The project is no longer only about GSM8K:
   - there is now a substantial **audio branch**
7. The current audio branch is:
   - original Huginn backbone
   - Whisper-small encoder
   - frozen backbone + frozen encoder
   - trainable compressor/projector
8. The current audio project already has:
   - smoke training
   - tiny overfit
   - full AQA training
   - caption continuation training
   - alignment evaluation scripts

---

## Suggested Operating Rules For Future Chats

1. Be explicit about whether a path/command is **local Windows** or **remote Linux**.
2. Prefer giving the user exact remote commands instead of vague instructions.
3. Do not assume remote file contents unless the user has synced or shown them.
4. Do not suggest committing weights, checkpoints, or outputs into Git.
5. When editing scripts, keep the queue rule in mind:
   - single GPU jobs must stay within `8 CPU / 32G MEM`
6. For local work, prefer:
   - code edits
   - static syntax checks
   - README / script maintenance
7. If debugging remote runtime behavior, ask for:
   - log snippets
   - `grep` results
   - file listings
   - exact traceback lines

---

## Suggested Local-to-Remote Routine

1. edit locally in this repo
2. run `python -m py_compile` on changed Python files
3. `git status`
4. `git add/commit/push`
5. remote side: `git pull`
6. remote side: submit the corresponding `run_*.sh`
7. inspect `log/*.log`
8. if something fails, paste back:
   - traceback
   - related grep output
   - the exact checkpoint / dataset / script path used

---

## Last Known Practical Notes

- Current audio and GSM8K branches coexist in the same sync repo.
- The repo already contains both training and evaluation entrypoints for each line.
- The most likely future work is:
  - continue improving audio alignment / caption quality
  - evaluate checkpoints with retrieval / visualization / caption metrics
  - possibly add new audio datasets or unfreeze more modules in later stages

Before any long remote run:

- confirm the intended script is the latest synced version
- confirm the checkpoint path is the one you actually want
- confirm the output `run_name` will not collide with old runs
- confirm the queue resource request still follows the current rules

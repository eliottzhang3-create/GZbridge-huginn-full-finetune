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

It is also the **authoritative project memory** for future Codex / AI-agent chats:

- local-vs-remote path conventions
- active experiment goals
- current remote runtime assumptions
- what has already been debugged
- what is historical vs what is the current mainline

---

## Project Scope

This repo now contains **two active experiment lines**:

1. **Huginn full-parameter GSM8K finetuning**
   - based on FSDP
   - mainly for multi-GPU remote training
   - currently adapted for **8x RTX 5090**

2. **Huginn audio-modality experiment branch**
   - based on the **original Huginn backbone**, not the GSM8K-finetuned checkpoint
   - current codebase now contains **two sub-lines**:
     - the earlier standalone audio-training line in `code/recurrent-pretraining-main`
     - the newer **Swift-based LoRA multimodal line** in `code/huginn_lora`
   - current focus is **audio-to-text understanding/alignment**, not speech generation

### Current highest-priority task

The current main active task is:

- keep using the **original Huginn backbone**
- move the audio experiment toward a **Swift-based training pipeline**
- connect:
  - **Whisper-large encoder**
  - **adapter = temporal compressor + projector**
  - **Huginn text backbone**
- training policy for the new Swift line:
  - freeze `audio_encoder`
  - train `aligner` (`temporal_compressor`, `audio_projector`, optional audio boundary embeddings)
  - train **LoRA on Huginn backbone only**
  - do **not** LoRA-wrap or full-train the Whisper encoder

This Swift LoRA multimodal route is the current forward path for new audio-model work.

### Current highest-priority execution status (updated 2026-07-12)

The newest confirmed mainline is now:

- model family:
  - original Huginn backbone
  - multimodal model package in `models/huginn-audio-whisper-v1`
- audio encoder:
  - `whisper-large`
  - remote path:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large`
- framework:
  - **ms-swift / `swift sft`**
  - remote env version observed in logs:
    - `swift==4.1.3`
- training split that has been debugged and verified:
  - `audio_encoder` must stay **frozen**
  - `aligner` stays **full-trainable**
  - Huginn language model base weights stay **frozen**
  - Huginn language model gets **LoRA only**
- current dataset mainline for Swift audio work:
  - **ACAVCAPS**
  - loaded from shared public remote storage
  - training data is read from `.tar.gz` shards without copying the whole dataset into the personal workspace

The immediate practical mainline is no longer "just make Swift run once". It is now:

1. maintain the verified Swift audio training path
2. keep `audio_encoder` frozen under Swift multimodal registration
3. generate / maintain ACAVCAPS training manifests and chunked manifests
4. move toward formal training on ACAVCAPS using the verified chunked-manifest route
5. only let the user execute remote jobs through `vc submit`

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

- Remote Swift/LoRA code root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/huginn_lora`

- Remote model root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125`

- Remote audio experiment model root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-whisper-v1`

- Remote Whisper encoder root:
  - current mainline:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large`
  - historical / earlier audio branch:
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

### Current remote dataset / artifact roots that matter for the Swift audio line

- Public ACAVCAPS tar-shard root:
  - `/hpc_stor03/public/shared/data/raa/ACAVCAPS`
- Remote repo-side generated Swift dataset artifacts:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps`
- Current formal chunk output directory mainline:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/formal_chunks_all_4tar_256`

### Current remote tool assumptions already checked by logs / manual commands

- `python=3.10.20`
- system audio tools available:
  - `/usr/bin/ffmpeg`
  - `/usr/bin/ffprobe`
  - `/usr/bin/sox`
  - `/usr/bin/flac`
- Python TensorBoard package is available in `swift_huginn`:
  - `tensorboard==2.20.0`

Important note:

- the Swift audio plugin was extended to support **tar-backed FLAC decoding**
- Python audio backends such as `soundfile` / `torchaudio` were not assumed available
- current robust fallback path uses **`ffmpeg`** on the remote side

---

## Queue / Submission Constraints

The current queue in active use for the Swift audio line is:

- `pdgpu-3090`

Historical / sometimes used queue:

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

Important operational rule from the user:

- on the remote side, do **not** assume you can freely run arbitrary long commands interactively
- for practical work, always prepare:
  - a runtime shell script
  - a matching `vc submit` wrapper
- then let the user submit that job on the cluster

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
    huginn_lora/
      plugins/
        huginn_swift.py
        huginn_audio_swift.py
        huginn_swift_39.py
      scripts/
        train_huginn_sft_lora.sh
        train_huginn_scienceqa_lora.sh
        prepare_huginn_audio_dataset.py
        acavcaps_common.py
        inspect_swift_mllm_registration.py
        inspect_huginn_audio_swift_trainables.py
        inspect_huginn_audio_freeze_path.py
        inspect_acavcaps_dataset.py
        smoke_huginn_audio_swift.py
        smoke_huginn_audio_swift.sh
        smoke_acavcaps_huginn_audio_swift.py
        smoke_acavcaps_huginn_audio_swift.sh
        prepare_acavcaps_swift_dataset.py
        prepare_acavcaps_smoke_swift_dataset.sh
        prepare_acavcaps_pilot_swift_dataset.sh
        prepare_acavcaps_mid_swift_dataset.sh
        prepare_acavcaps_formal_chunked_swift_dataset.py
        prepare_acavcaps_formal_chunked_swift_dataset.sh
        prepare_acavcaps_formal_chunked_swift_dataset_limited.sh
        prepare_acavcaps_formal_full_chunked_swift_dataset.sh
        train_acavcaps_huginn_audio_swift_mid.sh
      run_smoke_huginn_audio_swift_5090.sh
      run_smoke_huginn_audio_swift_3090.sh
      run_inspect_swift_mllm_registration_5090.sh
      run_inspect_huginn_audio_swift_trainables_3090.sh
      run_inspect_huginn_audio_freeze_path_4090.sh
      run_inspect_acavcaps_dataset_3090.sh
      run_prepare_acavcaps_smoke_swift_dataset_3090.sh
      run_prepare_acavcaps_pilot_swift_dataset_3090.sh
      run_prepare_acavcaps_mid_swift_dataset_3090.sh
      run_prepare_acavcaps_formal_chunked_swift_dataset_limited_3090.sh
      run_prepare_acavcaps_formal_full_chunked_swift_dataset_3090.sh
      run_smoke_acavcaps_huginn_audio_swift_3090.sh
      run_train_acavcaps_huginn_audio_swift_mid_3090.sh
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

### Important historical split

The audio work now has **two stages** that must not be confused:

1. **Earlier standalone audio branch**
   - lives mainly in `code/recurrent-pretraining-main`
   - directly trains the custom Huginn-audio model with ordinary PyTorch scripts
   - was used to validate that:
     - audio prefix injection works
     - smoke test works
     - tiny overfit works
     - full ClothoAQA and caption continuation can run

2. **Current Swift multimodal LoRA branch**
   - lives mainly in `code/huginn_lora`
   - goal is to move to a more reusable **ms-swift training path**
   - current requirement is:
     - original Huginn backbone
     - Whisper-large encoder
     - adapter trainable
     - Huginn backbone LoRA trainable
     - audio encoder frozen

When the user says "current audio task", prefer to interpret it as the **Swift multimodal LoRA branch**, unless they explicitly refer to the older standalone training scripts.

### V1 architecture

Current experiment branch:

- audio encoder:
  - historical standalone branch:
    - **Whisper-small**
  - current Swift mainline target:
    - **Whisper-large**
- temporal compressor:
  - historical version:
    - Conv1d downsampling + normalization + activation + adaptive pooling
  - current updated design:
    - **Conv-GMLP-style compressor with shortcut path**
    - downsample with strided 1D conv branches
    - gate branch + feature branch
    - residual shortcut branch
    - final adaptive pool back to fixed token count
- audio projector:
  - project audio-side features into Huginn text hidden space
  - current implementation uses a **SwiGLU-style gated MLP projector**
- Huginn text backbone:
  - frozen in earlier V1 standalone branch
  - in the current Swift LoRA branch, backbone stays frozen at base weights but receives **LoRA adapters**

Historical standalone V1 training policy:

- freeze **Huginn backbone**
- freeze **Whisper encoder**
- train only:
  - `temporal_compressor`
  - `audio_projector`
  - optional `audio_bos`
  - optional `audio_eos`

Important clarification:

- the policy above describes the **earlier standalone adapter-only branch**
- it is **not** the current Swift mainline policy
- the current Swift mainline policy is:
  - freeze `audio_encoder`
  - full-train `aligner`
  - LoRA-train Huginn language model only

### Current architecture details that matter

For the current `models/huginn-audio-whisper-v1` implementation:

- Whisper output:
  - `last_hidden_state: [B, T_audio, hidden_audio]`
- compressor:
  - Conv-GMLP style temporal compression
  - current config target:
    - kernel size `7`
    - stride `12`
    - residual shortcut enabled
    - final `AdaptiveAvgPool1d(32)`
- projector:
  - LayerNorm
  - `w1`, `w2`
  - gated activation `w1(x) * SiLU(w2(x))`
  - `c_proj`
  - output LayerNorm
- boundary embeddings:
  - optional `audio_bos`
  - optional `audio_eos`
- final audio prefix:
  - prepended before text embeddings

### Important model files

- `models/huginn-audio-whisper-v1/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-v1/raven_config_minimal.py`
- `models/huginn-audio-whisper-v1/_base.py`

### Important Swift LoRA files

- plugin:
  - `code/huginn_lora/plugins/huginn_audio_swift.py`
- data conversion helper:
  - `code/huginn_lora/scripts/prepare_huginn_audio_dataset.py`
- lightweight manifest sanity check:
  - `code/huginn_lora/scripts/smoke_huginn_audio_swift.py`
- actual Swift smoke training launcher:
  - `code/huginn_lora/scripts/smoke_huginn_audio_swift.sh`
- current smoke submit scripts:
  - `code/huginn_lora/run_smoke_huginn_audio_swift_3090.sh`
  - `code/huginn_lora/run_smoke_huginn_audio_swift_5090.sh`

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

- ACAVCAPS shared public tar dataset:
  - `/hpc_stor03/public/shared/data/raa/ACAVCAPS`

- ACAVCAPS repo-side generated Swift manifests / chunk outputs:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps`

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

### Newest progress: Swift multimodal LoRA branch

On top of the earlier standalone audio branch, the repo has now entered a **new integration stage**:

1. **LoRA baseline code was synced into `code/huginn_lora`**
   - this provides the prior Huginn text-only Swift/LoRA baseline context

2. **A new multimodal Swift plugin was added**
   - file:
     - `code/huginn_lora/plugins/huginn_audio_swift.py`
   - purpose:
     - register the Huginn-audio model as a Swift multimodal model
     - register model arch split:
       - language model
       - aligner
       - frozen audio tower path
     - define a multimodal template that reads local audio and produces `audio_input_features`

3. **The new Swift route now targets `swift sft`, not ad-hoc manual forward loops**
   - this is important:
     - earlier intermediate attempts looked Swift-like but were not yet a true Swift multimodal training path
     - current code was rewritten specifically to align with the official Swift multimodal registration pattern

4. **Smoke-training entrypoints now exist for the Swift route**
   - prepare dataset into Swift JSONL
   - sanity print first sample
   - run a tiny `swift sft` smoke job through `vc submit`
   - current active single-GPU queue is mainly 3090

### Important current status of the Swift branch

- This branch is no longer only "implemented locally".
- Multiple remote validation stages have already succeeded.
- Therefore, the Swift multimodal LoRA path should currently be treated as:
  - **implemented locally**
  - **remote smoke-verified**
  - **remote trainability-verified on single 3090**
  - **still under active iteration for larger-scale formal ACAVCAPS training**

### Newest verified Swift progress (updated 2026-07-12)

The following points are already important confirmed project memory:

1. **Swift MLLM registration compatibility was debugged for the installed remote Swift version**
   - remote Swift version from logs:
     - `4.1.3`
   - `MultiModelKeys` registration path required compatibility handling
   - duplicate registration handling was added so repeated imports do not crash the pipeline

2. **The critical audio-encoder-freezing bug was found and fixed**
   - earlier logs showed the final Swift trainer model had:
     - `audio_encoder` trainable
     - total trainable params around `696M`
     - of which around `636M` wrongly came from the Whisper audio encoder
   - root cause:
     - the Swift multimodal model-arch split did not originally map our custom audio tower in the same way Swift expects frozen "generator/vision-tower-like" modules to be treated
   - fix:
     - the plugin now registers `audio_encoder` under the **`generator`** branch in the Swift model-arch split
   - result:
     - final validated route keeps `audio_encoder` frozen
     - aligner remains trainable
     - Huginn LoRA remains trainable

3. **The shift-loss patch remains important and is still in use**
   - `code/huginn_lora/plugins/huginn_audio_swift.py`
   - this patch is needed for the multimodal SFT label-shift behavior
   - an earlier monkey-patch debug hook did not intercept the exact internal Swift call path, but that did **not** mean the real shift-loss patch was unused

4. **Remote inspect / validation scripts were added and used**
   - Swift registration inspection
   - freeze-path inspection
   - final trainable-parameter inspection
   - ACAVCAPS tar / schema / decode inspection
   - these are now part of the active project memory and should be reused before future large changes

5. **Single-GPU smoke training now runs successfully**
   - Huginn audio Swift smoke route completed on remote
   - ACAVCAPS smoke route also completed
   - this proves:
     - plugin registration works
     - multimodal forward path works
     - loss path works
     - LoRA path works
     - tar-backed audio decode works

6. **Single-GPU mid-scale ACAVCAPS training also completed successfully**
   - a mid training run on 3090 finished successfully
   - observed memory was around `21.7 GiB`
   - this is important because it means the current mainline is no longer blocked at the runtime-validation stage

7. **OOM still matters for larger runs**
   - earlier attempts could OOM when the wrong parameter split left the audio encoder trainable
   - current formal-data work therefore uses manifest chunking and controlled sample counts per tar
   - the user currently prefers to stay on **single 3090** rather than immediately moving to multi-GPU / FSDP for this audio Swift line

### Current ACAVCAPS status and design

The current Swift audio mainline has moved from Clotho-only smoke work to **ACAVCAPS**.

Important ACAVCAPS facts:

- dataset is stored in the **public remote shared area**
- data is organized as category directories containing `.tar.gz` shards
- each shard contains paired:
  - `.flac`
  - `.json`
- the repo must **not** copy or rewrite the shared dataset in place
- the current training-data route reads those tar shards directly

Current implementation strategy:

1. inspect tar shard structure and decode support
2. build Swift JSONL records that reference:
   - `tar_path`
   - `audio_member`
   - `json_member`
3. let the plugin open tar members and decode FLAC on the fly
4. train through ordinary `swift sft`

This means:

- audio files are **not** eagerly copied into the repo workspace
- the manifest stores **tar-backed metadata**, not duplicated audio payloads
- decoding happens at training time

### Current ACAVCAPS manifest / chunk pipeline

There are now several different ACAVCAPS preparation layers and they must not be confused:

1. **Smoke manifest**
   - very small
   - used only to prove the full route runs end-to-end

2. **Pilot manifest**
   - larger than smoke
   - still for validation / sanity checks

3. **Mid manifest**
   - moderate-size training manifest
   - used to verify longer single-GPU training stability

4. **Formal chunk manifests**
   - used for the current mainline large-scale ACAVCAPS preparation
   - chunking exists to keep preparation resumable and easier to debug

### Current formal chunk mainline (updated 2026-07-12)

The current intended formal-manifest configuration is:

- use **all ACAVCAPS tar shards**
  - currently around `1071` tar files total across categories
- put **4 tar files per chunk**
- take the **first 256 samples per tar**
- generate one Swift JSONL file per chunk
- support resume / partial regeneration through:
  - `start_chunk`
  - `end_chunk`
  - `skip_existing`

Current default formal chunk output directory:

- `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/formal_chunks_all_4tar_256`

Current formal chunk entrypoints:

- runtime wrapper:
  - `code/huginn_lora/scripts/prepare_acavcaps_formal_full_chunked_swift_dataset.sh`
- submit wrapper:
  - `code/huginn_lora/run_prepare_acavcaps_formal_full_chunked_swift_dataset_3090.sh`

Separate smaller formal-scale validation route:

- select 56 tar shards:
  - `00A=12,0M0=8,S00=10,S0A=12,SMA=8,0MA=3,SM0=3`
- use the complete JSON sample set from every selected tar
- use one tar per chunk
- require a full sequential scan and verify every JSON has a same-stem `.flac` member
- do not set `FORMAL_SAMPLES_PER_TAR` for this route
- full 56-chunk record count:
  - `239854`
  - note: a later resumable job reported `235333` only because it processed chunk `001..055`; chunk `000` was completed separately with `4521` records
- output directory:
  - `data/audio_swift/acavcaps/subset_56_full_1tar_chunks`
- runtime wrapper:
  - `code/huginn_lora/scripts/prepare_acavcaps_subset_full_1tar_chunked_swift_dataset.sh`
- submit wrapper:
  - `code/huginn_lora/run_prepare_acavcaps_subset_full_1tar_chunked_swift_dataset_3090.sh`

Current formal chunk behavior:

- default category scope:
  - `ALL`
- default chunk size:
  - `4 tar / chunk`
- default sample cap:
  - `256 records / tar`
- default resume behavior:
  - `skip existing`

### Current formal-training configuration (updated 2026-07-13)

The formal ACAVCAPS training route uses the verified metadata-only master manifest:

- master manifest:
  - `data/audio_swift/acavcaps/acavcaps_subset_56_full_master_shuffled.jsonl`
- source records:
  - `239854` samples from the 56 full-tar subset chunks
- audio/caption integrity:
  - the master builder verifies each JSON caption, same-stem FLAC member, and tar membership before writing the master manifest
- active queue:
  - `pdgpu-5090`
- single-GPU formal configuration:
  - micro-batch: `8`
  - gradient accumulation: `4`
  - effective batch: `32`
  - `bf16=true`
  - audio encoder frozen
  - aligner full-trainable
  - Huginn language model LoRA-only
- data I/O configuration:
  - `HUGINN_AUDIO_TARFILE_CACHE_LIMIT=64`
  - this keeps all 56 gzip-tar indexes available in the single-worker loader and avoids the severe cache-thrashing observed with the old cache limit of four
- observability and recovery:
  - `report_to=tensorboard`
  - `logging_steps=10`
  - `save_steps=200`
  - `save_total_limit=2`
  - `save_only_model=false` so optimizer/scheduler/RNG state is available for resume
  - the runtime script prints a 30-second CPU RSS, cgroup-memory, and GPU-memory snapshot while training; this is required to diagnose external job termination without a Python traceback
- first complete-run target:
  - `max_steps=7500`, approximately one epoch at effective batch 32

This is the current mainline dataset-preparation route for the Swift ACAVCAPS project.

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

### Current Swift multimodal LoRA training intent

For the new `code/huginn_lora` path, the intended training split is:

- `audio_encoder`
  - frozen
- `aligner`
  - trainable full params
  - includes:
    - `temporal_compressor`
    - `audio_projector`
    - optional boundary embeddings
- `language_model`
  - base weights frozen
  - train **LoRA only**

This is the most important high-level requirement for any future edit on the Swift branch.

### Current Swift audio training status that should be assumed by new agents

As of 2026-07-12, the correct assumption is:

- the Huginn Swift audio route is **already runnable**
- the frozen-audio-encoder policy is **already enforced in the current mainline**
- the main unresolved work is **not** basic registration anymore
- the main unresolved work is:
  - scaling from smoke / mid training toward formal ACAVCAPS training
  - keeping dataset preparation resumable
  - managing larger training jobs carefully on single-GPU resources

### Current useful Swift training entrypoints

- Huginn/Clotho-style smoke:
  - `code/huginn_lora/scripts/smoke_huginn_audio_swift.sh`
- trainable-parameter validation:
  - `code/huginn_lora/scripts/inspect_huginn_audio_swift_trainables.sh`
- ACAVCAPS smoke:
  - `code/huginn_lora/scripts/smoke_acavcaps_huginn_audio_swift.sh`
- ACAVCAPS mid training:
  - `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_mid.sh`
- ACAVCAPS formal chunk generation:
  - `code/huginn_lora/scripts/prepare_acavcaps_formal_full_chunked_swift_dataset.sh`
- ACAVCAPS subset full-tar duration inspection:
  - `code/huginn_lora/scripts/inspect_acavcaps_subset_full_1tar_durations.sh`
- ACAVCAPS formal-manifest train probe:
  - `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_formal_probe.sh`
- ACAVCAPS subset full-tar master manifest preparation:
  - `code/huginn_lora/scripts/prepare_acavcaps_subset_full_master.sh`
- ACAVCAPS formal 100-step B4/GA4 stress test:
  - `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_formal_stress100.sh`

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

### Swift multimodal LoRA path

- `code/huginn_lora/plugins/huginn_audio_swift.py`
- `code/huginn_lora/scripts/prepare_huginn_audio_dataset.py`
- `code/huginn_lora/scripts/smoke_huginn_audio_swift.py`
- `code/huginn_lora/scripts/smoke_huginn_audio_swift.sh`
- `code/huginn_lora/run_smoke_huginn_audio_swift_5090.sh`
- `code/huginn_lora/run_smoke_huginn_audio_swift_3090.sh`
- `code/huginn_lora/scripts/inspect_huginn_audio_swift_trainables.py`
- `code/huginn_lora/scripts/inspect_huginn_audio_freeze_path.py`
- `code/huginn_lora/scripts/inspect_acavcaps_dataset.py`
- `code/huginn_lora/scripts/prepare_acavcaps_swift_dataset.py`
- `code/huginn_lora/scripts/smoke_acavcaps_huginn_audio_swift.py`
- `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_mid.sh`
- `code/huginn_lora/scripts/prepare_acavcaps_formal_chunked_swift_dataset.py`
- `code/huginn_lora/scripts/prepare_acavcaps_formal_full_chunked_swift_dataset.sh`
- `code/huginn_lora/run_prepare_acavcaps_formal_full_chunked_swift_dataset_3090.sh`

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
   - historical standalone branch:
     - original Huginn backbone
     - Whisper-small encoder
     - frozen backbone + frozen encoder
     - trainable compressor/projector
   - current forward branch:
     - original Huginn backbone
     - Whisper-large encoder
     - trainable adapter
     - Huginn backbone LoRA
     - Swift multimodal training path
8. The current audio project already has:
   - smoke training
   - tiny overfit
   - full AQA training
   - caption continuation training
   - alignment evaluation scripts
   - and now also:
     - Swift multimodal plugin code
     - Swift-format dataset conversion helper
     - Swift smoke-training submit path
     - Swift freeze-path inspection scripts
     - Swift trainable-parameter validation scripts
     - ACAVCAPS tar-backed dataset path
     - ACAVCAPS smoke + mid training scripts
     - ACAVCAPS formal chunk generation scripts

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
8. Distinguish carefully between:
   - the older standalone audio scripts in `code/recurrent-pretraining-main`
   - the newer Swift multimodal LoRA route in `code/huginn_lora`
9. Do not forget that the Swift branch has already passed remote smoke and mid training; do not regress it back into an "unverified" mental model.
10. For current audio development requests, default to the **Swift multimodal LoRA path** unless the user explicitly asks to modify the older standalone scripts.
11. For the Swift audio line, prefer `pdgpu-3090` single-GPU submission scripts unless the user explicitly asks to move elsewhere.
12. For ACAVCAPS, remember that the current formal-data mainline is tar-backed chunk generation, not copying raw audio into the repo.

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
- The audio line itself now contains both:
  - a historical standalone PyTorch training route
  - a current Swift multimodal LoRA route
- The most likely future work is:
  - continue scaling the already-validated Swift multimodal LoRA route
  - continue improving audio alignment / caption quality
  - evaluate checkpoints with retrieval / visualization / caption metrics
  - compare finetuning strategies:
    - LoRA
    - broader finetuning if needed later
  - compare audio encoders:
    - Whisper-large
    - future alternatives such as LoSAtok if the project moves there
  - possibly add new audio datasets or unfreeze more modules in later stages

### Current immediate next-step expectation

If a new agent is asked "what should we do now", the best default interpretation is:

1. work on the **Swift multimodal LoRA audio branch**
2. keep:
   - original Huginn backbone
   - Whisper-large encoder
   - adapter trainable
   - Huginn LoRA trainable
   - audio encoder frozen
3. assume the current dataset mainline is **ACAVCAPS**
4. assume the current formal data-prep mainline is:
   - all tar shards
   - 4 tar per chunk
   - 256 samples per tar
   - resumable chunk generation
5. do local code edits only
6. let the user run all remote jobs and bring logs back

Before any long remote run:

- confirm the intended script is the latest synced version
- confirm the checkpoint path is the one you actually want
- confirm the output `run_name` will not collide with old runs
- confirm the queue resource request still follows the current rules
- confirm whether the job is:
  - smoke
  - mid
  - formal chunk generation
  - actual formal training

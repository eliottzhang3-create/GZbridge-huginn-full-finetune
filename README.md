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

This repo contains **two major experiment families**:

1. **Huginn full-parameter GSM8K finetuning**
   - historical FSDP work on the text model
   - adapted for multi-GPU RTX 5090 training

2. **Huginn audio-modality experiment branch**
   - based on the **original Huginn backbone**, not the GSM8K-finetuned checkpoint
   - current codebase contains:
     - earlier standalone PyTorch audio experiments in `code/recurrent-pretraining-main`
     - the current **Swift multimodal route** in `code/huginn_lora`
   - objective: audio-to-text understanding and modality alignment, not speech generation

### Current highest-priority tasks (updated 2026-07-22)

Two audio lines coexist and must remain strictly separate:

1. **Whisper-large FSDP full finetuning** uses frozen Whisper-large, a full-trainable aligner, and full-trainable Huginn under Swift FSDP2. The historical 8-GPU `checkpoint-2802` is an evaluation artifact, not a cross-world-size resume source. Do not infer the current remote job state without a user-supplied log.
2. **LoSATok Swift LoRA continuation** is the current single-GPU experimental line. It uses a frozen official LoSATok stack, a full-trainable aligner, and Huginn LoRA. Its three-epoch AudioCaps-v2 training is complete, and a one-epoch ClothoAQA warm-start from its epoch-1 checkpoint is also complete. Current immediate work is evaluation, beginning with MMAU `test_mini` for the ClothoAQA checkpoint.

The shared audio architecture is:

- frozen audio encoder: Whisper-large on the Whisper route, or full LoSATok on the LoSATok route
- trainable aligner: temporal compressor, audio projector, and audio boundary embeddings
- Huginn text backbone
- audio prefix of `audio_bos + 32 compressed audio tokens + audio_eos`, concatenated before text embeddings

There are two distinct Swift fine-tuning policies; do not confuse them:

- historical/currently usable LoRA route:
  - audio encoder frozen
  - aligner full-trainable
  - Huginn base frozen, Huginn LoRA trainable
- current FSDP full-training route:
  - audio encoder frozen
  - aligner full-trainable
  - Huginn backbone full-trainable

Whisper is never LoRA-wrapped or full-trained in either policy.

The equivalent rule for the new LoSATok LoRA branch is stricter: the complete official LoSATok stack, including its semantic and acoustic components, is always frozen. Only the new temporal compressor/projector/boundary embeddings and Huginn LoRA tensors may train.

### Current execution status

#### LoSATok Swift LoRA replacement branch: completed training and active evaluation

- The official LoSATok code and weights are remote-only assets; they are deliberately not committed to this sync repository.
- Remote LoSATok asset roots:
  - weights and local MiDasheng snapshot:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok`
  - copied official code:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/huginn_lora/LosatokCode`
- Required files were remote-checked:
  - `ckpts/semantic_encoder.pth`
  - `ckpts/losatok_kl1e-3.pth`
  - local `midashenglm/` Hugging Face snapshot
  - `LosatokCode/config/16k_16k_25Hz_losatok.yml`
- `torchaudio==2.11.0+cu128` was installed offline into `swift_huginn` from the matching CPython 3.10 Linux wheel. It matches `torch==2.11.0+cu128`; do not install LoSATok's complete upstream requirements or replace the working Swift Torch stack.
- The standalone remote encoder inspect passed:
  - the supplied 24 kHz WAV was resampled with torchaudio band-limited sinc to 16 kHz;
  - LoSATok emitted `semantic_emb`, `acoustic_emb`, and `unified_emb` with shape `[1, 77, 1280]` for a 3.089-second sample, about `24.93 Hz`;
  - both official checkpoints loaded with no missing or unexpected keys;
  - all original LoSATok parameters must nevertheless be explicitly frozen by the Huginn wrapper because the official model defaults leave about `171.8M` parameters trainable.
- The dedicated Swift integration is remote-verified:
  - `models/huginn-audio-losatok-v1/`
  - `code/huginn_lora/plugins/huginn_losatok_swift.py`
  - `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.py`
  - `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.sh`
  - `code/huginn_lora/run_inspect_huginn_losatok_swift_trainables_5090.sh`
- Completed remote validations:
  - final Swift `lora_llm` parameter inspection passed: LoSATok trainables `0`, aligner `47,224,608`, Huginn LoRA `12,541,440`, Huginn base `0`;
  - real AudioCaps one-update smoke passed at `B=1, GA=1`, with loss/backward and `20.51 GiB` peak memory;
  - real AudioCaps one-update smoke also passed at the formal micro-batch configuration `B=8, GA=4` (32 samples/update), with loss/backward and `22.42 GiB` peak memory;
  - full LoRA checkpoint save/resume validation passed: `checkpoint-1` was saved, inspected, restored into a new process, and produced `checkpoint-2`; both checkpoints contain 66 LoRA tensors, 20 aligner tensors, and `audio_bos/audio_eos`.
- LoSATok design decisions encoded in the wrapper:
  1. decode to mono 16 kHz and keep only the first 30 seconds;
  2. templates pad waveforms only for collation, while the model uses the stored sample mask to slice each waveform back to its true length before LoSATok;
  3. this per-example encoding is intentional because the official LoSATok encoder-forward does not apply an input attention mask, so batch padding could otherwise change representations;
  4. use `unified_emb` rather than the 128-dimensional low bottleneck output;
  5. use compressor stride `4`, then `AdaptiveAvgPool1d(32)`, because LoSATok is about 25 Hz and the Whisper stride `12` would over-compress short clips before the 32-token pool;
  6. preserve the official LoSATok load dtypes when Swift casts Huginn and the trainable aligner to BF16 (MiDasheng begins in BF16 while other official modules retain their own dtype); cast only the frozen encoder output at the compressor boundary.
- Formal LoSATok AudioCaps-v2 LoRA run: completed remotely.
  - runtime: `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_swift_5090.sh`
  - submit: `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_swift_5090.sh`
  - configuration: 3 epochs, `B=8`, `GA=4`, effective batch 32, Huginn/aligner LR `1e-4`, TensorBoard, dataset/DataLoader shuffle, first-30-second truncation, and one full checkpoint per epoch.
  - completed run root:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632`
  - known epoch checkpoints:
    - `checkpoint-2802`
    - `checkpoint-5604`
    - `checkpoint-8406`
- LoSATok ClothoAQA LoRA warm-start: completed remotely.
  - source checkpoint:
    - the LoSATok AudioCaps epoch-1 `checkpoint-2802` above
  - semantic rule: this is a **weight warm-start**, not a Trainer resume. The runtime sets `HUGINN_LOSATOK_INIT_ALIGNER_CHECKPOINT=<checkpoint>` and Swift receives `--adapters <checkpoint> --load_args false`; LoRA plus aligner weights are restored, while optimizer, scheduler, RNG, global step, and data position start fresh for ClothoAQA.
  - the plugin now strictly restores all `20` tensors in `vit.safetensors` before PEFT loads the `66` LoRA tensors; this includes `audio_bos` and `audio_eos`. It then re-enables the aligner while asserting LoSATok remains frozen.
  - runtime and submit scripts:
    - `code/huginn_lora/scripts/train_clotho_aqa_huginn_losatok_swift_5090.sh`
    - `code/huginn_lora/run_train_clotho_aqa_huginn_losatok_swift_5090.sh`
  - completed run checkpoint:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_clothoaqa_e1_warmstart2802_b8ga4_5090/v0-20260722-024418/checkpoint-659`
  - configuration: 1 epoch, `B=8`, `GA=4`, effective batch 32, LoRA/aligner LR `1e-4`, one epoch checkpoint, TensorBoard, and 10-second resource snapshots.
- Current LoSATok MMAU-mini evaluation target:
  - checkpoint: the ClothoAQA `checkpoint-659` above
  - submit script: `code/huginn_lora/run_eval_mmau_test_mini_losatok_swift_5090.sh`
  - output directory: `outputs/mmau_test_mini_losatok_clothoaqa_e1_checkpoint659`
  - no MMAU result has been supplied yet; do not claim a score.
- LoSATok evaluation restore rules:
  - caption generation and MMAU restore both LoRA (`66` tensors) and aligner (`20` tensors);
  - retrieval restores the aligner only because its definition pools encoder/projector tokens and raw Huginn input embeddings without running LoRA-modified recurrent blocks.

#### Verified Whisper end-to-end multimodal chain

- framework: `swift==4.1.3`, using `swift sft`
- model package: `models/huginn-audio-whisper-v1`
- Whisper-large: `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large`
- AudioCaps-v2 manifest:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl`
  - `89658` verified unique WAV-caption records
  - `1599` CSV rows excluded (`3` empty IDs, `1596` missing WAVs)
  - every included WAV was checked as readable mono, 32 kHz, 16-bit PCM
- actual training path, verified by the plugin audit:
  1. decode WAV/FLAC and retain at most the first 30 seconds;
  2. Whisper feature extractor creates `[B, 80, 3000]` features;
  3. frozen Whisper produces audio hidden states;
  4. temporal compressor produces `32` audio tokens;
  5. projector maps them into Huginn's `5280`-dimensional space;
  6. boundary embeddings form a `34`-token audio prefix;
  7. prefix plus Huginn text embeddings enter the recurrent Huginn model;
  8. plugin shift-loss performs next-token prediction with all audio-prefix labels masked as `-100`.
- Huginn recurrence remains native:
  - `mean_recurrence=32`
  - long-tail recurrence sampling remains enabled
  - only the final at-most `8` recurrent iterations build a gradient graph; earlier iterations use `no_grad`.

#### FSDP full-training route: completed validations

- requested topology: `pdgpu-5090`, `8x RTX 5090`, `-c 32 -m 256G -g 8 -n 1`
- audit-confirmed trainables:
  - audio encoder: `0`
  - aligner: `47,224,608`
  - Huginn backbone: `3,564,976,800`
  - full trainable total: approximately `3.612B`
- required FSDP2 mode:
  - `full_shard auto_wrap`
  - `fsdp_version=2`
  - `SHARDED_STATE_DICT`
  - FSDP activation checkpointing: `false`
  - ordinary Trainer/model gradient checkpointing: `false`
- why FSDP activation checkpointing is disabled:
  - Swift's FSDP2 preset enables native activation recomputation.
  - Huginn's recurrent forward path reuses integer step-state; recomputation triggered an autograd LongTensor version-counter error.
  - disabling FSDP activation checkpointing avoids that recomputation path. This is separate from saving on-disk training checkpoints.
- FSDP2 compatibility already implemented in `huginn_audio_swift.py`:
  - `HUGINN_AUDIO_FSDP2_NONPERSISTENT_ROPE=1` makes `freqs_cis` non-persistent so Accelerate does not incorrectly load a normal RoPE buffer as a DTensor.
  - `HUGINN_AUDIO_TRAIN_CHAIN_AUDIT=1` logs parameter groups, audio prefix shape, and shifted-loss evidence on the first batch.
- completed remote tests:
  - 1-step backward smoke passed after disabling FSDP activation checkpointing.
  - 20-step 8-GPU stability smoke passed with `exit_status=0`, around `53.8 s/update`, and about `26.3 GiB` GPU memory.
  - 8-GPU sharded save/resume passed: a saved `checkpoint-2` resumed in a fresh job and produced `checkpoint-3`, validating FSDP model, optimizer, scheduler, RNG, and Trainer state recovery.

#### Formal FSDP run and historical fresh-run plan

- runtime script:
  - `code/huginn_lora/scripts/train_audiocaps_v2_huginn_audio_swift_full_fsdp8.sh`
- submit wrapper:
  - `code/huginn_lora/run_train_audiocaps_v2_huginn_audio_swift_full_fsdp8_5090.sh`
- historical 8-GPU stage:
  - the formal 8-GPU run used micro-batch `1` per GPU and gradient accumulation `4`, global effective batch `32`
  - `2802` updates make one 8-GPU epoch
  - completed checkpoint:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_full_fsdp8_e2_b1ga4/v0-20260717-084419/checkpoint-2802`
- 7-GPU fresh-run plan (do not assume it is still active without logs):
  - initializes from the original Huginn audio model; it passes no `resume_from_checkpoint` and has no dependency on `checkpoint-2802`
  - keeps the same FSDP2 configuration, micro-batch `1`, and gradient accumulation `4`; global effective batch is `28`
  - `89658` samples produce `3203` optimizer updates per 7-GPU epoch and `6406` updates across 2 epochs
  - checkpoints are saved at the two exact epoch boundaries: `checkpoint-3203` and `checkpoint-6406`
  - Huginn LR `1e-5`; aligner LR `1e-4`
  - cosine schedule, warmup ratio `0.05`, weight decay `0.01`, max grad norm `1.0`
  - TensorBoard enabled, logging every `10` updates, resource monitor every `30` seconds
  - `save_only_model=false` so each FSDP checkpoint remains fully resumable if a same-world-size continuation is later requested.
- the runtime prechecks manifest statistics, Swift argument compatibility, a clean output directory, and at least `200 GB` free storage.
- same-world-size FSDP save/resume is remote-verified. Cross-world-size resume is deliberately not used by the current plan.
- FSDP sharded checkpoints must not be loaded as LoRA adapters. The current evaluators restore `pytorch_model_fsdp_0` directly through DCP, one tensor at a time, into an ordinary one-GPU model. Do not use an all-at-once full-weight merge: the 32G single-GPU queue cap kills that CPU-heavy operation. The streaming restore later completed a caption-generation run successfully.

#### Historical but relevant routes

- ACAVCAPS tar-backed LoRA curriculum route is validated historical infrastructure. It reads shared `.tar.gz` files directly without copying raw audio.
- AudioCaps-v2 LoRA run produced at least `checkpoint-5604` and `checkpoint-8406`; existing retrieval, caption-generation, and MMAU-mini scripts target this checkpoint format.
- WavCaps AudioSet-SL LoRA warm-start route:
  - shared read-only root: `/hpc_stor03/public/shared/data/raa/WavCaps`
  - `108056` verified FLAC-caption pairs prepared
  - warm-start source: AudioCaps `checkpoint-5604`
  - corrected checkpoints save all `20` aligner tensors, including `audio_bos` and `audio_eos`
  - do not claim its multi-epoch training completed without a final remote log.
- Historical planned LoRA continuation after WavCaps (not the active LoSATok continuation and not confirmed as executed):
  - start checkpoint:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_wavcaps_audioset_sl_e2_warmstart5604_b8ga4_5090/v0-20260715-101351/checkpoint-6754`
  - stage 1: one epoch of direct full concatenation of AudioCaps-v2 and Clotho-v2 caption records, both learning rates `5e-5`
  - stage 2: one epoch of ClothoAQA with `20%` caption replay, also both learning rates `5e-5`
  - mandatory preflight scripts:
    - `code/huginn_lora/run_inspect_clotho_continuation_inputs_5090.sh`
    - `code/huginn_lora/run_prepare_audiocaps_clotho_caption_mixture_5090.sh`
  - these scripts verify the `66` LoRA tensors, all `20` aligner tensors including boundary embeddings, Clotho training records/audio paths, and the resulting metadata-only caption mixture before any continuation training script is added.

The practical mainline is:

1. do not disturb any active Whisper-large FSDP job; its runtime state must be established from logs, not guessed;
2. treat the LoSATok AudioCaps-v2 and ClothoAQA training runs above as completed checkpoint sources;
3. evaluate LoSATok and Whisper checkpoints only with their matching model/plugin path;
4. keep FSDP checkpoint streaming evaluation separate from LoRA adapter checkpoint handling;
5. submit all remote work through matching `vc submit` wrappers.

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

- Remote LoSATok Huginn model package after Git synchronization:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/models/huginn-audio-losatok-v1`

- Remote LoSATok weights and official-code roots are not part of Git:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/models/LoSATok`
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/huginn_lora/LosatokCode`

- Remote Whisper encoder root:
  - Whisper route:
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

### Remote dataset / artifact roots that matter for the Swift audio line

- Public ACAVCAPS tar-shard root:
  - `/hpc_stor03/public/shared/data/raa/ACAVCAPS`
- Remote repo-side generated Swift dataset artifacts:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps`
- Historical formal subset chunk directory:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/subset_56_full_1tar_chunks`
- Historical formal curriculum master:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/acavcaps_subset_56_full_curriculum_ordered.jsonl`
- Personal AudioCaps v2 root (inspected and manifest-prepared):
  - `/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2`
  - layout: `train.csv` plus `train/*.wav`; `val` and `test` remain reserved for later evaluation
  - prepared train manifest:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/audiocaps_v2/audiocaps_v2_train_swift.jsonl`
  - valid records: `89658`; excluded source rows: `1599`
- Public WavCaps root (read-only; do not modify it):
  - `/hpc_stor03/public/shared/data/raa/WavCaps`
  - active AudioSet-SL FLAC directory:
    - `/hpc_stor03/public/shared/data/raa/WavCaps/audio/AudioSet_SL_flac`
  - source metadata:
    - `/hpc_stor03/public/shared/data/raa/WavCaps/json/AudioSet_SL.jsonl`
  - prepared Swift manifest:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/wavcaps_audioset/wavcaps_audioset_sl_train_swift.jsonl`
  - verified records: `108056`
- MMAU local development dataset root:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU test_mini`
  - file: `test_mini.parquet` (`1000` labeled samples)
  - this is the local development subset, not the hidden-answer formal test set

### Current remote tool assumptions already checked by logs / manual commands

- `python=3.10.20`
- system audio tools:
  - the active container has working `ffmpeg` and `ffprobe` (observed as `/opt/conda/bin/ffmpeg` and `/opt/conda/bin/ffprobe`)
  - the login host also exposed `/usr/bin/ffmpeg`, `/usr/bin/ffprobe`, `/usr/bin/sox`, and `/usr/bin/flac`
- Python TensorBoard package is available in `swift_huginn`:
  - `tensorboard==2.20.0`

Important note:

- the Swift audio plugin was extended to support **tar-backed FLAC decoding**
- `soundfile` is still unavailable in `swift_huginn`
- `torchaudio==2.11.0+cu128` is now installed and verified in `swift_huginn` specifically for the LoSATok branch; it must remain version-matched to `torch==2.11.0+cu128`
- the Whisper/tar route retains **`ffmpeg`** as its robust decoding fallback

---

## Queue / Submission Constraints

The current default queue for new Swift audio training and evaluation jobs is:

- `pdgpu-5090`

Historical scripts may still name:

- `pdgpu-3090`; do not select it by default unless the user explicitly asks.

Important queue rule from the user:

- the limit is **per requested GPU** for every `vc submit` job:
  - CPU cores per GPU must be `<= 8`
  - memory per GPU must be `<= 32G`
- therefore a `-g N` job must satisfy `-c <= 8*N` and `-m <= 32*N G`.

Therefore the standard single-GPU submit shape is:

- `-c 8 -m 32G -g 1 -n 1`

For **8 GPU** jobs, the current full-training submit script uses:

- `-c 32 -m 256G -g 8 -n 1`

which satisfies the per-GPU rule.

For the active **7 GPU** fresh FSDP training job, use:

- `-c 28 -m 224G -g 7 -n 1`

This requests four CPU cores and 32G memory per GPU, safely within the queue limit. Do not use `-g 7 -m 256G`, and do not use `-g 1 -m 64G`.

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
    huginn-audio-losatok-v1/
      _base.py
      raven_config_losatok.py
      raven_modeling_losatok.py
      config.json
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
        huginn_losatok_swift.py
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
        prepare_clotho_aqa_huginn_losatok_swift_dataset.sh
        train_audiocaps_v2_huginn_losatok_swift_5090.sh
        train_clotho_aqa_huginn_losatok_swift_5090.sh
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

### Whisper architecture details that matter

For the Whisper-specific `models/huginn-audio-whisper-v1` implementation:

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

### LoSATok architecture details that matter

For `models/huginn-audio-losatok-v1` and `huginn_losatok_swift.py`:

- input audio is decoded to mono 16 kHz, deterministically truncated to the first 30 seconds;
- LoSATok emits `unified_emb: [B, T, 1280]` at about 25 Hz;
- batch waveforms are padded only for collation. The wrapper slices every item back to its true length and encodes examples individually because the upstream LoSATok encoder-forward does not apply an input attention mask;
- trainable alignment path: LoSATok `unified_emb` -> stride-4 temporal compressor -> `AdaptiveAvgPool1d(32)` -> projector to Huginn width `5280` -> learned BOS/EOS boundaries;
- final prefix remains 34 tokens: `audio_bos + 32 audio tokens + audio_eos`;
- all official LoSATok modules stay frozen. Only compressor, projector, boundary embeddings, and Huginn LoRA tensors train.

### Important model files

- `models/huginn-audio-whisper-v1/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-v1/raven_config_minimal.py`
- `models/huginn-audio-whisper-v1/_base.py`

### LoSATok model replacement files

- `models/huginn-audio-losatok-v1/raven_modeling_losatok.py`
- `models/huginn-audio-losatok-v1/raven_config_losatok.py`
- `models/huginn-audio-losatok-v1/_base.py`
- `models/huginn-audio-losatok-v1/config.json`
- `code/huginn_lora/plugins/huginn_losatok_swift.py`
- `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.py`
- `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.sh`
- `code/huginn_lora/run_inspect_huginn_losatok_swift_trainables_5090.sh`
- `code/huginn_lora/scripts/smoke_huginn_losatok_swift.py`
- `code/huginn_lora/scripts/smoke_huginn_losatok_swift.sh`
- `code/huginn_lora/run_smoke_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/checkpoint_resume_huginn_losatok_swift.sh`
- `code/huginn_lora/run_checkpoint_resume_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/prepare_clotho_aqa_huginn_losatok_swift_dataset.sh`
- `code/huginn_lora/run_prepare_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/train_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/run_train_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/run_eval_huginn_losatok_text_retrieval_swift_5090.sh`
- `code/huginn_lora/run_generate_clotho_caption_samples_losatok_swift_5090.sh`
- `code/huginn_lora/run_eval_mmau_test_mini_losatok_swift_5090.sh`

This is a separate model type/template pair (`huginn_losatok_raven`, `huginn_losatok_text`). Do not substitute it into the Whisper plugin or reuse Whisper checkpoints as LoSATok aligner checkpoints.

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
  - prepared LoSATok Swift manifest:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/clotho_aqa/clotho_aqa_train_swift.jsonl`
  - companion stats:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/clotho_aqa/clotho_aqa_train_swift.jsonl.stats.json`

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
  - **remote trainability-verified on single 3090** for smoke and mid runs
  - **remote formal I/O-verified on single 5090** for B8/GA4 curriculum training
  - **usable as a stable historical training route; current work has moved to checkpoint evaluation**

### Newest verified Swift progress (updated 2026-07-13)

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
   - this established the historical tar-backed route was not blocked at runtime validation

7. **Formal 5090 memory and I/O behavior are now characterized**
   - earlier attempts could OOM when the wrong parameter split left the audio encoder trainable; that split is no longer the current route
   - the correct frozen-audio-encoder configuration uses about `24.14 GiB` on a 32-GiB RTX 5090 at micro-batch `8`, gradient accumulation `4`
   - a globally shuffled formal master caused severe gzip-tar cache thrashing and about `140 s/step`
   - the replacement curriculum master keeps records from each tar contiguous and disables Swift dataset/DataLoader shuffling
   - its 20-step 5090 validation completed normally at about `6.2 s/step`

### Historical ACAVCAPS status and design

ACAVCAPS was a validated Swift audio training route after the early Clotho-only smoke work. It is retained as reusable tar-backed infrastructure, not the current LoSATok continuation dataset.

Important ACAVCAPS facts:

- dataset is stored in the **public remote shared area**
- data is organized as category directories containing `.tar.gz` shards
- each shard contains paired:
  - `.flac`
  - `.json`
- the repo must **not** copy or rewrite the shared dataset in place
- the historical training-data route reads those tar shards directly

Historical implementation strategy:

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

### Historical ACAVCAPS manifest / chunk pipeline

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
   - used for the historical large-scale ACAVCAPS preparation
   - chunking exists to keep preparation resumable and easier to debug

### Historical formal chunk and master-manifest route (updated 2026-07-13)

The historical formal-training route is the verified 56-tar full-record subset:

- select 56 tar shards:
  - `00A=12,0M0=8,S00=10,S0A=12,SMA=8,0MA=3,SM0=3`
- use the complete JSON sample set from every selected tar
- use one tar per chunk
- require a full sequential scan and verify every JSON has a same-stem `.flac` member
- do not set `FORMAL_SAMPLES_PER_TAR` for this route
- full 56-chunk record count:
  - `239854`
  - note: a later resumable job reported `235333` only because it processed chunk `001..055`; chunk `000` was completed separately with `4521` records
- remote chunk output directory:
  - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/subset_56_full_1tar_chunks`
- runtime wrapper:
  - `code/huginn_lora/scripts/prepare_acavcaps_subset_full_1tar_chunked_swift_dataset.sh`
- submit wrapper:
  - `code/huginn_lora/run_prepare_acavcaps_subset_full_1tar_chunked_swift_dataset_3090.sh`

The chunks are preparation artifacts, not audio copies. Each JSONL row retains tar path, FLAC member name, JSON member name, category, and the selected caption. The public tar archives remain unchanged.

Historical note:

- an all-ACAVCAPS experimental route (`1071` tars, `4` tars/chunk, first `256` records/tar) exists in the repository for resumable preprocessing experiments
- it is **not** the current formal-training dataset and must not be substituted for the 56-tar curriculum master without an explicit new experiment decision

### Historical formal-training configuration (updated 2026-07-13)

The formal ACAVCAPS training route uses the verified metadata-only master manifest:

- historical curriculum master manifest:
  - `data/audio_swift/acavcaps/acavcaps_subset_56_full_curriculum_ordered.jsonl`
- companion stats file:
  - `data/audio_swift/acavcaps/acavcaps_subset_56_full_curriculum_ordered.jsonl.stats.json`
- source records:
  - `239854` samples from the 56 full-tar subset chunks
- audio/caption integrity:
  - the master builder verifies each JSON caption, same-stem FLAC member, and tar membership before writing the master manifest
- historical formal queue:
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
  - curriculum master category order: `00A,0M0,S00,S0A,0MA,SM0,SMA`
  - `dataset_shuffle=false`
  - `train_dataloader_shuffle=false`
  - `sortish_sampler=false`
  - `group_by_length=false`
  - these preserve tar-local order from the curriculum master so gzip tar members are read sequentially instead of randomly across shards
  - `HUGINN_AUDIO_TARFILE_CACHE_LIMIT=4` is sufficient because each tar is consumed contiguously
- exact sampler conclusion from remote Swift `4.1.3` source inspection:
  - `dataset_shuffle` is passed by `SwiftSft` to dataset loading
  - `train_dataloader_shuffle` is consumed by Swift Trainer's DataLoader construction
  - without the Swift override, the base Transformers Trainer would choose `RandomSampler` for a length-known dataset
  - all four ordering flags above are therefore required for this single-rank curriculum run
- observability and recovery:
  - `report_to=tensorboard`
  - `logging_steps=10`
  - `save_steps=200`
  - `save_total_limit=2`
  - `save_only_model=false` so optimizer/scheduler/RNG state is available for resume
  - the runtime script prints a 30-second CPU RSS, cgroup-memory, and GPU-memory snapshot while training; this is required to diagnose external job termination without a Python traceback
- completed I/O validation:
  - `20` steps at B8/GA4, `exit_status=0`, `6.2 s/step`, `24.14 GiB`
- next full-run target:
  - `max_steps=7500`, approximately one epoch at effective batch 32

This is historical validated ACAVCAPS infrastructure. Do not treat it as the active LoSATok training dataset unless the user explicitly returns to ACAVCAPS.

---

## Historical Standalone Audio Training Defaults

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

- the standalone Clotho scripts above are historical single-GPU scripts; they do not describe the current Swift FSDP route.
- current AudioCaps-v2 full training is distributed across eight 5090 GPUs through Swift's internal launch path.

### Swift multimodal training policies

All current Swift policies keep the same frozen-audio rule:

- `audio_encoder`: frozen
- `aligner`: full parameters trainable
  - `temporal_compressor`
  - `audio_projector`
  - `audio_boundary_embeddings`, including `audio_bos` and `audio_eos`

The Huginn language-model policy depends on experiment type:

- LoRA experiments: base Huginn frozen, Huginn LoRA trainable.
- FSDP full experiments: Huginn base parameters trainable, no LoRA tensors expected.

Do not change the audio encoder's policy without an explicit new experiment decision.

### Current Swift audio status that new agents must assume

As of 2026-07-22:

- Swift registration, tar/WAV decoding, audio-prefix insertion, shifted NTP loss, and audio-encoder freezing have all been remote-verified.
- single-GPU Whisper LoRA routes on ACAVCAPS/AudioCaps are historical validated baselines.
- the LoSATok single-GPU LoRA route has completed three AudioCaps-v2 epochs and one ClothoAQA continuation epoch; it is a current checkpoint-producing/evaluation line.
- 8-GPU Swift FSDP2 initialization, one-step backward, 20-step stability, and sharded checkpoint resume have passed.
- the formal 8-GPU run reached historical `checkpoint-2802` (epoch 1). A separate fresh 7-GPU plan exists, but its live remote status must be confirmed from logs.
- FSDP checkpoint evaluation is implemented in the existing Clotho retrieval, Clotho sample-generation, and MMAU-mini scripts. They stream DCP tensors directly from the original 8 shard files into a one-GPU model and never create a merged full-weight cache. Submit these one-GPU 5090 jobs sequentially, each with the queue-limited `8 CPU / 32G` request.
- all remote work is still launched through `vc submit`; Codex edits only this local sync repository.

### AudioCaps v2 routes (updated 2026-07-22)

- AudioCaps v2 is in personal remote storage, so it uses ordinary WAV paths rather than ACAVCAPS tar references.
- data preparation passed:
  - `91257` source CSV rows
  - `89658` valid unique WAV-caption records
  - `1599` excluded rows: `3` empty audio IDs and `1596` unavailable WAVs
  - every included WAV is verified as readable mono, 32 kHz, 16-bit PCM.
- historical LoRA baseline:
  - a five-epoch B8/GA4 5090 run produced at least:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604`
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406`
  - do not infer final five-epoch completion without its final remote log.
- completed LoSATok LoRA route:
  - model/package: `huginn_losatok_raven` with `huginn_losatok_text`
  - run root: `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632`
  - known checkpoints: `checkpoint-2802`, `checkpoint-5604`, `checkpoint-8406`
  - all three LoSATok checkpoints use the normal Swift LoRA layout: `adapter_model.safetensors` (66 LoRA tensors) plus `vit.safetensors` (20 aligner tensors, including boundaries).
- current full-parameter route:
  - starts from the original audio model, not a LoRA checkpoint
  - uses FSDP2 across eight 5090 GPUs
  - trains Huginn plus aligner while keeping Whisper frozen
  - formal schedule and scripts are defined in the top-level current-status section.

### Swift Clotho Retrieval Evaluation (updated 2026-07-22)

- Purpose: compare checkpoints on grouped Clotho caption retrieval. The existing evaluator supports both Whisper and LoSATok Swift LoRA checkpoints through the selected plugin path.
- Embedding definition follows the earlier standalone retrieval implementation:
  - audio: mean pool of `audio_encoder -> temporal_compressor -> audio_projector` tokens, excluding audio boundary embeddings
  - text: masked mean of raw Huginn input token embeddings for each caption, without recurrent hidden states
  - metric: cosine-similarity audio-to-text and text-to-audio Recall@1/5/10, MRR, positive/negative similarity gap, and failure examples
- This is an adapter-alignment metric: LoRA is intentionally not restored because neither side traverses LoRA-modified Huginn blocks. The evaluator restores the aligner; it must never evaluate with a randomly initialized compressor/projector.
- Current LoSATok checkpoints have `66` LoRA tensors and `20` aligner tensors, including `audio_bos/audio_eos`. The boundary embeddings are excluded from the pooled retrieval representation by definition, but still remain part of complete generation/MMAU restoration.
- LoSATok retrieval submit wrapper:
  - `code/huginn_lora/run_eval_huginn_losatok_text_retrieval_swift_5090.sh`
  - it currently targets AudioCaps LoSATok `checkpoint-5604` and `checkpoint-8406`; change its fixed checkpoint list deliberately for other comparisons.

### Current Evaluation Mainline (added 2026-07-15)

#### Direct audio-conditioned caption generation

- scripts:
  - `code/huginn_lora/scripts/generate_clotho_caption_samples_swift.py`
  - `code/huginn_lora/scripts/generate_clotho_caption_samples_swift.sh`
  - `code/huginn_lora/run_generate_clotho_caption_samples_swift_5090.sh`
- task:
  - load one AudioCaps checkpoint, sample Clotho audio, generate a caption, and print its five reference captions for manual comparison
- generation must use the custom manual decoder, not `model.generate()`:
  1. audio-prefill the direct Huginn-audio model with `use_cache=True`
  2. select the next token
  3. feed each later token with the cache's current sequence position
  4. stop at EOS (`65505`) or the configured token limit
- reason:
  - generic Hugging Face generation creates text-only positions before the audio prefix is injected, producing a RoPE length mismatch; manual prefill observes the true combined audio-plus-text length
- validated facts:
  - normal prefix length is `34` (`audio_bos + 32` compressed audio tokens `+ audio_eos`)
  - a prefill with `38` text prompt tokens correctly produced a cache length of `72`
  - cached next-token forward correctly advanced `72 -> 73`
  - two different audios produced non-identical next-token logits, confirming audio reaches the model
  - a successful sample from `checkpoint-8406` generated `a stream of water flows and splashes` for a water reference
- recurrence:
  - default generation uses the model configuration `mean_recurrence=32`
  - do not add a hard-coded lower recurrence value unless the user explicitly requests an experiment
- LoSATok generation support:
  - the same generic Python evaluator now branches on `MODEL_TYPE == huginn_losatok_raven`, sends 16 kHz waveform values plus masks, and restores both LoRA and aligner tensors.
  - submit wrapper: `code/huginn_lora/run_generate_clotho_caption_samples_losatok_swift_5090.sh`

#### MMAU `test_mini` evaluation

- dataset:
  - local labeled development split, `1000` rows in `test_mini.parquet`
  - rows contain embedded encoded-audio bytes, instruction, choices, reference answer, and `other_attributes` JSON metadata
  - embedded bytes are not always RIFF WAV; the evaluator decodes all rows through the plugin's ffmpeg-byte route rather than assuming WAV headers
- scripts:
  - environment inspect: `scripts/inspect_mmau_environment.py` and `run_inspect_mmau_environment_5090.sh`
  - five-sample smoke: `scripts/smoke_eval_mmau_test_mini_swift.py` and `run_smoke_eval_mmau_test_mini_swift_5090.sh`
  - resumable full mini evaluation: `scripts/eval_mmau_test_mini_swift.py`, `scripts/eval_mmau_test_mini_swift.sh`, and `run_eval_mmau_test_mini_swift_5090.sh`
  - LoSATok single-checkpoint submit wrapper: `code/huginn_lora/run_eval_mmau_test_mini_losatok_swift_5090.sh`
- scoring protocol:
  - this is multiple-choice evaluation, not free caption generation
  - for every complete answer choice, the custom evaluator computes its mean teacher-forced token log-probability conditioned on audio and prompt
  - it selects the highest-scoring complete choice and compares it against the labeled answer
  - metadata fields (`task`, `difficulty`, `sub-category`, etc.) are used for result aggregation, never passed to the model as answer hints
- runtime behavior:
  - full evaluation appends and `fsync`s a JSONL result per sample, then skips already completed IDs only when the saved run configuration matches
  - use distinct output directories for different checkpoints or recurrence values
  - `MMAU_NUM_STEPS` maps to the evaluator's `--num-steps`; unset means the default model recurrence
- current requested evaluation:
  - LoSATok ClothoAQA `checkpoint-659`:
    - `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_clothoaqa_e1_warmstart2802_b8ga4_5090/v0-20260722-024418/checkpoint-659`
  - output directory: `outputs/mmau_test_mini_losatok_clothoaqa_e1_checkpoint659`
  - no score has been supplied, so README must not claim a winner.
- formal MMAU note:
  - mini is for local development and has answers
  - the formal hidden-answer set is a separate acquisition/submission step; final predictions must preserve the selected complete option text in the official submission JSON format

### Current useful Swift training entrypoints

- LoSATok AudioCaps-v2 formal LoRA (completed checkpoint source):
  - `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_swift_5090.sh`
  - `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_swift_5090.sh`
- LoSATok ClothoAQA manifest preparation:
  - `code/huginn_lora/scripts/prepare_clotho_aqa_huginn_losatok_swift_dataset.sh`
  - `code/huginn_lora/run_prepare_clotho_aqa_huginn_losatok_swift_5090.sh`
- LoSATok ClothoAQA one-epoch LoRA warm-start (completed checkpoint source):
  - `code/huginn_lora/scripts/train_clotho_aqa_huginn_losatok_swift_5090.sh`
  - `code/huginn_lora/run_train_clotho_aqa_huginn_losatok_swift_5090.sh`
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
- ACAVCAPS tar-local curriculum master preparation:
  - `code/huginn_lora/scripts/prepare_acavcaps_subset_full_curriculum_master.sh`
  - category order: `00A,0M0,S00,S0A,0MA,SM0,SMA`
  - this master keeps each tar's records contiguous and its pair verification passed
- ACAVCAPS formal 100-step B4/GA4 stress test:
  - `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_formal_stress100.sh`
- Swift Trainer sampler/shuffle source inspection:
  - `code/huginn_lora/scripts/inspect_swift_sampler_behavior.sh`
- historical formal 5090 runtime script:
  - `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_formal_5090.sh`
- historical formal 5090 submit wrapper:
  - `code/huginn_lora/run_train_acavcaps_huginn_audio_swift_formal_5090.sh`

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
- `models/huginn-audio-losatok-v1/raven_modeling_losatok.py`
- `models/huginn-audio-losatok-v1/raven_config_losatok.py`

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
- `code/huginn_lora/plugins/huginn_losatok_swift.py`
- `code/huginn_lora/scripts/prepare_clotho_aqa_huginn_losatok_swift_dataset.sh`
- `code/huginn_lora/scripts/train_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/run_prepare_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/run_train_clotho_aqa_huginn_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/inspect_losatok_encoder_remote.py`
- `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.py`
- `code/huginn_lora/scripts/inspect_huginn_losatok_swift_trainables.sh`
- `code/huginn_lora/run_inspect_losatok_encoder_remote_5090.sh`
- `code/huginn_lora/run_inspect_huginn_losatok_swift_trainables_5090.sh`
- `code/huginn_lora/scripts/acavcaps_common.py`
- `code/huginn_lora/scripts/prepare_huginn_audio_dataset.py`
- `code/huginn_lora/scripts/inspect_clotho_huginn_continuation_inputs.py`
- `code/huginn_lora/scripts/inspect_clotho_continuation_inputs.sh`
- `code/huginn_lora/run_inspect_clotho_continuation_inputs_5090.sh`
- `code/huginn_lora/scripts/prepare_audio_caption_mixture.py`
- `code/huginn_lora/scripts/prepare_audiocaps_clotho_caption_mixture.sh`
- `code/huginn_lora/run_prepare_audiocaps_clotho_caption_mixture_5090.sh`
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
- `code/huginn_lora/scripts/prepare_acavcaps_subset_full_1tar_chunked_swift_dataset.sh`
- `code/huginn_lora/scripts/prepare_acavcaps_subset_full_master.py`
- `code/huginn_lora/scripts/prepare_acavcaps_subset_full_curriculum_master.sh`
- `code/huginn_lora/scripts/inspect_swift_sampler_behavior.py`
- `code/huginn_lora/scripts/train_acavcaps_huginn_audio_swift_formal_5090.sh`
- `code/huginn_lora/run_train_acavcaps_huginn_audio_swift_formal_5090.sh`
- `code/huginn_lora/scripts/inspect_audiocaps_v2_dataset.py`
- `code/huginn_lora/scripts/prepare_audiocaps_v2_swift_dataset.py`
- `code/huginn_lora/scripts/train_audiocaps_v2_huginn_audio_swift_5090.sh`
- `code/huginn_lora/run_inspect_audiocaps_v2_dataset_5090.sh`
- `code/huginn_lora/run_prepare_audiocaps_v2_swift_dataset_5090.sh`
- `code/huginn_lora/run_smoke_audiocaps_v2_huginn_audio_swift_5090.sh`
- `code/huginn_lora/run_train_audiocaps_v2_huginn_audio_swift_5090.sh`
- `code/huginn_lora/scripts/inspect_huginn_audio_swift_full_fsdp.py`
- `code/huginn_lora/scripts/inspect_huginn_audio_swift_full_fsdp7.sh`
- `code/huginn_lora/run_inspect_huginn_audio_swift_full_fsdp7_5090.sh`
- `code/huginn_lora/scripts/inspect_swift_fsdp2_launch_path.py`
- `code/huginn_lora/run_inspect_swift_fsdp2_launch_path_5090.sh`
- `code/huginn_lora/scripts/inspect_accelerate_fsdp2_huginn_compat.py`
- `code/huginn_lora/run_inspect_accelerate_fsdp2_huginn_compat_5090.sh`
- `code/huginn_lora/scripts/smoke_audiocaps_v2_huginn_audio_swift_full_fsdp7.sh`
- `code/huginn_lora/run_smoke_audiocaps_v2_huginn_audio_swift_full_fsdp7_5090.sh`
- `code/huginn_lora/scripts/train_audiocaps_v2_huginn_audio_swift_full_fsdp8.sh`
- `code/huginn_lora/run_train_audiocaps_v2_huginn_audio_swift_full_fsdp8_5090.sh`
- `code/huginn_lora/scripts/inspect_wavcaps_audioset_dataset.py`
- `code/huginn_lora/scripts/prepare_wavcaps_audioset_swift_dataset.py`
- `code/huginn_lora/scripts/smoke_wavcaps_audioset_huginn_audio_swift_5090.sh`
- `code/huginn_lora/scripts/train_wavcaps_audioset_huginn_audio_swift_5090.sh`
- `code/huginn_lora/run_inspect_wavcaps_audioset_dataset_5090.sh`
- `code/huginn_lora/run_prepare_wavcaps_audioset_swift_dataset_5090.sh`
- `code/huginn_lora/run_smoke_wavcaps_audioset_huginn_audio_swift_5090.sh`
- `code/huginn_lora/run_train_wavcaps_audioset_huginn_audio_swift_5090.sh`
- `code/huginn_lora/scripts/inspect_swift_huginn_audio_checkpoints.py`
- `code/huginn_lora/scripts/eval_huginn_audio_text_retrieval_swift.py`
- `code/huginn_lora/run_inspect_swift_huginn_audio_checkpoints_5090.sh`
- `code/huginn_lora/run_eval_huginn_audio_text_retrieval_swift_5090.sh`
- `code/huginn_lora/run_eval_huginn_losatok_text_retrieval_swift_5090.sh`
- `code/huginn_lora/scripts/generate_clotho_caption_samples_swift.py`
- `code/huginn_lora/scripts/generate_clotho_caption_samples_swift.sh`
- `code/huginn_lora/run_generate_clotho_caption_samples_swift_5090.sh`
- `code/huginn_lora/run_generate_clotho_caption_samples_losatok_swift_5090.sh`
- `code/huginn_lora/scripts/inspect_mmau_environment.py`
- `code/huginn_lora/scripts/inspect_mmau_environment.sh`
- `code/huginn_lora/run_inspect_mmau_environment_5090.sh`
- `code/huginn_lora/scripts/smoke_eval_mmau_test_mini_swift.py`
- `code/huginn_lora/scripts/smoke_eval_mmau_test_mini_swift.sh`
- `code/huginn_lora/run_smoke_eval_mmau_test_mini_swift_5090.sh`
- `code/huginn_lora/scripts/eval_mmau_test_mini_swift.py`
- `code/huginn_lora/scripts/eval_mmau_test_mini_swift.sh`
- `code/huginn_lora/run_eval_mmau_test_mini_swift_5090.sh`
- `code/huginn_lora/run_eval_mmau_test_mini_losatok_swift_5090.sh`

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
   - separate Whisper Swift branch:
     - original Huginn backbone
     - Whisper-large encoder
     - Swift multimodal training path
     - frozen audio encoder and full-trainable aligner
     - historical LoRA and separate FSDP full-parameter modes
   - current encoder-replacement LoSATok branch:
     - LoSATok with 16 kHz waveform input and `unified_emb` output
     - Swift LoRA registration/model/template code is locally implemented
     - complete LoSATok is frozen; only aligner plus Huginn LoRA train
     - standalone encoder inspection, Swift final parameter inspection, B1 and B8/GA4 real smoke, and LoRA checkpoint save/resume all passed
     - completed three AudioCaps-v2 epochs at `checkpoint-2802`, `checkpoint-5604`, and `checkpoint-8406`
     - completed one ClothoAQA warm-start epoch from LoSATok `checkpoint-2802` to ClothoAQA `checkpoint-659`
     - current immediate evaluation target is the ClothoAQA checkpoint on MMAU `test_mini`
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
     - AudioCaps v2 manifest preparation and formal training scripts
     - Swift FSDP2 launch/configuration compatibility inspection scripts
     - 8-GPU FSDP smoke, sharded-checkpoint resume validation, and formal-training submit scripts
     - WavCaps AudioSet-SL inspection, manifest-preparation, smoke, and warm-start training scripts
     - direct cache-aware Clotho caption generation scripts
     - Clotho embedding-retrieval evaluation scripts
     - MMAU environment inspection, smoke, and resumable full-mini evaluation scripts
     - LoSATok remote encoder inspection and Swift trainable-split inspection entrypoints

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
   - the newer Swift multimodal LoRA and FSDP route in `code/huginn_lora`
9. Do not forget that the Swift branch has already passed remote smoke and mid training; do not regress it back into an "unverified" mental model.
10. For Whisper full-training requests, default to the **Swift multimodal FSDP full-training path**. For encoder replacement requests, use the dedicated LoSATok Swift files; its inspect/smoke/checkpoint validation now passes, but do not modify the verified Whisper plugin or cross-load Whisper and LoSATok checkpoints.
11. For current Swift audio training and evaluation, use the `pdgpu-5090` submit wrappers unless an existing legacy smoke/preparation wrapper explicitly targets `pdgpu-3090`.
12. For ACAVCAPS, remember that its formal route is the pair-verified tar-backed curriculum master, not raw-audio copying and not the old globally shuffled master. It is historical infrastructure unless the user explicitly chooses ACAVCAPS again.
13. For audio generation and MMAU scoring, do not call generic Hugging Face `generate()` on the multimodal wrapper; use the repository's manual audio-prefill/cache path so RoPE positions include the audio prefix.

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
- The audio line itself now contains:
  - a historical standalone PyTorch training route
  - a validated Swift multimodal LoRA route
  - a validated Swift FSDP2 full-parameter route with separate Whisper checkpoint handling
  - a completed LoSATok AudioCaps-v2 LoRA run and completed LoSATok-to-ClothoAQA LoRA continuation
- The most likely immediate work is:
  - run and inspect MMAU-mini for LoSATok ClothoAQA `checkpoint-659`
  - run the matching LoSATok Clotho retrieval and qualitative Clotho caption-generation evaluations when requested
  - compare LoSATok checkpoints only after result logs are available
  - continue monitoring the independent Whisper FSDP line only when the user supplies its remote status

### Current immediate next-step expectation

If a new agent is asked "what should we do now", the best default interpretation is:

1. determine whether the request concerns the Whisper FSDP branch or the completed/current LoSATok Swift LoRA branch; do not silently mix their model files, plugins, datasets, or checkpoints
2. for the running Whisper FSDP branch, keep:
   - original Huginn backbone
   - Whisper-large encoder
   - aligner trainable
   - Huginn full-trainable under FSDP
   - audio encoder frozen
3. treat ACAVCAPS as a validated historical training route:
   - 56 selected tars, `239854` pair-verified records
   - curriculum order: `00A,0M0,S00,S0A,0MA,SM0,SMA`
   - tar-backed FLAC decode with shuffle disabled for sequential shard access
4. treat AudioCaps v2 as the latest LoSATok checkpoint-producing route and a separate Whisper FSDP dataset:
   - `89658` valid WAV-caption samples
   - LoSATok completed run root: `outputs/huginn_losatok_audiocaps_v2_train_e3_b8ga4_5090/v1-20260720-162632`
   - LoSATok checkpoints: `checkpoint-2802`, `checkpoint-5604`, `checkpoint-8406`
   - historical Whisper FSDP epoch-1 checkpoint exists at its separate FSDP output root; never load it through PEFT/LoRA code
5. treat ClothoAQA as the latest LoSATok continuation dataset:
   - completed output: `outputs/huginn_losatok_clothoaqa_e1_warmstart2802_b8ga4_5090/v0-20260722-024418/checkpoint-659`
   - it was initialized by adapter-plus-aligner weight warm-start from LoSATok AudioCaps `checkpoint-2802`, not by Trainer resume.
6. retain the evaluation routes as separate work:
   - Clotho retrieval checks adapter alignment
   - manual cached decoding checks qualitative caption behavior
   - MMAU mini scores multiple-choice audio understanding
7. for the current MMAU experiment, evaluate LoSATok ClothoAQA `checkpoint-659` through `run_eval_mmau_test_mini_losatok_swift_5090.sh`; no result should be assumed before logs are supplied.
8. do local code edits only; let the user run all remote jobs and bring logs back.

For LoSATok requests, encoder inspect, Swift parameter inspection, B1 and B8/GA4 smoke, checkpoint save/resume, formal AudioCaps-v2 LoRA training, and ClothoAQA warm-start training have passed. Keep all LoSATok checkpoints separate from Whisper checkpoints. For a new cross-dataset LoSATok continuation, use adapter-plus-aligner warm-start rather than `resume_from_checkpoint` unless the user explicitly wants to continue the same dataset/trainer state.

Before any long remote run:

- confirm the intended script is the latest synced version
- confirm the checkpoint path is the one you actually want
- confirm the output `run_name` will not collide with old runs
- for resumable evaluation, confirm the output directory has the intended matching run configuration
- confirm the queue resource request still follows the current rules
- confirm whether the job is:
  - smoke
  - mid
  - formal chunk generation
  - actual formal training
  - retrieval / generation / benchmark evaluation

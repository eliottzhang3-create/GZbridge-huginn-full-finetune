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

## Independent HRM-Text Audio Exploration (updated 2026-07-26)

This is an **additional experimental line with HRM-Text-1B as the recurrent text backbone**. It is not a rename,
continuation, replacement, or status update of any Huginn experiment below. The Huginn and HRM-Text lines have separate
owners, environments, model code, checkpoints, outputs, and progress records.

### Ownership and isolation rules

- HRM-Text work is owned under `code/HRM_Audio/` and `models/hrm-text-audio-v1/` only.
- `code/huginn_lora/`, `code/recurrent-pretraining-main/`, and `models/huginn-*` are read-only design references for this
  line. Do not modify them while implementing HRM-Text, and do not report HRM results as Huginn results.
- The Huginn `Project Scope`, priority list, and current-status sections below remain authoritative for Huginn only. This
  section is the authoritative current-status record for the independent HRM-Text line.
- Model weights, checkpoints, generated outputs, and large logs remain remote-only and must not be committed to Git.

### Goal and fixed training policy

- Goal: attach the proven audio encoder/aligner pattern to HRM-Text for audio-to-text understanding, then train through
  **ms-swift**, not through a separate standalone training route.
- First audio tower: frozen local Whisper-large encoder, trainable temporal compressor, trainable audio projector, and
  trainable `audio_bos`/`audio_eos` boundary embeddings.
- First compression policy: mono 16 kHz, first 30 seconds, fixed 32 compressed audio tokens; the complete prefix is
  `audio_bos + 32 audio tokens + audio_eos` (34 tokens).
- Final trainability policy: Whisper-large fully frozen; HRM-Text base fully frozen; aligner fully trainable; LoRA on both
  HRM H/L stacks trainable. Swift `lora_llm` and the dedicated multimodal `model_arch` implement this split.
- The Huginn aligner architecture is a reference, not a shape-compatible checkpoint. Huginn uses hidden size `5280`,
  while HRM-Text-1B uses `1536`; the HRM audio projector output layer must be newly initialized and trained.

### Local, remote, and environment layout

- Local/remote synchronized code: `code/HRM_Audio/`.
- Wrapper-model directory: `models/hrm-text-audio-v1/`. It contains `configuration_hrm_text_audio.py`,
  `modeling_hrm_text_audio.py`, `config.json`, and package exports, and has passed the dedicated remote wrapper and
  generation/cache audits.
- Remote repository root:
  `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune`.
- Remote HRM-Text snapshot: `/hpc_stor03/sjtu_home/jinwei.zhang/models/HRM-text`; it contains `config.json`,
  `model.safetensors`, tokenizer files, `README.md`, and `LICENSE`. The weight SHA-256 verified by the load audit is
  `f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584`.
- Verified remote Whisper asset: `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large`.
- Dedicated remote conda environment: `swift_HRM`, cloned from `swift_huginn` and then independently updated. Verified
  versions are `ms-swift==4.4.2`, `transformers==5.9.0`, `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`,
  `torchvision==0.26.0+cu128`, `accelerate==1.13.0`, `peft==0.18.1`, and `trl==0.29.1`.
- Verified accelerator: one RTX 5090 (`31.367 GiB`, BF16 supported). Existing `code/HRM_Audio/run_*_5090.sh` launchers
  submit the corresponding scripts on the remote cluster.

### Verified text-only HRM-Text baseline

- Environment, CUDA/SDPA, package consistency, native HRM imports, local snapshot integrity, BF16 model loading, and native
  generation have passed remotely. The checkpoint has `1,182,795,264` parameters and loads fully on `cuda:0`.
- `code/HRM_Audio/plugins/hrm_text_swift.py` registers native model type `hrm_text_native` and PrefixLM-aware templates
  `hrm_text_direct` and `hrm_text_synth_cot` in ms-swift 4.4.2. Registration, template/collator encoding, Swift model
  loading, prefill equivalence, and deterministic generation have passed.
- HRM-Text-1B runtime recurrence is `L,L,L,H,L,L,L,H`: `H_cycles=2`, `L_cycles=3`, 16 physical layers in each stack.
  The current downstream SFT policy is static final `K=5`, encoded by `L_bp_cycles=[0,3]`; no K=2-to-K=5 warmup is used.
- Next-token labels, Swift compact labels/`logits_to_keep`, manual shifted cross entropy, PrefixLM masks, recurrence order,
  gradient-enabled trailing K steps, and H/L gradient coverage have all passed dedicated audits.
- Text-only rank-8 LoRA discovery/injection has passed: 256 target modules total (128 H, 128 L), 512 saved adapter
  tensors, and `8,257,536` trainable parameters; all original HRM parameters remain frozen.
- The official one-step `SwiftSft`/Trainer smoke has passed end to end: real collation, loss/backward, optimizer update in
  both H and L LoRA stacks, checkpoint-1 save, fresh Swift trainable reload, and checkpoint integrity checks.
- Reload validation requires exact adapter tensors/dtypes, exact frozen-base parameters, exact buffers, and exact runtime
  semantics. Each independently allocated model is self-repeat bitwise deterministic. Cross-instance BF16 logits are
  accepted with an epsilon-derived numerical envelope plus 100% per-position top-1 agreement; do not reintroduce an
  invalid `1e-5` bitwise cross-instance logits requirement for recurrent CUDA BF16 GEMMs.

### HRM PrefixLM audio contract

- Planned combined sequence: `[audio_bos, audio_1 ... audio_32, audio_eos, HRM prompt, response]`.
- Audio plus prompt positions use `token_type_ids=1` and form the bidirectional PrefixLM block. Response/EOS positions use
  `token_type_ids=0` and remain causal.
- Audio and prompt labels are `-100`; only response/EOS tokens are next-token targets. Audio padding, if introduced later,
  must also use attention mask `0` and label `-100`.
- The wrapper must preserve Swift 4.4.2 compact-label/`logits_to_keep` behavior: multi-sample batches use an integer suffix
  length, while a single sample uses a boolean text-position mask that must be left-padded with 34 `False` audio positions.
  During generation it must encode and prepend audio only on the initial prefill, never again during cached token decoding.

### Next implementation gates

1. The first local Whisper-large fixed-32 `HrmTextAudioForConditionalGeneration` and config are implemented in
   `models/hrm-text-audio-v1/`, reusing Transformers 5.9.0 native HRM classes rather than copying or modifying HRM loops.
   The published checkpoint stores fused `gqkv_proj`/`gate_up_proj` tensors, so initialization must first use native
   `HrmTextForCausalLM.from_pretrained` (which performs the official conversion) and only then upgrade that same loaded
   instance with the audio modules. Direct base-checkpoint loading under the custom audio `model_type` is invalid because
   it bypasses the HRM conversion and leaves the H/L backbone missing.
2. The wrapper-only audit in `code/HRM_Audio/scripts/inspect_hrm_audio_wrapper.py` has passed remotely: exact text-only
   passthrough, strict native-converted HRM/Whisper loading, frozen Whisper, `[B,34,1536]` audio prefix, full and compact
   labels/NTP shift, unchanged static K=5 recurrence, finite audio loss, and aligner-only gradients were verified before
   LoRA injection.
3. The audio generation/cache audit in `code/HRM_Audio/scripts/inspect_hrm_audio_generation.py` has passed remotely:
   one-time audio prefill, exact cache-length growth, causal response token types/positions, no repeated audio encoding
   during decode, and agreement between manual cached greedy tokens and `model.generate` were verified.
4. The independent multimodal Swift plugin is implemented in `code/HRM_Audio/plugins/hrm_text_audio_swift.py` with model
   type `hrm_text_audio_whisper`, template `hrm_text_audio`, a dedicated processor/loader, and model arch
   `hrm_text_audio_whisper`. Its `MultiModelKeys` groups are language model `model`/`lm_head`, trainable aligner modules,
   and frozen generator `audio_encoder`. The registration/encoding/load/audio-prefill audit has passed remotely via
   `code/HRM_Audio/scripts/inspect_hrm_audio_swift_registration.py` and
   `code/HRM_Audio/run_inspect_hrm_audio_swift_registration_5090.sh`. The verified text-only model type/templates remain
   separate and were not replaced.
5. The independent Swift `lora_llm` trainability audit and 5090 launcher are implemented in
   `code/HRM_Audio/scripts/inspect_hrm_audio_swift_trainability.py` and
   `code/HRM_Audio/run_inspect_hrm_audio_swift_trainability_5090.sh`. This gate passed remotely with exact counts of
   Whisper `0`, HRM base `0`, aligner `39,538,176`, and rank-8 H/L LoRA `8,257,536` (256 modules, 128 per stack), with
   `47,795,712` total trainable parameters and no unclassified parameters.
6. The real-audio one-update Trainer gate is implemented in
   `code/HRM_Audio/scripts/smoke_hrm_audio_swift_trainer.py`, with fresh-process reload in
   `code/HRM_Audio/scripts/reload_hrm_audio_swift_checkpoint.py` and submission through
   `code/HRM_Audio/run_smoke_hrm_audio_swift_trainer_5090.sh`. It uses the first two distinct records from the verified
   `89,658`-record AudioCaps-v2 train manifest through an HRM-specific metadata-only view: WAV, user prompt, assistant
   caption, and metadata are preserved, while the Huginn-specific system message is removed because the verified HRM
   direct template intentionally does not support system prompts. The source manifest is unchanged. The gate audits the actual native
   PrefixLM mask, compact-label NTP loss, static K=5 recurrence, aligner/H/L LoRA gradients and updates, exact frozen
   Whisper/HRM hashes, `512` LoRA plus `20` aligner checkpoint tensors, and a second-process reload. This B2/GA1/rank-8
   gate has passed remotely, including exact cross-process persistent-state/runtime checks and bounded long-PrefixLM BF16
   numerical checks.
7. The final-configuration one-update smoke is submitted independently through
   `code/HRM_Audio/run_smoke_hrm_audio_swift_trainer_formal_config_5090.sh`. It fixes B8/GA4 (effective batch `32`), rank-16
   H/L LoRA (`alpha=32`, effective dropout `0.0`), and LoRA/aligner learning rates `1e-4`. The dropout value follows the
   actual ms-swift 4.4.2 `LoRALLMTuner` implementation, which does not forward the generic `lora_dropout` argument into
   PEFT (the Huginn `lora_llm` route has the same effective behavior). It selects 32 distinct real AudioCaps-v2
   records and audits four complete forward/backward micro-steps, three accumulation substeps, one optimizer/global step,
   per-micro-step PrefixLM/NTP/static-K5 semantics, rank-16 parameter/checkpoint counts, frozen weights, memory, and fresh
   reload. This final-configuration gate has passed remotely.
8. Stage 1 of formal data preparation is implemented in
   `code/HRM_Audio/scripts/prepare_audiocaps_v2_hrm_audio_manifest.py`, with the remote environment wrapper
   `code/HRM_Audio/scripts/prepare_audiocaps_v2_hrm_audio_manifest.sh` and cluster launcher
   `code/HRM_Audio/run_prepare_audiocaps_v2_hrm_audio_manifest_5090.sh`. It streams the verified `89,658`-record source
   manifest into the remote-only `data/audio_swift/audiocaps_v2/audiocaps_v2_train_hrm_audio.jsonl`, removing only the
   fixed generic system message and preserving the user prompt, caption, audio path, and metadata exactly. The gate
   requires exact schema/role/prompt/count/uniqueness checks, reopens every WAV as mono 32-kHz PCM16, hashes both source
   files before and after conversion, writes atomically, and accepts an existing output only when its hashes agree. The
   source Huginn manifest is read-only. This gate has passed remotely: all `89,658` records, audio paths, and sample IDs
   are unique; every WAV header passed; the source manifest SHA-256 is
   `5e1539480932d9348630c1007ba162977c03441f5c0bac4c2814cef99eb6270c`; and the derived HRM manifest SHA-256 is
   `e3f81f068710a32f903d8326973e73c6b156c1f0ad939647bb89d0550380caef`.
9. The former tiny-overfit and Trainer-resume gates are intentionally merged. The implementation is
   `code/HRM_Audio/scripts/smoke_hrm_audio_tiny_overfit_resume.sh`, with fixture/final auditing in
   `code/HRM_Audio/scripts/audit_hrm_audio_tiny_overfit_resume.py`, the fresh-process audited Swift resume in
   `code/HRM_Audio/scripts/resume_hrm_audio_tiny_overfit_swift.py`, and submission through
   `code/HRM_Audio/run_smoke_hrm_audio_tiny_overfit_resume_5090.sh`. It uses four real records from the verified
   HRM-specific manifest, repeated deterministically into exactly one formal effective batch of 32. Phase 1 runs
   `swift sft` for 12 optimizer steps at B8/GA4, rank-16/alpha-32, and `1e-4` LoRA/aligner learning rates. Phase 2 starts
   a separate process, requires exact checkpoint-12 LoRA/aligner values plus optimizer step 12, scheduler step 12, and
   Trainer global step 12 before any new update, then continues through checkpoint 24. The final gate requires complete
   `512`-LoRA/`20`-aligner checkpoints, continued H/L/aligner updates, optimizer/scheduler/RNG advancement, all 24 finite
   per-step losses, and at least a 10% reduction from the first-three-step mean to the last-three-step mean. It is
   implemented and has passed remotely. The exact checkpoint-12 boundary values, optimizer/scheduler step 12, and Trainer
   global step 12 were restored before resumed optimization; checkpoint 24 was complete; all 256 H-stack and 256 L-stack
   LoRA tensors continued updating; 18 of 20 aligner tensors changed; and the loss fell from a first-three-step mean of
   `4.427666` to a last-three-step mean of `0.002329` (`99.9474%` reduction). No separate throughput-only gate is planned.
10. Formal two-epoch AudioCaps-v2 training is implemented in
    `code/HRM_Audio/scripts/train_audiocaps_v2_hrm_audio_swift_5090.sh`, submitted through
    `code/HRM_Audio/run_train_audiocaps_v2_hrm_audio_swift_5090.sh`, with full-manifest preflight and epoch-checkpoint
    auditing in `code/HRM_Audio/scripts/audit_hrm_audio_formal_training.py`. It fixes B8/GA4 (effective batch `32`), two
    epochs, rank-16/alpha-32 H/L LoRA, effective LoRA dropout `0.0`, and `1e-4` LoRA/aligner learning rates; Whisper and
    the HRM base remain frozen. The `89,658`-record dataset gives exactly `2,802` optimizer steps per epoch and `5,604`
    total steps, with epoch checkpoints expected at `checkpoint-2802` and `checkpoint-5604`. Formal data loading uses
    Swift lazy tokenization so the full corpus does not materialize every `[80,3000]` Whisper feature tensor in host
    memory. Both dataset and DataLoader shuffling are enabled, checkpoints retain optimizer/scheduler/RNG state, and an
    optional resume is deliberately restricted to the complete epoch-1 checkpoint. The post-run audit requires both
    complete `512`-LoRA/`20`-aligner checkpoints, exact optimizer/scheduler steps, finite improving loss history, continued
    H/L/aligner updates in epoch 2, and RNG advancement. It is ready for its first remote launch.

**Current exact status:** the text-only HRM-Text Swift/LoRA/Trainer foundation is complete. The first audio wrapper has
passed remote forward/backward and generation/cache audits, and the independent multimodal Swift registration,
processor/template/collator, model load, and audio-prefill audit has also passed remotely. The exact `lora_llm`
trainability audit and the real B2/GA1/rank-8 AudioCaps-v2 one-update Trainer/save/fresh-reload smoke have passed remotely.
The B8/GA4/rank-16 final-configuration smoke, full HRM-specific AudioCaps-v2 manifest preparation, and combined
tiny-overfit plus fresh-process Trainer-resume gate have all passed remotely. There is no separate throughput-only stage.
The formal two-epoch AudioCaps-v2 Swift job and strict epoch-checkpoint audit are implemented and ready to launch; formal
training has not started yet.

**HRM-Text status update 2026-07-26 (this paragraph supersedes the immediately preceding
`ready to launch / not started` sentence for the independent HRM line only):** Formal two-epoch AudioCaps-v2
training completed remotely under Swift at B8/GA4 (effective batch `32`), rank-16/alpha-32 H/L LoRA, aligner learning
rate `1e-4`, and frozen Whisper/HRM base. The verified epoch checkpoints are:

- `.../audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-2802`
- `.../audio_audiocaps_v2_train_e2_b8ga4_r16_5090/20260726-084202/swift_output/v0-20260726-084236/checkpoint-5604`

An independent HRM MMAU `test_mini` evaluator is now implemented in
`code/HRM_Audio/scripts/eval_mmau_test_mini_hrm_audio_swift.py`, with local wrapper
`code/HRM_Audio/scripts/eval_mmau_test_mini_hrm_audio_swift.sh` and 5090 submitter
`code/HRM_Audio/run_eval_mmau_test_mini_hrm_audio_swift_5090.sh`. It uses the HRM Swift checkpoint reload path,
restores and audits `adapter_model.safetensors` plus `vit.safetensors`, decodes embedded MMAU audio bytes through
ffmpeg into Whisper features, scores complete choices by mean conditional token log-probability using the HRM
34-token audio-prefix/cache path, and writes one isolated resumable output directory per checkpoint. It does not
import or modify any Huginn evaluator. Default evaluation is both checkpoints. Use a separate output root for the
first smoke, for example `MMAU_MAX_SAMPLES=5 MMAU_OUTPUT_ROOT=.../outputs/hrm_text/mmau_test_mini_smoke`, then
unset `MMAU_MAX_SAMPLES` and use a fresh `.../mmau_test_mini_full` root for the complete 1000-row mini
evaluation; the strict run-config guard intentionally rejects mixing those ranges in one output directory. No MMAU
score has been reported yet.

An independent HRM Clotho-v2 qualitative generation evaluator is also implemented in
`code/HRM_Audio/scripts/generate_clotho_caption_samples_hrm_audio_swift.py`, with environment wrapper
`code/HRM_Audio/scripts/generate_clotho_caption_samples_hrm_audio_swift.sh` and 5090 submitter
`code/HRM_Audio/run_generate_clotho_caption_samples_hrm_audio_swift_5090.sh`. It reads the same read-only
`clotho_caption_huginn/test_expand.jsonl` evaluation convention used by the existing project data, selects the
same three deterministic audio groups for both checkpoints by default (seed `74`), prints generated captions beside
all reference captions, and saves one `clotho_caption_samples.json` per checkpoint. It restores HRM LoRA/aligner
weights through the Swift route, uses the direct HRM prompt and manual audio-prefill/cache decoding, and verifies
that the Whisper encoder and aligner execute exactly once per sample. This is a qualitative generation check; no
caption metric or score is inferred from the three samples.

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

For the current project handoff, Huginn is the primary line. The independent HRM-Text section above is preserved for
background and isolation, but it must not be mixed into Huginn model, data, plugin, or checkpoint decisions unless the
user explicitly switches lines.

### Authoritative current Huginn Whisper mainline (updated 2026-08-05)

The active Huginn formal-training route is **Whisper-large + dynamic-30s audio + 240ms/token + Swift FSDP4**. Some
filenames and environment variables still contain `dynamic90s` because the already-validated model/plugin and checkpoint
tooling were retained for compatibility; the current runtime semantics are dynamic-30s, not dynamic-90s. This route is
isolated from the historical fixed-32 Whisper route and from both LoSATok routes.

#### Current status snapshot (2026-08-05)

This is the authoritative handoff snapshot for the Huginn line. The project is currently in the Whisper-large
dynamic-30s/240ms-token phase; the old dynamic-90s names are compatibility names only.

- The active model path is the original Huginn-0125 backbone plus a trainable Whisper-large encoder and trainable audio
  aligner. The native Huginn backbone and LM head are frozen; only Huginn-side rank-8 LoRA is trainable on the language
  model. Whisper, aligner, audio BOS/EOS, and Huginn LoRA use the validated `1e-4` optimizer groups.
- The active audio contract is one mono 16-kHz chunk per record, first-30-second truncation for longer input, real
  dynamic length for shorter input, and one content token per `240ms`. There is no 90-second splitting or chunk
  concatenation in the current runtime.
- The four-GPU FSDP4 topology, SDPA attention, removal of Whisper's internal per-layer checkpointing, outer Whisper
  activation checkpointing, recurrent-core `reshard_after_forward=false`, real-data chain, sampler audit, acceleration
  Stage 0/1/2, Stage 3/4, Stage 5, and full-model four-rank save/cold-resume smoke have passed. These are validation
  gates; they are not themselves evidence that a long formal run has completed.
- The two formal data schedules are isolated: hierarchical AAC/ASR no-replacement multitask and finite globally shuffled
  multiplier. Their registries, plugins, statistics, plans, and checkpoints must never be mixed.
- Formal-training completion is not inferred from an intermediate loss line, step count, or an output directory. It
  requires the final success banner, retained-checkpoint audit, and cumulative training-statistics audit.
- The current finite multiplier run is still the active remote training job. The expected final
  `checkpoint-46050` had not yet been confirmed at the last handoff, so it must not be used as a warm-start source
  until the checkpoint directory and final formal audit exist. If the current run root is unchanged, the expected path
  is `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-46050`;
  verify this on Linux rather than assuming it exists.
- The latest confirmed intermediate multiplier checkpoint,
  `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-25000`,
  passed the read-only full-model FSDP DCP audit. This validates the source checkpoint contract, but it is not the
  requested final warm-start source and does not mean the multiplier run is complete.
- The complete ACAVCAPS flat manifest has now been generated and strictly audited: all `1071` tar files from all
  three source stages are flattened into one deterministic global tar permutation, with `4,664,169` JSON/FLAC pairs.
  The manifest is ready; it is not itself a training result.
- The current 8-card ACAVCAPS model-only warm-start/save/resume smoke is implemented but intentionally not launched
  until `checkpoint-46050` is available and formally verified.
- **X-ARES modality-alignment evaluation is explicitly unfinished.** Environment setup, checkpoint inspection, VoxCeleb1
  path audit, encoder synthetic/real smoke, and the API-contract gate have passed. The VoxCeleb1 K-NN run has not yet
  completed successfully, no X-ARES score is available, and no full evaluation result may be reported as completed.

#### Current formal schedules

Both formal jobs use the same model, optimizer ownership, loss, FSDP topology, audio decode contract, and checkpoint
contract. They differ only in their data registry/schedule, total steps, and checkpoint retention.

1. **Hierarchical multitask/no-replacement schedule**

   - Tasks: AAC `60%`, ASR `40%`.
   - AAC split: WavCaps without BBC Sound Effects `60%`, AudioCaps-v2 `30%`, Clotho-v2 train `10%`.
   - Pools: `wavcaps_no_bbc_aac`, `audiocaps_v2_aac`, `clotho_v2_aac`, and `gigaspeech_l_asr`.
   - Each pool is sampled without replacement within its current pool epoch. After exhaustion it is deterministically
     reshuffled for the next pool epoch. The seed and exact global position define deterministic arbitrary-position resume.
   - Formal script: `code/huginn_lora/scripts/train_huginn_audio_whisper_dynamic90s_multitask_fsdp4.sh`.
   - Schedule: `20000` optimizer steps, FSDP4 global batch `32`, `640000` scheduled samples, checkpoints at
     `5000/10000/15000/20000`. The latest retention requirement is to keep at most the two most recent checkpoints;
     the multiplier launcher already passes `save_total_limit=2`, but the multitask launcher currently still contains
     `--save_total_limit 4` and must be synchronized before a new multitask formal launch. Do not assume the live
     multitask job has two-checkpoint retention until its source script and launch log agree. The runtime final audit
     checks whether realized, decoded, first-30-second-capped duration exceeded `3000` hours.

2. **Finite multiplier/single-global-epoch schedule**

   - GigaSpeech-M `1x`, AudioCaps-v2 `3x`, Clotho-v2 train `3x`, WavCaps AudioSet `2x`, WavCaps SoundBible `2x`, and
     a deterministic quarter of WavCaps FreeSound `1x`.
   - All expanded occurrences are concatenated into one finite pool and globally shuffled once. Training then consumes
     that frozen permutation sequentially; dataset and dataloader reshuffling are disabled. A multiplier means that a
     source record occurs the requested number of times in the expanded schedule.
   - Current prepared schedule: `1,473,600` occurrences, `46050` optimizer steps at global batch `32`.
   - Formal script: `code/huginn_lora/scripts/train_huginn_audio_whisper_dynamic30s_multiplier_fsdp4.sh`.
   - Checkpoints are saved every `5000` steps and at the final step, while at most the most recent two are retained
     (`save_total_limit=2`).

   - Current execution status: the 4-card formal run is still in progress; `checkpoint-46050` is the required final
     source for the next ACAVCAPS warm-start phase and must be confirmed together with the formal terminal/audit
     output before starting that phase.

The supplied 2026-08-03 logs showed approximately `13 s/step` for the multiplier route and `39 s/step` for the
multitask route. These are runtime observations only; a run is complete only after its final success banner and formal
checkpoint/statistics audit.

#### Current ACAVCAPS flat global-tar route

This is the next dataset route for the **current Whisper-large dynamic-30s mainline**. It must not be confused with the
historical LoSATok ACAVCAPS-quarter routes documented later in this README.

- The source full preflight manifest is:
  `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps_wds/acavcaps_wds_stage_schedule_full_seed20260723.json`.
- The completed private flat manifest is:
  `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json`.
  Its companion stats file is the same path with `.stats.json`.
- The manifest contains exactly `1071` tars and `4,664,169` JSON/FLAC pairs. Source tar counts are `stage1=651`,
  `stage2=398`, and `stage3=22`; these fields are provenance only.
- Training order is one global permutation over the concatenation of all three source stages, using seed `20260723`.
  There are no stage-level training boundaries. Within each tar, WebDataset uses a runtime sample shuffle buffer of
  `512`. The public ACAVCAPS root remains read-only; only private manifest/stats files are generated.
- The flat-manifest preparation and strict preflight both passed, including source lineage, exact permutation,
  tar-count/sample-count consistency, category counts, and tar-file existence checks. The preflight's reported
  `145756` updates is only the nominal `ceil(4,664,169 / 32)` count for an 8-card `B=1, GA=4` configuration; it is
  not a completed training run.
- The current Whisper loader is
  `code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_acavcaps_swift.py`. It opens one tar at a time, performs
  the per-tar buffer shuffle, validates JSON/FLAC pairs and the non-empty `long` caption, and passes
  `tar_path + audio_member` metadata to the current Whisper template for lazy decoding. It deliberately does not add
  manual rank sharding because Accelerate's `DataLoaderDispatcher` owns rank-level batch dispatch for this route.
- For formal ACAVCAPS training, `ACAVCAPS_FLAT_MAX_TARS` must be unset so all `1071` tars are consumed. A positive
  `ACAVCAPS_FLAT_MAX_TARS` is allowed only for the bounded smoke test and is not a formal data configuration.

#### Current 4-card-to-8-card model-only warm-start plan

The user explicitly chose a new training, not a Trainer continuation of the 4-card multiplier job. Therefore the
semantics are:

1. Use the completed multiplier `checkpoint-46050` only as a full-model DCP **model-weight source**.
2. Restore Whisper encoder tensors, aligner tensors, and the `66` Huginn-only LoRA tensors. The frozen native Huginn
   backbone is rebuilt from the canonical Huginn-0125 base and is not copied as a continuation state.
3. Initialize a new optimizer covering only current trainable Whisper/aligner/LoRA parameters; do not restore the old
   optimizer, scheduler, Trainer global step, RNG state, multiplier statistics, or old data position.
4. Start a fresh 8-card FSDP2 process group and fresh ACAVCAPS data stream. The source 4-card DCP shard count is not
   assumed to match the target world size; the loader streams source tensors through `torch.distributed.checkpoint`
   before the new FSDP wrapping.
5. Run the bounded smoke: real ACAVCAPS tar decode, forward, backward, save at a short step, then start a separate
   8-card process and resume from that newly-created smoke checkpoint. The second phase tests ordinary same-world-size
   checkpoint resume inside the new ACAVCAPS run; it is not a resume of the old 4-card multiplier run.

The ACAVCAPS warm-start learning-rate contract is intentionally different from the preceding multiplier run: audio
encoder `1e-5`, aligner `5e-5`, and Huginn LoRA `5e-5`. The multiplier source training remains on its existing
`1e-4` configuration; only the fresh ACAVCAPS optimizer uses this lower, per-group schedule.

The implementation and gates are:

- model-only DCP loader/audit in `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`;
- ACAVCAPS route in `code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_acavcaps_swift.py`;
- 8-card smoke runtime:
  `code/huginn_lora/scripts/smoke_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume.sh`;
- strict save/resume inspector:
  `code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic30s_acavcaps_warmstart_resume.py`.

The smoke is pending on remote execution and final source-checkpoint replacement. It must verify restored Whisper,
aligner, and LoRA tensors, exact source-to-target tensor copy verification, empty fresh optimizer state in phase 1,
optimizer ownership limited to current trainables, restored optimizer/scheduler state in phase 2, and frozen Huginn
backbone equality across the smoke save/resume boundary. Its optimizer audit must also report the per-group rates
`audio_encoder=1e-5`, `aligner=5e-5`, and `lora=5e-5`.

#### Current architecture and training contract

- Local waveform input is mono 16 kHz. Every source record is retained. Audio longer than `30s` is decoded/truncated to
  the first `30s`; shorter audio keeps its real effective duration. There is exactly one audio chunk per sample: no
  90-second splitting, no multi-chunk concatenation, and no duration discard threshold.
- Whisper-large receives up to `3000` log-mel frames with `80` channels. The feature tensor is collated to that model
  limit, but true feature lengths/masks are carried through the encoder and downstream compressor, so padding is not
  treated as valid audio. Its encoder is approximately `50` frames/s, hidden size `1280`, and produces about `1500` time
  steps for a full 30-second input. The complete encoder is one trainable FSDP unit with learning rate `1e-4`.
- The complete aligner is one FSDP unit. Its temporal compressor is exactly one trainable
  `Conv1d(1280,1280,kernel_size=12,stride=12,padding=0)`. With approximately `20ms` per Whisper encoder frame, this
  produces one content token per `240ms`. A full 30-second sample has `125` content tokens; shorter samples remain
  dynamic. Trainable `audio_bos` and `audio_eos` add two boundary positions, for a maximum valid prefix of `127`.
- The audio projector is `LayerNorm(1280)`, parallel `Linear(1280,2048)` branches with a SiLU gate, then
  `Linear(2048,5280)` and `LayerNorm(5280)`. The resulting prefix is concatenated before the text embeddings.
- Prefix padding is local-batch dynamic: each batch pads only to its longest valid prefix. Padding positions have
  attention-mask `0` and label `-100`; shorter samples are not padded to 125 content tokens or to 30 seconds. The
  flattened valid segments preserve the original audio/text pairs and are encoded by Whisper once and aligned once per
  model forward.
- AAC and ASR have distinct user prompts. AAC asks for an audible-event description; ASR asks for speech transcription.
  Clotho-v2 uses the training split and one deterministic caption per occurrence.
- Loss is response-only shifted causal next-token prediction: `shift_logits = logits[:, :-1]` against
  `shift_labels = labels[:, 1:]`. Audio prefix, prompt, prefix padding, and other non-response positions are `-100`.
  Audio BOS/EOS participate in the prefix forward path and are trainable/saved, but are not direct text targets.
- Swift `--max_length 192` is the text-side limit used by these routes; it is not a fixed combined audio-plus-text
  sequence length. The actual combined prefix-plus-text sequence remains bounded by Huginn's model context contract.
- Native Huginn backbone and LM head are frozen. Only Huginn transformer LoRA is trainable: rank `8`, alpha `16`,
  dropout `0.05`, learning rate `1e-4`. Whisper and the aligner do not receive LoRA. The only trainable groups are the
  Whisper encoder, aligner (compressor/projector/boundaries), and Huginn LoRA.
- FSDP4 uses five coarse units: complete Whisper encoder; complete aligner; Prelude two `SandwichBlock`s; recurrent
  adapter plus all four core `SandwichBlock`s; and Coda two `SandwichBlock`s. Only the recurrent-core unit uses
  `reshard_after_forward=false`; all other units remain resharded. Whisper internal per-layer checkpointing is disabled,
  while one outer FSDP activation-checkpoint wrapper covers the complete Whisper unit. Whisper attention uses PyTorch
  SDPA.
- Checkpoints are paired full-model FSDP2 DCP checkpoints, not adapter-only checkpoints. They include model shards,
  optimizer state, scheduler, per-rank RNG, Trainer state, and cumulative `audio_training_statistics.json`.
  Same-world-size four-card save/resume has passed, including exact data position, no-replacement continuity,
  effective-duration accounting, and trainable/frozen-state audits. Direct four-card to eight-card continuation has not
  been validated and is not the planned operation. The planned cross-world-size step is a **model-only DCP
  warm-start**: load only Whisper/aligner/LoRA model tensors into a fresh 8-card run, rebuild optimizer/scheduler,
  start global step and RNG fresh, and start ACAVCAPS data position fresh. The dedicated 8-card smoke must pass before
  any formal ACAVCAPS training.

#### Current remote/data rules

- This Windows checkout contains code and documentation only. Model weights, public data, manifests generated remotely,
  outputs, checkpoints, and large logs remain on Linux. Sync through GitHub and launch remote work only through
  `vc submit` wrappers.
- Current remote repository root is `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune`; current
  formal jobs use the remote `swift_huginn` environment. Whisper-large weights are remote at
  `/hpc_stor03/sjtu_home/jinwei.zhang/models/whisper-large` and are not part of this checkout.
- The active 4-card multiplier formal job uses `pdgpu-5090`, four RTX 5090 GPUs, and `-c32 -m128G -g4`, which is
  within the enforced maximum of `8` CPU cores and `32G` host memory per GPU. Do not replace this with an oversized
  request. The completed ACAVCAPS manifest-preparation and checkpoint-25000 audit jobs were deliberately submitted to
  `pdgpu-4090` because the 5090 queue was occupied; their wrapper filenames retain `_5090` for repository naming
  compatibility, but their actual resource pool, job names, and log names are `4090`.
- WavCaps and GigaSpeech public roots are read-only. Pool preparation is metadata/index/registry-only and does not
  download, copy, or bulk-decode the public audio. WavCaps FLAC and GigaSpeech Opus are decoded on demand at training
  time; GigaSpeech segment rows use their metadata start/end bounds and ffmpeg when needed.
- The passed metadata inventory recorded `91254` AudioCaps-v2 rows, `18364` Clotho source records grouped into `3839`
  audios, and `2264528` GigaSpeech-L segments representing about `2498.217` metadata hours. The full atomic-pool,
  indexed-mixture, multiplier-pool, and representative real-decode audits have passed; these audits do not imply that
  every public audio file was pre-decoded or copied locally.
- The intended `wavcaps_no_bbc_aac` pool excludes BBC Sound Effects. An earlier inventory exposed a BBC source label in
  the public WavCaps metadata; therefore the final registry/manifest is the authoritative exclusion check. Do not infer
  BBC removal from the pool name alone, and do not alter the read-only public WavCaps root.
- Effective training hours are measured after successful runtime decode/resample/truncation, using the actual retained
  waveform duration (`len(waveform)/16000`). Prefetch-only or duplicate template rows are not counted; raw metadata
  duration is not used for the cumulative training-hours statistic.
- Current model/plugin paths are `models/huginn-audio-whisper-dynamic90s-v1/`,
  `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`,
  `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py`, and
  `code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_multiplier_swift.py`.
- Current ACAVCAPS flat-manifest paths are under the private remote repo tree at
  `data/audio_swift/acavcaps/`; the public tar files remain under
  `/hpc_stor03/public/shared/data/raa/ACAVCAPS` and are opened read-only at training time.

#### Required validation state

The following have passed for the current contract: metadata-only pool preparation and audits, CPU no-replacement
sampler audit, real-audio decode chain, dynamic30s contract, acceleration Stage 0/1/2, four-GPU Stage 3-4 and Stage 5
history, and the updated four-GPU full-model checkpoint/resume smoke. The read-only audit of multiplier
`checkpoint-25000` also passed the current full-model DCP, model trainability, optimizer, scheduler, RNG, and runtime
contract checks. The ACAVCAPS flat global-tar manifest preflight passed for all `1071` tars and `4,664,169` pairs.
The current production switches are Whisper SDPA, no Whisper internal checkpointing, one outer Whisper checkpoint
wrapper, and recurrent-core `reshard_after_forward=false`.

Still pending: the current multiplier run's final `checkpoint-46050` and final formal completion audit, followed by the
8-card ACAVCAPS model-only warm-start/save/resume smoke. Do not reuse old dynamic90s checkpoint-4/6 artifacts, old
with-replacement sampler evidence, or any intermediate multiplier checkpoint as the intended final warm-start source.

#### X-ARES modality-alignment evaluation (unfinished; updated 2026-08-04)

X-ARES is a separate evaluation branch for measuring the representation quality of the Huginn audio encoder/aligner
output with the official X-ARES task/K-NN framework. It does not change the training route and must not be treated as a
training checkpoint or a completed benchmark.

- Official X-ARES code is remote-only; the actual uploaded third-party checkout is
  `/hpc_stor03/sjtu_home/jinwei.zhang/third_party/xares`. The active evaluation environment is the copied
  `env_xares` conda environment, separate from `swift_huginn`.
- The current reference checkpoint is the multiplier-line checkpoint
  `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_whisper_dynamic30s_multiplier_single_epoch_fsdp4/run-20260731_084946/swift_output/v0-20260731-085036/checkpoint-20000`.
- The wrapper restores the Whisper encoder and audio aligner from the paired full-model FSDP DCP checkpoint and returns
  the projected audio-prefix frame sequence for X-ARES. It deliberately does not use the Huginn recurrent core or
  Huginn LoRA as the X-ARES encoder representation, and audio BOS/EOS are excluded from the frame embedding sequence.
- Public evaluation data currently under consideration is read-only VoxCeleb1 at
  `/hpc_stor03/public/shared/data/mml/VoxCeleb1_origin`. Jobs use `pdgpu-4090`, `-c8 -m32G -g1`; no audio is copied
  into the repository. The X-ARES task is adapted through
  `code/huginn_lora/scripts/huginn_xares_voxceleb1_task.py`, and the wrapper is in
  `code/huginn_lora/scripts/huginn_whisper_xares_encoder.py` plus its entry module.
- Passed gates: `env_xares` import/package preflight, checkpoint read-only inspection, VoxCeleb1 path audit, synthetic/
  real encoder smoke, and the X-ARES API-contract inspection.
- Pending gates: rerun the VoxCeleb1 K-NN smoke with its writable work-root/cache fix, verify embeddings and task
  outputs, then run the complete VoxCeleb1 K-NN evaluation. Until those gates produce a final report and score, X-ARES
  must be labeled **unfinished**.
- A previous K-NN attempt failed first because an absolute script path was converted into an invalid relative import, and
  then because X-ARES tried to create its embedding cache under the read-only public dataset root. The current task
  adapter uses script basenames/PYTHONPATH and a writable isolated output work root with links to the public data; this
  fix has not yet been confirmed by a successful remote K-NN run.

### Historical LoSATok task record (updated 2026-07-27; superseded by the dynamic-30s Whisper mainline below)

This section records the previously active LoSATok work. It remains useful for reproducibility, but it is no longer the
current formal-training mainline. Keep its model construction, checkpoint format, checkpoint paths, and evaluation restore
paths strictly separate from the current Whisper route. Neither LoSATok route trains any official LoSATok parameter.

1. **Legacy fixed-32 LoSATok / single-5090 route — completed ACAVCAPS-quarter training; evaluation pending.** It uses
   first-30-second audio, compressor stride `4`, and `AdaptiveAvgPool1d(32)`, so the audio prefix is exactly
   `audio_bos + 32 + audio_eos` (34 positions). The completed one-epoch, quarter-ACAVCAPS continuation is:
   `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_acavcaps_wds_legacy_quarter_fixed32_warmstart2802_e1_b8ga4_5090/run-20260724_073239/v0-20260724-073259/checkpoint-36741`.
   It is a normal Swift adapter checkpoint: `adapter_model.safetensors` contains `66` LoRA tensors and
   `vit.safetensors` contains the `20` aligner tensors (including trainable `audio_bos/audio_eos`). This run was a
   **weight warm-start** from the completed legacy AudioCaps-v2 `checkpoint-2802`, not a Trainer resume.
2. **Dynamic-90s LoSATok / two-5090 FSDP2 route — historical/parallel training line.** It uses first-90-second audio,
   kernel `11` / stride `6` / padding `5`, no final adaptive pool, a cap of `375` compressed tokens, and therefore at
   most `377` audio-prefix positions after the trainable boundaries. The repaired two-rank FSDP2 save format is a
   sharded DCP containing exactly `66` LoRA plus `20` aligner tensors. Dynamic AudioCaps-v2 two-epoch training is
   complete and its audited usable checkpoints are:
   `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115/checkpoint-2802`
   and `.../checkpoint-5604`. The formal quarter-ACAVCAPS job warm-starts weights from `checkpoint-5604`; a supplied
   log proves it reached step `50/36741`, but no completion log is recorded here. Do not claim that run completed until
   its final remote log and checkpoint audit are supplied. The observed early-run metrics were `13.127781 s/it`,
   `23.37 GiB` GPU memory, and loss `2.13275414` at step 50; these are diagnostic observations, not a final result.
3. **Whisper-large FSDP full finetuning** is a separate historical/independent line: frozen Whisper-large,
   full-trainable aligner, full-trainable Huginn under Swift FSDP2. Its historical 8-GPU `checkpoint-2802` is an
   evaluation artifact, not a cross-world-size resume source.

#### Exact Huginn handoff: routes, data, and non-negotiable constraints

- This Windows checkout is code/documentation only. Model assets, public datasets, checkpoints, outputs, and large logs
  stay on remote Linux. Sync edits via GitHub; submit remote work only through existing `run_*.sh` wrappers, which use
  `vc submit`.
- The public ACAVCAPS root `/hpc_stor03/public/shared/data/raa/ACAVCAPS` is read-only. Never create, alter, unpack, or
  write manifests there. Private manifests, stats, and progress files live under
  `data/audio_swift/acavcaps_wds/` in the remote repository.
- The quarter ACAVCAPS manifest is
  `acavcaps_wds_stage_schedule_quarter_ceil_seed20260723.json`: `271` tars selected by `ceil(N/4)` per category
  (`00A=4`, `0M0=40`, `S00=120`, `S0A=25`, `SM0=74`, `0MA=2`, `SMA=6`). It preserves the private globally shuffled tar
  order within the three stages `00A+0M0+S00 -> S0A+SM0+0MA -> SMA`; WebDataset applies a per-tar streaming buffer
  shuffle of `512`. FLAC is read and decoded only at training time.
- Two-rank ACAVCAPS uses Accelerate `DataLoaderDispatcher`: rank 0 consumes/decodes the streaming source for each batch
  and dispatches prepared batches to rank 1. Do **not** add manual rank sharding inside the dataset; the distributed
  inspection passed with equal probes and no cross-rank overlap.
- Both LoSATok routes freeze the complete official LoSATok stack (MiDaSheng, semantic branch, acoustic branch, and all
  official LoSATok modules), train the new aligner (`temporal_compressor`, `audio_projector`,
  `audio_boundary_embeddings`), and train only Huginn LoRA rather than the Huginn base. Boundary embeddings are part of
  the aligner and are trainable; do not treat them as fixed delimiter token embeddings.
- Loss is shifted causal NTP. Audio-prefix positions and dynamic padding are labelled `-100`; only intended text target
  tokens supervise the loss. `--max_length 192` is the text-side maximum, not a fixed combined audio-plus-text limit.
- Permanently exclude historical dynamic DCPs under
  `outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928/checkpoint-{2802,5604}`: each has
  `66` LoRA and `0` aligner tensors, so it cannot be evaluated, resumed, or used as a warm-start.

The historical shared audio architecture was:

- frozen audio encoder: Whisper-large on the historical Whisper route, or full LoSATok on the LoSATok route
- trainable aligner: temporal compressor, audio projector, and audio boundary embeddings
- Huginn text backbone
- audio prefix concatenated before text embeddings: historical fixed-32 used `audio_bos + 32 compressed tokens + audio_eos`,
  while the former dynamic-90s route used up to `audio_bos + 375 compressed tokens + audio_eos`.
The current dynamic-30s Whisper architecture is documented in the authoritative section above.

There are two distinct Swift fine-tuning policies; do not confuse them:

- historical/currently usable LoRA route:
  - audio encoder frozen
  - aligner full-trainable
  - Huginn base frozen, Huginn LoRA trainable
- historical FSDP full-training route:
  - audio encoder frozen
  - aligner full-trainable
  - Huginn backbone full-trainable

In the historical policies below, Whisper was not LoRA-wrapped and was often frozen. This does **not** describe the
current dynamic-30s Whisper mainline: its complete Whisper encoder is full-trainable, but still receives no LoRA.

The equivalent rule for the new LoSATok LoRA branch is stricter: the complete official LoSATok stack, including its semantic and acoustic components, is always frozen. Only the new temporal compressor/projector/boundary embeddings and Huginn LoRA tensors may train.

**Current-status precedence rule:** the authoritative dynamic-30s Whisper section near the top and the dated dynamic-30s
section below are the source of truth for active Huginn Whisper work. LoSATok/ACAVCAPS sections are parallel or historical
unless explicitly selected by the user. Older sections retained later in this README document reusable infrastructure;
they must not override a newer dated status entry.

### Current execution status

#### Historical/parallel LoSATok Swift LoRA replacement branch: completed training and evaluation pending

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
- Current prepared legacy LoSATok evaluation target:
  - checkpoint: completed fixed-32 quarter-ACAVCAPS `checkpoint-36741` (full path in the current handoff above);
  - MMAU submit script: `code/huginn_lora/run_eval_mmau_test_mini_losatok_legacy_acavcaps_quarter_5090.sh`;
  - Clotho sample-generation submit script:
    `code/huginn_lora/run_generate_clotho_caption_samples_losatok_legacy_acavcaps_quarter_5090.sh`;
  - no MMAU score or sample result has been supplied yet; do not claim either outcome.
- LoSATok evaluation restore rules:
  - caption generation and MMAU restore both LoRA (`66` tensors) and aligner (`20` tensors);
  - retrieval restores the aligner only because its definition pools encoder/projector tokens and raw Huginn input embeddings without running LoRA-modified recurrent blocks.

#### LoSATok dynamic-compressor FSDP2 experiment: architecture, save repair, and current continuation (updated 2026-07-27)

- This is a new experimental branch layered on the same LoSATok model code. It is enabled only by
  `HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1`, so the completed kernel-7/stride-4/fixed-32 checkpoints retain their
  legacy construction when the variable is unset.
- New audio context policy:
  - retain at most the first `90` seconds at 16 kHz;
  - remove the final `AdaptiveAvgPool1d(32)`;
  - temporal compressor kernel `11`, stride `6`, symmetric padding `5`;
  - cap compressed audio at `375` tokens;
  - retain trainable `audio_bos` and `audio_eos`, so the maximum audio prefix is `377` tokens;
  - keep Swift `--max_length 192` for text, giving a maximum combined prefill of `569`, below Huginn's `4096` block size.
- Variable-length batches are padded only to the longest audio prefix in the current batch. Padded audio positions use
  attention mask `0` and labels `-100`; they are not supervised. Valid dynamic audio positions use a non-pad placeholder
  token ID so Huginn's compiled attention mask does not accidentally hide the supplied audio embeddings.
- Final trainability policy is aligner plus Huginn LoRA only:
  - full LoSATok encoder trainables: `0`;
  - dynamic aligner trainables, including learned `audio_bos/audio_eos`: `62,953,248`;
  - Huginn LoRA trainables: `12,541,440`;
  - Huginn base trainables: `0`;
  - total trainables: `75,494,688`.
- The plugin performs a first-forward post-PEFT/FSDP topology audit and aborts if LoSATok, original Huginn weights, or any
  unclassified parameter is trainable, or if the aligner/LoRA groups and learned audio boundary embeddings are not trainable.
- The loss path is standard causal next-token prediction: it obtains full prefix-plus-text logits with backbone labels disabled,
  prepends `-100` labels for the complete padded audio-prefix width, then computes cross entropy from `logits[:, :-1]` to
  `labels[:, 1:]`. Thus audio embeddings and per-batch audio padding are never direct loss targets. The first-forward audit also
  requires every sample to contain a supervised text target and requires the first text position to remain prompt-masked, so a
  shorter sample's right-padded audio-prefix tail cannot become the position directly responsible for a supervised target.
- Validated compute/data behavior:
  - the two-GPU AudioCaps-v2 FSDP2 no-save smoke passed with `B=4`, `GA=4`, global effective batch `32`;
  - the two-GPU dynamic ACAVCAPS WebDataset smoke also passed (`2` updates, `B=4`, `GA=4`), so the dynamic model can forward,
    backward, and receive batches from the distributed streaming input path;
  - retain the previously established FSDP2 execution settings: `full_shard`, `SHARDED_STATE_DICT`, FSDP activation checkpointing
    disabled, and Trainer/model gradient checkpointing disabled.
- **Checkpoint defect discovered after the smoke:** the former dynamic AudioCaps-v2 two-GPU run
  `outputs/huginn_losatok_dynamic90s_audiocaps_v2_e3_b4ga4_fsdp2/v0-20260723-054928/checkpoint-{2802,5604}` was audited from
  DCP metadata. Each checkpoint contains exactly `66` LoRA tensors and **no** aligner tensor (`expected=20`, `actual=0`).
  There are no aligner weight sidecars. These are incomplete checkpoints: do not evaluate, resume, warm-start, or treat them as
  evidence that dynamic checkpoint saving works.
- The first repair hypothesis, passing
  `--modules_to_save temporal_compressor audio_projector audio_boundary_embeddings`, was tested in a dedicated two-GPU save smoke.
  It also produced `checkpoint-2` with `66` LoRA and `0` aligner tensors. The smoke correctly stopped before its fresh-process
  resume phase because `--require_complete` rejected that incomplete checkpoint. Therefore CLI argument acceptance alone is not a fix.
- The targeted save trace then established the immediate root cause boundary:
  - the final model retains the original trainable `temporal_compressor` (`46,863,872` parameters), `audio_projector`
    (`16,078,816`), and `audio_boundary_embeddings` (`10,560`), but has no PEFT `ModulesToSaveWrapper` and exposes no
    `peft_config.modules_to_save` from this inner model;
  - immediately before DCP write, Accelerate calls `_get_model_state_dict(..., adapter_only=True)` and receives exactly
    `66` LoRA keys, `0` aligner keys, and `0` other keys;
  - the captured call stack and source locate that flag in the normal Transformers checkpoint flow:
    `transformers/trainer.py:_get_fsdp_ckpt_kwargs()` returns `{"adapter_only": True}` whenever the installed Accelerate
    `save_fsdp_model` exposes that argument; `Trainer._save_optimizer_and_scheduler` passes it to the FSDP writer. This is not
    caused by `save_only_model` or by the DCP writer;
  - Accelerate then deliberately calls PEFT `get_peft_model_state_dict` for a `FSDPPeftModelForCausalLM`; its paired load path
    calls `set_peft_model_state_dict`. This is the desired checkpoint format **if** the aligner is represented by PEFT
    `ModulesToSaveWrapper` entries;
  - Swift's `lora_layers.py` explicitly supports that wrapper and checks every model-module suffix before LoRA target matching.
    This historical trace is superseded by the current dynamic-only PEFT-constructor repair described below. The existing
    adapter-only FSDP save/load pairing can include LoRA plus aligner without any custom DCP format or sidecar. Do **not**
    simply force `adapter_only=False`, because that could serialize the frozen multi-billion-parameter Huginn backbone.
- Current targeted diagnostic:
  - `code/huginn_lora/scripts/smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.sh`;
  - submit only through `code/huginn_lora/run_smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.sh`;
  - it enables the opt-in `HUGINN_LOSATOK_FSDP_SAVE_DEBUG=1` trace and the dynamic-only
    `HUGINN_LOSATOK_PEFT_ALIGNER_MODULES_TO_SAVE=1` repair. The plugin restores the three aligner names onto the PEFT
    `LoraConfig` immediately before `PeftModel.__init__`, requires all three `ModulesToSaveWrapper` instances to appear, and
    then records the exact LoRA/aligner/other key counts returned by Accelerate immediately before DCP write.
  - Current evidence from the `2026-07-24` smoke: phase 1 passed its strict DCP audit (otherwise `set -e` would have stopped
    before phase 2), proving the newly saved checkpoint contains `66` LoRA and `20` aligner tensors. In phase 2, the plugin
    rebuilt the exact PEFT topology from DCP metadata: `33` LoRA targets, rank `16`, alpha `32`, dropout `0.05`, plus all
    three `ModulesToSaveWrapper` aligners. Each wrapper intentionally owns an original frozen branch and one trainable
    `modules_to_save.default` branch; its total parameter count is therefore doubled while its trainable count remains the
    expected aligner total `62,953,248`.
  - The remaining fresh-resume failure is now isolated to `swift/tuner_plugin/lora_llm.py`: after PEFT reconstruction it
    unconditionally loads the legacy fixed-32 sidecar `vit.safetensors`, which does not exist and must not be fabricated for
    an adapter-only DCP. The plugin now contains a dynamic-DCP-only interception that bypasses this legacy sidecar read and
    lets Accelerate restore the `20` aligner tensors from DCP alongside LoRA, optimizer, scheduler, and RNG. The first launch
    of that interception stopped during external-plugin import because remote Swift 4.1.3 declares
    `LoRALLMTuner.from_pretrained` as a `staticmethod`, while the initial defensive patch accepted only an instance method.
    No model, dataset, CUDA tensor, process group, or checkpoint was touched in that failed run. The plugin now preserves all
    three Python descriptor forms and logs the selected form; the expected remote value is
    `tuner_class=LoRALLMTuner descriptor=staticmethod`.
  - Final remote result: the dedicated smoke completed end to end. It saved
    `.../run-20260724_112507/save_phase/v0-20260724-112551/checkpoint-2`, restored it in a fresh two-rank process, continued
    from global step `2` to `3`, saved
    `.../run-20260724_112507/resume_phase/v0-20260724-113128/checkpoint-3`, and printed
    `LOSATOK DYNAMIC FSDP2 MODULES-TO-SAVE SAVE/RESUME SMOKE PASSED`. Both strict DCP audits are therefore complete.
- Formal dynamic AudioCaps-v2 training completed after the save/resume repair:
  - runtime: `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2.sh`;
  - submit: `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2_5090.sh`;
  - configuration: two GPUs, `B=4`, `GA=4`, global effective batch `32`, two epochs, `save_only_model=false`, and a required
    `66 + 20` DCP audit for every resulting checkpoint;
  - completed run root:
    `/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/outputs/huginn_losatok_dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115`;
  - audited usable epoch checkpoints: `checkpoint-2802` and `checkpoint-5604`. These are DCP checkpoints, not legacy
    `adapter_model.safetensors` / `vit.safetensors` directories.
- Dynamic ACAVCAPS-quarter continuation uses
  `scripts/train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh` and its matching submit wrapper. It is a
  **DCP adapter-weight warm-start only**: it restores the `66 + 20` tensors from the selected AudioCaps checkpoint before
  FSDP preparation while optimizer, scheduler, RNG, global step, and dataset position begin fresh. Its current default source
  is the completed dynamic `checkpoint-5604`, not `checkpoint-2802`. The quarter warm-start/save/fresh-reload smoke passed
  before the formal launch.
- Do not load older fixed-32 aligner checkpoints while the dynamic environment variable is enabled unless a deliberate
  architecture-conversion procedure is implemented and separately validated.

#### ACAVCAPS WebDataset preparation and continuation routes (updated 2026-07-27)

This subsection is historical LoSATok infrastructure, including the quarter-curriculum route. It is retained for
reproducibility but is not the current Whisper ACAVCAPS training plan. The current Whisper route uses the flat global
manifest and one permutation across all `1071` tars described in the authoritative mainline above.

- The full read-only ACAVCAPS preflight completed successfully:
  - `1071` tar shards;
  - `4,664,169` JSON/FLAC sample pairs;
  - stage schedule `00A/0M0/S00 -> S0A/SM0/0MA -> SMA`;
  - all tar pair, JSON, and non-empty `long` caption checks passed;
  - private full manifest, stats, and resumable progress checkpoint were written.
- No offline audio decoding is part of the formal preparation. The FLAC bytes stay inside the public tar shards and are decoded by the Swift LoSATok template during training.
- Full-manifest validation and loader tests are complete:
  - `run_inspect_acavcaps_wds_dynamic_training_config_5090.sh` passed manifest/stat consistency and reported two-GPU
    effective batch `32`, `145756` updates per full epoch;
  - `run_inspect_acavcaps_wds_distributed_sharding_5090.sh` passed with equal probe lengths `[12474, 12474]`, no cross-rank
    sample overlap, and `DataLoaderDispatcher` process-0 batch dispatch. Rank 0 reads/decodes the stream for a batch and
    Accelerate dispatches batches; the dataset must therefore **not** add a second manual rank sharding layer;
  - `run_smoke_acavcaps_wds_huginn_losatok_dynamic90s_swift_lora_fsdp2_5090.sh` passed the short dynamic two-GPU FSDP2
    forward/backward path without checkpoint saving.
- The first two-GPU probe exposed and fixed a WebDataset guard: its default `single_node_only` rejects any `torch.distributed` world, even on one physical node. The ACAVCAPS loader now passes an identity `nodesplitter` because each WebDataset instance contains one tar; Accelerate remains responsible for rank-level batch/sample partitioning.
- A private **quarter** manifest was then derived from the fully validated full manifest without opening, copying, or changing any
  public tar. It uses `ceil(category_tar_count / 4)`: `00A=4`, `0M0=40`, `S00=120`, `S0A=25`, `SM0=74`, `0MA=2`, `SMA=6`, for
  `271` tars total. It preserves the full manifest's shuffled global tar order inside each three-stage curriculum and retains
  the WebDataset runtime buffer shuffle (`512`) inside every tar.
  - preparation: `prepare_acavcaps_wds_quarter_manifest.py` and `run_prepare_acavcaps_wds_quarter_manifest_5090.sh`;
  - strict manifest inspection and legacy streaming smoke passed;
  - the public root `/hpc_stor03/public/shared/data/raa/ACAVCAPS` is read-only: manifests/stats/progress live only under the
    user's private repo tree.
- Legacy continuation is deliberately separate from dynamic continuation:
  - legacy source: fixed-32 AudioCaps-v2 `checkpoint-2802` above, ordinary adapter plus `vit.safetensors` aligner checkpoint;
  - legacy ACAVCAPS quarter warm-start save/reload smoke passed, including saving a new checkpoint and restoring it in a fresh
    process;
  - formal legacy quarter script:
    `train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh` with
    `run_train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh`; single 5090, `B=8`, `GA=4`, effective batch `32`,
    one epoch. This completed at `checkpoint-36741` under
    `outputs/huginn_losatok_acavcaps_wds_legacy_quarter_fixed32_warmstart2802_e1_b8ga4_5090/run-20260724_073239/v0-20260724-073259/`.
    The corresponding legacy MMAU-mini and Clotho qualitative-generation submit wrappers are
    `run_eval_mmau_test_mini_losatok_legacy_acavcaps_quarter_5090.sh` and
    `run_generate_clotho_caption_samples_losatok_legacy_acavcaps_quarter_5090.sh`; no evaluation result is recorded yet.
- Dynamic continuation uses the same quarter manifest after the completed dynamic AudioCaps-v2 DCP audit. Its formal training
  script warm-starts from dynamic `checkpoint-5604` as weights only, never as a cross-dataset Trainer resume. The supplied
  formal-run log proves training reached `50/36741`; wait for a final log plus per-checkpoint `66 + 20` audit before recording
  an ACAVCAPS completion.

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

#### Superseded Whisper-large dynamic-90s route (historical architecture record)

The former dynamic-90s route was isolated from the historical fixed-32 Whisper route. Historical files remain at
`models/huginn-audio-whisper-v1/` and `code/huginn_lora/plugins/huginn_audio_swift.py`; do not point historical
checkpoints or evaluation scripts at the dynamic package. The current runtime has since been simplified to one
dynamic-30s chunk and 240ms/token; use the authoritative section above and the dated dynamic-30s section below.

- dynamic model package: `models/huginn-audio-whisper-dynamic90s-v1/`
- dynamic Swift plugin: `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`
- model type/template/model arch: `huginn_audio_whisper_dynamic90s`
- Whisper-large and the aligner are fully trainable at `1e-4`; Huginn uses `lora_llm` with rank `8`, alpha `16`, and effective
  dropout `0.05`. The installed ms-swift `LoRALLMTuner` does not forward the generic dropout argument into PEFT, so the
  isolated dynamic plugin patches `peft.LoraConfig` before LoRA-layer creation and the Stage 0-2 gate audits both the
  saved PEFT config value and every instantiated LoRA dropout module.
- LoRA is restricted to the Huginn transformer only. Neither the complete Whisper encoder unit nor the complete audio
  aligner unit may contain a LoRA tensor. Huginn's native recurrent adapter is the `Linear(2*n_embd -> n_embd)` that
  combines the current recurrent latent with the fixed input embeddings; it is now owned by the recurrent-core FSDP
  unit and remains one of the 33 Huginn LoRA targets.
- FSDP transformer-based auto-wrap now targets exactly five callable coarse units: complete Whisper encoder; complete
  aligner; both prelude SandwichBlocks together; recurrent adapter plus all four reused core SandwichBlocks together;
  and both coda SandwichBlocks together. Every unit uses `reshard_after_forward=true`. The 1/2/3 Whisper segments in a
  local batch are flattened so Whisper is called once and the aligner is called once per model forward.
- the compressor is exactly one Conv1d with kernel `6`, stride `6`, and padding `0`.
- audio token count is dynamic: each complete `120 ms` produces one token. Only a complete 30-second segment produces
  `250` tokens; shorter audio is never padded to 250 tokens. Complete 60/90-second inputs produce 500/750 tokens.
- audio is split into non-overlapping windows of at most 30 seconds; at most the first 90 seconds are included. Every
  input longer than 90 seconds is retained and truncated to exactly 90 seconds, including inputs longer than 120
  seconds; this route has no duration-based discard threshold.
- prefix embeddings are padded to the longest prefix in each collated batch; padding uses zero embeddings, attention
  mask `0`, and labels `-100`.
- The plugin preserves Whisper log-mel features in FP32 instead of intentionally quantizing them to the LLM BF16 dtype.
  The dynamic model also defensively casts each segment to the encoder parameter dtype/device immediately before the
  Whisper call, then casts encoder outputs to the trainable aligner dtype. Stage 0-2 audits the dtype seen by every real
  Whisper encoder invocation.

The data-independent Stage 0-2 gate generates deterministic WAV fixtures inside its remote output directory and does
not use AudioCaps, ACAVCAPS, WavCaps, or any future formal dataset. It checks the production duration planner, real Swift
template/collator, real Whisper-large/Huginn loading, effective LoRA configuration, trainable split, dynamic prefix
lengths, padding masks/labels, and one real backward pass on one RTX 5090.

- runtime: `code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage02.sh`
- submit wrapper: `code/huginn_lora/run_inspect_huginn_audio_whisper_dynamic90s_stage02_5090.sh`

After Git sync, submit with:

```bash
bash code/huginn_lora/run_inspect_huginn_audio_whisper_dynamic90s_stage02_5090.sh
```

The pre-grouping architecture passed Stage 0-2 remotely with the terminal banner
`HUGINN WHISPER DYNAMIC90S STAGE 0-2 VALIDATION PASSED`. The merged Stage 3 (four-rank FSDP2 construction and DTensor
sharding) plus Stage 4 (one real optimizer update) gate uses only generated synthetic WAV files.
It uses Swift CLI's internal torchrun path, four RTX 5090 GPUs, the previously verified custom FSDP2 full-shard config,
per-device batch size 1, gradient accumulation 1, and `max_steps=1`; it deliberately saves no checkpoint.

The first distributed step covers four different prefixes across the four ranks: 1 second / 10 prefix tokens, 30
seconds / 252, 60 seconds / 502, and 120.01 seconds truncated to 90 seconds / 752. Every rank must write both an FSDP
marker and an optimizer-step marker. Post-run validation requires world size 4, CUDA devices 0-3, FSDP2 DTensors, the
the exact `66 LoRA + 14 aligner + complete Whisper encoder` trainable split, frozen Huginn base, and `global_step=1` on all ranks.
It additionally requires every trainable tensor to be a DTensor and verifies that all parameters in each of the five
coarse units are DTensors. The earlier per-SandwichBlock Stage 3-4 attempt failed correctly at `64/80` DTensor
trainables: the 64 block LoRA tensors were sharded, while the recurrent-adapter LoRA pair and 14 aligner tensors were
outside FSDP. That topology has been replaced rather than weakening the audit.

- synthetic fixture preparation:
  `code/huginn_lora/scripts/prepare_huginn_audio_whisper_dynamic90s_stage34.py`
- runtime: `code/huginn_lora/scripts/smoke_huginn_audio_whisper_dynamic90s_stage34_fsdp4.sh`
- submit wrapper: `code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_stage34_fsdp4_5090.sh`

After Git sync, submit with:

```bash
bash code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_stage34_fsdp4_5090.sh
```

The revised coarse-unit Stage 3-4 gate passed remotely on four RTX 5090 GPUs. Every rank reported `640` DTensor
parameters, all five expected FSDP units, the correct dynamic first-step prefixes (`10/252/502/752` across ranks), an
`AcceleratedOptimizer` update at `global_step=1`, and `exit_status=0`.

Stage 5 is the synthetic four-GPU multi-step stability gate. It deliberately remains data-independent and performs 20
real optimizer updates with per-device batch size 1 and gradient accumulation 1. It keeps the exact Stage 3-4 model,
LoRA, learning-rate, FSDP, and dynamic-audio contracts; checks raw training loss plus logged loss/gradient norms for
non-finite values; requires one finite loss log per update on every rank; and revalidates the five FSDP units and all
`80` trainable DTensors. It uses `save_strategy=no`; checkpoint save/reload was a later gate at the time of this
historical run and has since been implemented and passed separately. This paragraph is retained as historical Stage 5
evidence, not as the current checkpoint status.

- runtime: `code/huginn_lora/scripts/smoke_huginn_audio_whisper_dynamic90s_stage5_stability_fsdp4.sh`
- marker inspector: `code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage5_markers.py`
- submit wrapper: `code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_stage5_stability_fsdp4_5090.sh`

Submit Stage 5 only through:

```bash
bash code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_stage5_stability_fsdp4_5090.sh
```

Stage 5 passed remotely on all four ranks. Every rank completed `20` finite losses and `20` finite gradient norms,
retained `640` DTensor parameters, used `AcceleratedOptimizer`, and reached `global_step=20`; the job ended with
`HUGINN WHISPER DYNAMIC90S STAGE 5 STABILITY PASSED` and `exit_status=0`.

At the time, the next active work was formal data preparation before the checkpoint gate. That data preparation, sampler
audit, real-data chain, and four-GPU save/resume gate have since passed; the following policy is retained as historical
Stage 5 context. The fixed eligible pools and hierarchical sample-draw policy were:

- AAC `60%`, composed of WavCaps without BBC Sound Effects `60%`, AudioCaps-v2 `30%`, and Clotho-v2 train only `10%`;
- ASR `40%`, composed of GigaSpeech segment-level `{L}` records;
- therefore global draw probabilities are WavCaps `36%`, AudioCaps-v2 `18%`, Clotho-v2 `6%`, and GigaSpeech-L `40%`;
- realized dynamic audio-token totals are accumulated and reported during training, rather than precomputed or used as
  the initial sampler unit;
- Clotho references remain grouped by audio, but each scheduled training occurrence emits exactly one caption;
- all datasets share one atomic manifest schema and are normalized at the model input boundary to mono 16-kHz float32.
  Source WAV/FLAC/Opus files remain in place; public WavCaps and GigaSpeech roots are read-only.

The first read-only data gate has passed remotely:

- contract: `code/huginn_lora/configs/huginn_whisper_dynamic90s_data_contract_v1.json`;
- inspector: `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_data_pools.py`;
- runtime: `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_data_pools.sh`;
- submit wrapper: `code/huginn_lora/run_inspect_huginn_whisper_dynamic90s_data_pools_5090.sh`.

It performs no model load, training-manifest generation, schedule generation, token accumulation, audio conversion,
audio decoding, download, copy, or public-root write. The first heavy implementation was simplified after its audio-side
I/O proved unnecessary. The current gate reads AudioCaps/Clotho/WavCaps/GigaSpeech metadata, streams the large
GigaSpeech top-level `audios` array to identify segment-level `{L}` records, verifies source-level BBC exclusion, groups
Clotho train captions by audio, and checks only a small deterministic sample of audio locations. It never scans or opens
every audio file. The remote-only JSON report is written under
`data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/audits/`.

The passed inventory reported `91,254` valid AudioCaps metadata rows, all four expected WavCaps sources,
`18,364` Clotho train caption rows grouped into `3,839` audio samples, and `2,264,528` segment-level GigaSpeech-L
records totaling `2,498.217` metadata hours. It ended with no blocking issues. Per-record dynamic token counts are not
stored by this gate or required in the atomic manifest; realized token totals will be accumulated during training.

Submit it only through:

```bash
bash code/huginn_lora/run_inspect_huginn_whisper_dynamic90s_data_pools_5090.sh
```

The atomic mapping pilot has passed remotely. It creates only `16` metadata-only pilot records per
pool, validates real source-field/audio-path mappings, excludes BBC by source, preserves grouped Clotho references with
one-caption-per-training-occurrence policy, cleans GigaSpeech `text_tn` placeholders, and proves that all four pools use
one atomic schema. It does not decode/copy audio or calculate tokens.

- implementation: `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_atomic_pilot.py`;
- runtime: `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_atomic_pilot.sh`;
- submit wrapper: `code/huginn_lora/run_prepare_huginn_whisper_dynamic90s_atomic_pilot_5090.sh`.

Submit it only through:

```bash
bash code/huginn_lora/run_prepare_huginn_whisper_dynamic90s_atomic_pilot_5090.sh
```

The complete atomic-pool generation gate has passed remotely. It streams all four
metadata pools into `data/audio_swift/huginn_whisper_dynamic90s_multitask/v1/pools/*.jsonl`, with one little-endian
uint64 byte-offset index per manifest, per-pool stats and SHA-256 values, `pool_registry.json`, and
`full_pool_report.json`. All pool files remain temporary until every pool has completed and passed count checks. The
GigaSpeech-L emitted count must exactly equal the passed inventory count `2,264,528`, and Clotho must exactly equal
`3,839` grouped train audios. It still performs no audio decode/copy/full-path scan or token calculation.

- implementation: `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_full_atomic_pools.py`;
- runtime: `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic90s_full_atomic_pools.sh`;
- submit wrapper: `code/huginn_lora/run_prepare_huginn_whisper_dynamic90s_full_atomic_pools_5090.sh`.

Submit it only through:

```bash
bash code/huginn_lora/run_prepare_huginn_whisper_dynamic90s_full_atomic_pools_5090.sh
```

The indexed random-access gate now validates the in-place no-replacement v2 sampler. It validates random reads from
every JSONL/uint64-index pair, simulates `1,000,000` global pool selections, audits both hierarchy levels
(`AAC/ASR=60/40`, then AAC `60/30/10`), and exhaustively checks two complete independently shuffled epochs of every
pool. Every atomic record must appear exactly once per pool epoch, epoch 1 must reorder epoch 0, the same seed must be
reproducible, and uninterrupted versus arbitrary-position resumed streams must be identical. It also writes a
`4,096`-entry metadata-only pilot schedule. It reads no audio and performs no token calculation.

- reusable indexed mixture module: `code/huginn_lora/data_pipeline/indexed_atomic_mixture.py`;
- inspector: `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_indexed_mixture.py`;
- runtime: `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_indexed_mixture.sh`;
- submit wrapper: `code/huginn_lora/run_inspect_huginn_whisper_dynamic90s_indexed_mixture_5090.sh`.

Submit it only through:

```bash
bash code/huginn_lora/run_inspect_huginn_whisper_dynamic90s_indexed_mixture_5090.sh
```

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
   - uses the reusable **ms-swift training path**
   - current dynamic-30s requirement is:
     - original Huginn backbone
     - Whisper-large encoder
     - Whisper encoder and aligner fully trainable at `1e-4`
     - Huginn native backbone frozen; Huginn-only rank-8 LoRA trainable
     - one 30-second-capped chunk per sample and dynamic 240ms/token prefix

When the user says "current audio task", prefer to interpret it as the **Swift multimodal LoRA branch**, unless they explicitly refer to the older standalone training scripts.

### V1 architecture and historical variants

The following older V1 descriptions are retained to explain checkpoint/script separation. The current Whisper contract is
the dynamic-30s architecture in the authoritative section near the top.

- audio encoder:
  - historical standalone branch:
    - **Whisper-small**
  - current Swift mainline target:
    - **Whisper-large**
- temporal compressor:
  - historical fixed-32 versions used Conv-GMLP/shortcut variants and adaptive pooling;
  - current Whisper dynamic-30s version uses exactly one `Conv1d(1280,1280,kernel=12,stride=12,padding=0)` and no
    adaptive pool, producing dynamic token counts at `240ms/token`.
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
- the current Whisper dynamic-30s Swift policy is:
  - full-train `audio_encoder` at `1e-4`
  - full-train `aligner` at `1e-4`
  - freeze native Huginn backbone/LM head
  - LoRA-train Huginn transformer only at rank `8`, alpha `16`, dropout `0.05`, learning rate `1e-4`

### Whisper architecture details that matter

For the **historical fixed-32** Whisper-specific `models/huginn-audio-whisper-v1` implementation:

- Whisper output:
  - `last_hidden_state: [B, T_audio, hidden_audio]`
- compressor:
  - historical Conv-GMLP style temporal compression
  - historical kernel/stride/adaptive-pool settings belong only to that fixed-32 package
- current Whisper dynamic-30s compressor:
  - located under `models/huginn-audio-whisper-dynamic90s-v1/`
  - one Conv1d, kernel/stride `12`, padding `0`, no adaptive pooling
  - 125 content tokens at 30 seconds and dynamic shorter prefixes
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
- `models/huginn-audio-whisper-dynamic90s-v1/raven_modeling_minimal.py`
- `models/huginn-audio-whisper-dynamic90s-v1/raven_config_minimal.py`

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

ACAVCAPS was a validated Swift audio training route after the early Clotho-only smoke work. Those JSONL/chunk routes remain
historical infrastructure. A **separate historical/parallel LoSATok continuation route** uses the private WebDataset manifests
documented in the dated section near the beginning of this README; do not substitute either representation for the other.

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

The policy depends on the isolated experiment line; do not generalize a historical route to the current Whisper
mainline:

- **Current Whisper dynamic-30s formal route:** `audio_encoder`/Whisper fully trainable at `1e-4`; aligner fully
  trainable at `1e-4`; native Huginn backbone and LM head frozen; Huginn-only LoRA rank `8`, alpha `16`, dropout
  `0.05`, and learning rate `1e-4`. No LoRA is attached to Whisper or the aligner.
- **Historical LoSATok LoRA route:** complete official LoSATok stack frozen; new aligner and Huginn LoRA trainable;
  Huginn base frozen.
- **Historical Swift full-parameter route:** aligner and Huginn base trainable, no Huginn LoRA; its old frozen-Whisper
  setting must not be confused with current Whisper dynamic-30s training.

In every route, `audio_bos` and `audio_eos` belong to the aligner when present and must be included in the correct
trainable/checkpoint contract.

### Historical Swift audio status and reusable validation facts (updated 2026-07-27)

Historical validation facts are retained here; the dated top-level Huginn handoff is authoritative for live status:

- Swift registration, tar/WAV decoding, audio-prefix insertion, shifted NTP loss, and audio-encoder freezing have all been remote-verified.
- single-GPU Whisper LoRA routes on ACAVCAPS/AudioCaps are historical validated baselines.
- the legacy fixed-32 LoSATok single-GPU LoRA route completed three AudioCaps-v2 epochs, one ClothoAQA continuation epoch,
  and the one-epoch quarter-ACAVCAPS continuation ending at `checkpoint-36741`; it uses the normal adapter plus
  `vit.safetensors` checkpoint layout.
- the distinct dynamic-90s LoSATok FSDP2 route now passes complete `66 LoRA + 20 aligner` DCP save, fresh-process two-rank
  Trainer resume, continued optimization, and re-save. Its completed two-epoch AudioCaps-v2 run produced audited
  `checkpoint-2802` and `checkpoint-5604` under `...dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115/`.
  The formal dynamic quarter-ACAVCAPS continuation is weight-warm-started from `checkpoint-5604`; only its observed
  `50/36741` progress is known here. The older `20260723-054928` checkpoints remain incomplete and forbidden.
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
- The **legacy fixed-32** LoSATok adapter/vit checkpoints have `66` LoRA tensors and `20` aligner tensors, including
  `audio_bos/audio_eos`. The boundary embeddings are excluded from the pooled retrieval representation by definition, but still
  remain part of complete generation/MMAU restoration. This statement does not apply to the incomplete dynamic FSDP DCPs.
- LoSATok retrieval submit wrapper:
  - `code/huginn_lora/run_eval_huginn_losatok_text_retrieval_swift_5090.sh`
  - it currently targets AudioCaps LoSATok `checkpoint-5604` and `checkpoint-8406`; change its fixed checkpoint list deliberately for other comparisons.

### Current Evaluation Mainline (updated 2026-07-27)

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
  - dedicated legacy-quarter wrapper (fixed `HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=0`) evaluates the completed
    fixed-32 ACAVCAPS checkpoint `36741`:
    `code/huginn_lora/run_generate_clotho_caption_samples_losatok_legacy_acavcaps_quarter_5090.sh`.

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
  - dedicated legacy-quarter wrapper for fixed-32 ACAVCAPS `checkpoint-36741`:
    `code/huginn_lora/run_eval_mmau_test_mini_losatok_legacy_acavcaps_quarter_5090.sh`
- scoring protocol:
  - this is multiple-choice evaluation, not free caption generation
  - for every complete answer choice, the custom evaluator computes its mean teacher-forced token log-probability conditioned on audio and prompt
  - it selects the highest-scoring complete choice and compares it against the labeled answer
  - metadata fields (`task`, `difficulty`, `sub-category`, etc.) are used for result aggregation, never passed to the model as answer hints
- runtime behavior:
  - full evaluation appends and `fsync`s a JSONL result per sample, then skips already completed IDs only when the saved run configuration matches
  - use distinct output directories for different checkpoints or recurrence values
  - `MMAU_NUM_STEPS` maps to the evaluator's `--num-steps`; unset means the default model recurrence
- explicitly prepared current evaluation:
  - completed fixed-32 ACAVCAPS-quarter `checkpoint-36741` above;
  - MMAU output: `outputs/mmau_test_mini_losatok_legacy_fixed32_acavcaps_quarter_e1_checkpoint36741`;
  - Clotho qualitative output:
    `outputs/huginn_losatok_legacy_fixed32_acavcaps_quarter_e1_checkpoint36741_clotho_caption_samples`;
  - no MMAU score or generated-sample result has been supplied, so this README must not infer either result.
- formal MMAU note:
  - mini is for local development and has answers
  - the formal hidden-answer set is a separate acquisition/submission step; final predictions must preserve the selected complete option text in the official submission JSON format

### Historical/parallel LoSATok Swift training entrypoints

These are the reproducible LoSATok entrypoints, not the current Whisper dynamic-30s formal launchers. Prefer the exact
paths below only when the user explicitly selects the LoSATok branch; never cross-load their checkpoints into Whisper.

- dynamic-90s two-GPU FSDP2 AudioCaps-v2 formal source training (completed; retains the reproducible route):
  - `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2.sh`
  - `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2_5090.sh`
- dynamic-90s two-GPU FSDP2 quarter-ACAVCAPS formal continuation (historical/parallel route):
  - `code/huginn_lora/scripts/train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh`
  - `code/huginn_lora/run_train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh`
  - `code/huginn_lora/scripts/smoke_acavcaps_wds_huginn_losatok_dynamic90s_quarter_warmstart_save_reload_fsdp2.sh`
- legacy fixed-32 quarter-ACAVCAPS completed-training route and its complete checkpoint validators:
  - `code/huginn_lora/scripts/train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh`
  - `code/huginn_lora/run_train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh`
  - `code/huginn_lora/scripts/smoke_acavcaps_wds_huginn_losatok_legacy_quarter_warmstart_save_reload.sh`
- shared read-only ACAVCAPS streaming infrastructure:
  - `code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py`
  - `code/huginn_lora/scripts/prepare_acavcaps_wds_quarter_manifest.py`
  - `code/huginn_lora/scripts/inspect_acavcaps_wds_distributed_sharding.py`
- dynamic DCP contract audit:
  - `code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py`
  - `code/huginn_lora/run_inspect_losatok_dynamic_fsdp_checkpoint_5090.sh`

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

### Current Huginn Whisper dynamic-30s formal line

- model/config:
  - `models/huginn-audio-whisper-dynamic90s-v1/raven_modeling_minimal.py`
  - `models/huginn-audio-whisper-dynamic90s-v1/raven_config_minimal.py`
  - `models/huginn-audio-whisper-dynamic90s-v1/config.json`
- shared runtime/plugin:
  - `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`
  - `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_swift.py`
  - `code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_multiplier_swift.py`
  - `code/huginn_lora/plugins/huginn_audio_whisper_dynamic30s_acavcaps_swift.py`
- data and sampler:
  - `code/huginn_lora/data_pipeline/dynamic90s_mixture_rows.py`
  - `code/huginn_lora/data_pipeline/indexed_atomic_mixture.py`
  - `code/huginn_lora/data_pipeline/finite_multiplier_pool.py`
  - `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic30s_training_prerequisites.sh`
  - `code/huginn_lora/scripts/prepare_huginn_whisper_dynamic30s_multiplier_prerequisites.sh`
- formal training:
  - `code/huginn_lora/scripts/train_huginn_audio_whisper_dynamic90s_multitask_fsdp4.sh`
  - `code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh`
  - `code/huginn_lora/scripts/train_huginn_audio_whisper_dynamic30s_multiplier_fsdp4.sh`
  - `code/huginn_lora/run_train_huginn_audio_whisper_dynamic30s_multiplier_fsdp4_5090.sh`
- validation/audit:
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_contract.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_multiplier_pool.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_resume.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_formal_checkpoints.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_checkpoint_resume_markers.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_single_checkpoint.py`
  - `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_25000_5090.sh`
  - `code/huginn_lora/run_inspect_huginn_whisper_dynamic30s_multiplier_checkpoint_25000_5090.sh`
  - `code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4_5090.sh`
  - `code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic30s_multiplier_checkpoint_resume_fsdp4_5090.sh`

### Current ACAVCAPS global manifest and 8-card warm-start

- `code/huginn_lora/scripts/prepare_acavcaps_flat_global_tar_manifest.py`
- `code/huginn_lora/scripts/inspect_acavcaps_flat_global_tar_manifest.py`
- `code/huginn_lora/scripts/prepare_acavcaps_flat_global_tar_manifest_5090.sh`
- `code/huginn_lora/run_prepare_acavcaps_flat_global_tar_manifest_5090.sh`
- `code/huginn_lora/scripts/smoke_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume.sh`
- `code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic30s_acavcaps_fsdp8_warmstart_save_resume_5090.sh`
- `code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic30s_acavcaps_warmstart_resume.py`

The two wrappers whose filenames retain `_5090` but were explicitly moved to the `pdgpu-4090` pool are the manifest
preparation wrapper and the checkpoint-25000 audit wrapper. The 8-card ACAVCAPS smoke wrapper remains a later `-g8`
remote smoke and must not be submitted before the final `checkpoint-46050` is available and audited.

The word `dynamic90s` in these shared paths is a compatibility name. Always read the current 30-second contract and the
formal launcher before changing or submitting a job.

### X-ARES evaluation (unfinished)

- environment/checkpoint/data/API audits:
  - `code/huginn_lora/scripts/inspect_huginn_xares_environment.py`
  - `code/huginn_lora/scripts/inspect_huginn_xares_checkpoint.py`
  - `code/huginn_lora/scripts/inspect_huginn_xares_voxceleb1_data.py`
  - `code/huginn_lora/scripts/inspect_huginn_xares_voxceleb1_api.py`
- encoder/task wrapper:
  - `code/huginn_lora/scripts/huginn_whisper_xares_encoder.py`
  - `code/huginn_lora/scripts/huginn_whisper_xares_encoder_entry.py`
  - `code/huginn_lora/scripts/huginn_xares_voxceleb1_task.py`
- smoke/K-NN launchers:
  - `code/huginn_lora/scripts/smoke_huginn_whisper_xares_encoder.py`
  - `code/huginn_lora/scripts/run_huginn_xares_voxceleb1_knn.sh`
  - `code/huginn_lora/run_smoke_huginn_xares_voxceleb1_knn_4090.sh`
  - `code/huginn_lora/run_eval_huginn_xares_voxceleb1_knn_4090.sh`

This branch is **unfinished**: the preflight, checkpoint, data-path, encoder smoke, and API-contract gates passed, but the
writable-cache VoxCeleb1 K-NN smoke and complete K-NN evaluation have not yet produced a validated score.

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
- `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`
- `code/huginn_lora/scripts/inspect_huginn_audio_whisper_dynamic90s_stage02.py`
- `code/huginn_lora/run_inspect_huginn_audio_whisper_dynamic90s_stage02_5090.sh`
- `code/huginn_lora/plugins/huginn_losatok_swift.py`
- **Dynamic-90s LoSATok FSDP2 model, save/restore, and continuation:**
  - `code/huginn_lora/scripts/inspect_losatok_dynamic_fsdp_checkpoint.py`
  - `code/huginn_lora/run_inspect_losatok_dynamic_fsdp_checkpoint_5090.sh`
  - `code/huginn_lora/scripts/smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.sh`
  - `code/huginn_lora/run_smoke_audiocaps_v2_huginn_losatok_dynamic90s_modules_save_fsdp2_5090.sh`
  - `code/huginn_lora/scripts/train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2.sh`
  - `code/huginn_lora/run_train_audiocaps_v2_huginn_losatok_dynamic90s_swift_lora_fsdp2_5090.sh`
  - `code/huginn_lora/scripts/smoke_acavcaps_wds_huginn_losatok_dynamic90s_quarter_warmstart_save_reload_fsdp2.sh`
  - `code/huginn_lora/run_smoke_acavcaps_wds_huginn_losatok_dynamic90s_quarter_warmstart_save_reload_fsdp2_5090.sh`
  - `code/huginn_lora/scripts/train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh`
  - `code/huginn_lora/run_train_acavcaps_wds_huginn_losatok_dynamic90s_quarter_fsdp2_5090.sh`
- **ACAVCAPS WebDataset and legacy fixed-32 continuation/evaluation:**
  - `code/huginn_lora/plugins/huginn_losatok_acavcaps_wds_swift.py`
  - `code/huginn_lora/scripts/inspect_acavcaps_wds_preflight.py`
  - `code/huginn_lora/scripts/prepare_acavcaps_wds_quarter_manifest.py`
  - `code/huginn_lora/scripts/inspect_acavcaps_wds_distributed_sharding.py`
  - `code/huginn_lora/scripts/smoke_acavcaps_wds_huginn_losatok_legacy_quarter_warmstart_save_reload.sh`
  - `code/huginn_lora/scripts/train_acavcaps_wds_huginn_losatok_legacy_quarter_fixed32_5090.sh`
  - `code/huginn_lora/run_generate_clotho_caption_samples_losatok_legacy_acavcaps_quarter_5090.sh`
  - `code/huginn_lora/run_eval_mmau_test_mini_losatok_legacy_acavcaps_quarter_5090.sh`
  - corresponding `run_..._5090.sh` wrappers in `code/huginn_lora/`
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
7. The current Huginn audio branch is:
   - **current Whisper Swift dynamic-30s formal branch:**
     - original Huginn backbone with Whisper-large audio encoder
     - mono 16-kHz one-chunk input, first-30-second cap, dynamic prefix at 240ms/token
     - Whisper encoder and aligner fully trainable; Huginn native backbone frozen; Huginn-only rank-8 LoRA trainable
     - four-card FSDP4, coarse five-unit wrapping, full-model paired DCP save/resume
     - two separate schedules: hierarchical AAC/ASR no-replacement and finite global multiplier pool
     - X-ARES modality-alignment evaluation is a separate, currently **未完成** evaluation branch; do not report a K-NN
       score until its writable-cache smoke and full VoxCeleb1 run pass
   - historical standalone branch:
     - original Huginn backbone and earlier Whisper-small experiments
     - retained only as historical code/reference; not the current formal route
   - historical fixed-32 Whisper branch:
     - separate model/plugin and checkpoint path; do not cross-load with dynamic-30s checkpoints
   - historical/parallel encoder-replacement LoSATok branch:
     - LoSATok with 16 kHz waveform input and `unified_emb` output
     - Swift LoRA registration/model/template code is locally implemented
     - complete LoSATok is frozen; only aligner plus Huginn LoRA train
     - legacy fixed-32 uses a normal `66`-LoRA plus `20`-aligner adapter/vit checkpoint and completed quarter-ACAVCAPS
       at `checkpoint-36741`; its evaluation wrappers are prepared, with no result yet reported
     - historical dynamic 90-second two-GPU FSDP2 uses a sharded `66`-LoRA plus `20`-aligner DCP and completed two AudioCaps-v2
       epochs at `checkpoint-2802` and `checkpoint-5604` under `...v0-20260724-115115/`
     - the dynamic quarter-ACAVCAPS formal job starts from `checkpoint-5604` as a weight warm-start; only progress to
       `50/36741` is recorded, so do not claim final completion
     - old dynamic `20260723-054928` checkpoints are `66` / `0`, permanently incomplete, and forbidden
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
     - X-ARES environment/checkpoint/data/API/encoder smoke and VoxCeleb1 K-NN entrypoints; K-NN result still pending

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
10. For current Whisper training requests, use the **dynamic-30s/240ms/token Swift FSDP4 path** documented at the top. The
    shared filenames still contain `dynamic90s`, but the runtime contract is 30 seconds, one chunk, and 240ms/token.
    Keep current Whisper checkpoints separate from fixed-32 Whisper and all LoSATok checkpoints. The old dynamic `66` /
    `0` LoSATok DCPs and old dynamic-90s Whisper smoke artifacts are not valid current resume sources.
11. For current Swift audio training and evaluation, inspect the actual resource flag in the selected wrapper. The active
    four-card multiplier formal run uses `pdgpu-5090`; the manifest-preparation and checkpoint-25000 audit wrappers were
    explicitly changed to `pdgpu-4090` because 5090 was occupied. Filename suffixes such as `_5090` are compatibility
    names and do not by themselves prove the submitted pool.
12. The latest checkpoint-retention requirement is at most two retained checkpoints. The multiplier formal launcher is
    synchronized to `save_total_limit=2`; the multitask formal source currently still passes `4`, so reconcile that
    source/configuration before launching a new multitask formal run.
13. For the current Whisper ACAVCAPS route, use the private flat manifest
    `data/audio_swift/acavcaps/acavcaps_flat_global_tar_shuffle_seed20260723.json`: one global permutation across all
    `1071` tars from stage1/2/3, with no stage training boundaries, plus runtime buffer shuffle `512` within each tar.
    Stage labels are provenance only. Historical LoSATok stage/quarter manifests are separate and must not be mixed into
    this route. Never modify the shared public dataset root or add manual rank sharding on top of Accelerate's
    `DataLoaderDispatcher` behavior.
14. For audio generation and MMAU scoring, do not call generic Hugging Face `generate()` on the multimodal wrapper; use the repository's manual audio-prefill/cache path so RoPE positions include the audio prefix.

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
  - the current Whisper dynamic-30s/240ms/token formal FSDP4 route with trainable Whisper, aligner, and Huginn-only LoRA
  - two isolated current data schedules: hierarchical no-replacement multitask and finite globally shuffled multiplier
  - a historical standalone PyTorch training route
  - a validated legacy fixed-32 LoSATok Swift LoRA route
  - a validated Swift FSDP2 full-parameter route with separate Whisper checkpoint handling
  - a completed LoSATok AudioCaps-v2 LoRA run and completed LoSATok-to-ClothoAQA LoRA continuation
- Dynamic LoSATok is architecturally, computationally, and checkpoint-resume validated. The repaired route saves and reloads
  `66` LoRA plus `20` aligner tensors under two-rank FSDP2; its completed two-epoch AudioCaps-v2 run is the valid source for
  dynamic continuation. The old `20260723-054928` checkpoints still have `66` / `0` and remain permanently excluded from
  evaluation or continuation.

### Current immediate next-step expectation (updated 2026-08-05)

For a new Huginn Whisper request, first determine whether it targets the current formal route or a historical branch.
For the current formal route:

1. Use the dynamic-30s contract: one mono 16-kHz chunk, cap at 30 seconds, 240ms/token, local-longest-prefix padding,
   response-only shifted NTP, Whisper + aligner + Huginn-only LoRA trainable, native Huginn frozen.
2. Keep the two current schedules separate: hierarchical AAC/ASR no-replacement multitask versus the finite globally
   shuffled multiplier pool. Do not substitute one registry, plugin, statistics file, or checkpoint audit for the other.
3. The multiplier run is still the active four-card remote job. Wait for `checkpoint-46050`, then require its final
   terminal success banner, retained-checkpoint audit, and cumulative statistics audit before using it.
4. The next ACAVCAPS run is a new eight-card FSDP2 training. Load only Whisper/aligner/Huginn-LoRA model weights from
   `checkpoint-46050`; initialize optimizer, scheduler, global step, RNG, and ACAVCAPS position from scratch.
5. Use the completed flat ACAVCAPS manifest: one global permutation over all `1071` tars across stage1/2/3, no stage
   boundaries, and per-tar buffer `512`. Leave `ACAVCAPS_FLAT_MAX_TARS` unset for formal training.
6. Before formal ACAVCAPS training, run the bounded eight-card smoke with real tar decode, forward/backward, checkpoint
   save, process exit, and a separate same-world-size resume. Require the warm-start tensor-copy and fresh-state audits
   to pass before submitting the full run.
7. Treat any formal result as complete only after terminal success, final checkpoint audit, and cumulative statistics
   report. A high step count or an intermediate loss line is not completion evidence.

### Historical LoSATok immediate-next-step expectation (updated 2026-07-27)

If a new agent is asked "what should we do now", the best default interpretation is:

1. determine which of the three incompatible paths is requested: legacy fixed-32 LoSATok, dynamic-90s LoSATok FSDP2, or Whisper FSDP. Do not silently mix plugins, checkpoint layouts, or data loaders.
2. for the **dynamic route**, use only checkpoints produced after the complete save/resume smoke repair and require `66 + 20`
   DCP audit success. Never use the historical incomplete dynamic DCPs for MMAU, generation, or ACAVCAPS continuation.
3. for the completed **legacy fixed-32 ACAVCAPS-quarter path**, use
   `...huginn_losatok_acavcaps_wds_legacy_quarter_fixed32_warmstart2802_e1_b8ga4_5090/.../checkpoint-36741` only with
   fixed-32 construction and normal adapter/vit restore. The current required work is its separate MMAU-mini and Clotho
   qualitative generation evaluation; do not infer a result without logs.
4. for the **dynamic ACAVCAPS-quarter path**, use only the audited Dynamic AudioCaps-v2 DCP source
   `...huginn_losatok_dynamic90s_audiocaps_v2_e2_b4ga4_fsdp2_complete/v0-20260724-115115/checkpoint-5604` with
   `HUGINN_LOSATOK_DYNAMIC_AUDIO_TOKENS=1`. It is a weights-only cross-dataset warm-start, not `resume_from_checkpoint`.
   Audit every saved DCP for exactly `66 + 20` tensors before future evaluation or continuation.
5. retain evaluation as a separate line. Legacy adapter/vit checkpoints use the matching manual generation/MMAU restore;
   dynamic DCP checkpoints require their dynamic-DCP restore path. The historical incomplete dynamic DCPs cannot be restored.
6. do local code/docs edits only; all remote work must be submitted through the existing `run_*.sh` wrappers using `vc submit`.

Before any long remote run:

- confirm the intended script is the latest synced version
- confirm the checkpoint path is the one you actually want
- confirm the output `run_name` will not collide with old runs
- for resumable evaluation, confirm the output directory has the intended matching run configuration
- confirm the queue resource request still follows the current rules
- confirm whether the job is:
  - smoke
  - mid
  - actual formal training
  - checkpoint audit/save-reload validation
  - retrieval / generation / benchmark evaluation

### Huginn Whisper dynamic-30s current status (2026-08-05; supersedes all dynamic-90s execution guidance below)

The active Huginn route now uses exactly one dynamic Whisper chunk per sample. Existing `dynamic90s` filenames,
environment variables, model type, and model directory are retained only for Swift/checkpoint tooling compatibility;
their active runtime semantics are now:

- every eligible dataset record is retained regardless of source duration;
- source duration `>30s`: decode and retain only the first `30s`;
- source duration `<=30s`: retain the real duration;
- no 2/3-chunk splitting or concatenation; every local sample has exactly one Whisper chunk;
- one Conv1d compressor with kernel `12`, stride `12`, padding `0`, giving one token per complete `240ms`;
- a complete 30-second input produces `125` audio tokens, plus trainable audio BOS/EOS; shorter inputs remain dynamic;
- local-batch audio-prefix padding is still only to that rank's longest sample and padded prefix positions remain `-100`;
- Whisper, the complete aligner, and Huginn-only rank-8 LoRA remain trainable at `1e-4`; the native Huginn backbone
  and LM head remain frozen.

The active data route builds metadata-only v2 pools under
`data/audio_swift/huginn_whisper_dynamic90s_multitask/v2_dynamic30s/`. It reads source metadata, writes new JSONL/index
files, and does not copy, download, scan, or decode WavCaps audio during pool construction. Missing WavCaps duration
metadata is allowed because duration no longer controls sample eligibility. GigaSpeech retains exact segment durations;
AudioCaps and Clotho retain their existing metadata/verified assumptions. The deterministic hierarchical no-replacement
sampler runs over every non-BBC WavCaps record and every GigaSpeech-L segment, preserving complete per-pool epochs and
exact arbitrary-position resume.

The separate finite multiplier preparation is under
`data/audio_swift/huginn_whisper_dynamic30s_multiplier/v1_gigaspeech_m/`. Its registry expands GigaSpeech-M/AudioCaps/
Clotho/WavCaps components by the requested multipliers, selects a deterministic FreeSound quarter, concatenates the
expanded occurrences, and writes one global permutation. Its production launcher consumes that finite permutation in
order with shuffle disabled; it is not the hierarchical task-weight sampler above.

Run the current gates in this order after syncing code:

```bash
# One metadata-only 1-GPU job: v2 pools -> CPU sampler audit -> four real decode probes.
bash code/huginn_lora/run_prepare_huginn_whisper_dynamic30s_training_prerequisites_5090.sh

# Four-GPU, one-update acceleration Stage 0 diagnostic. It does not change
# attention/checkpoint/reshard behavior and saves no model checkpoint.
bash code/huginn_lora/run_smoke_huginn_whisper_dynamic30s_acceleration_stage0_fsdp4_5090.sh

# Four-GPU, one-update acceleration Stage 1 candidate. It keeps FSDP activation
# checkpointing, but disables only Whisper's internal per-layer checkpointing.
bash code/huginn_lora/run_smoke_huginn_whisper_dynamic30s_acceleration_stage1_fsdp4_5090.sh

# Four-GPU, one-update acceleration Stage 2 worst-case memory gate. It uses
# exact 30-second synthetic inputs and sets only recurrent-core reshard=false.
bash code/huginn_lora/run_smoke_huginn_whisper_dynamic30s_acceleration_stage2_fsdp4_5090.sh

# Four-GPU fresh save at step 4, process exit, cold resume to step 6.
bash code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4_5090.sh

# Only after the selected acceleration configuration and checkpoint/resume gates pass:
# fresh formal training.
bash code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh
```

The acceleration Stage 0 gate uses real deterministic mixture rows with FSDP4, per-device batch `2`, gradient
accumulation `4`, and one optimizer update. It preserves the formal baseline settings
`activation_checkpointing=true`, `vit_gradient_checkpointing=true`, and `reshard_after_forward=true`. Per-rank JSON
audits and the merged `acceleration_stage0_report.json` identify the actual Whisper attention implementation and observed
SDPA calls, map every activation-checkpoint wrapper to its contained FSDP unit, detect whether Whisper has simultaneous
inner and outer checkpointing, report all five units' effective reshard state, and record peak CUDA memory. No attention
backend, checkpoint policy, or FSDP reshard setting is changed by this gate.

Stage 0 confirmed PyTorch SDPA execution and exposed that the original wrapper detector was too narrow: FSDP activation
checkpointing wraps the complete `WhisperEncoder` inside `WhisperEncoderFSDPUnit`, not the FSDP unit class itself. The
shared detector now reports that wrapper as Whisper's outer activation checkpoint, relative to Whisper's internal
per-layer gradient checkpointing.

The Acceleration Stage 1 diagnostic callback is isolated behind
`HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE1_AUDIT_DIR`. Its dedicated smoke uses the same real deterministic data
window, FSDP4, per-device batch `2`, gradient accumulation `4`,
SDPA attention, and five coarse FSDP units as Stage 0. The only training-setting change is
`vit_gradient_checkpointing=false`, while FSDP `activation_checkpointing=true` preserves exactly one checkpoint wrapper
around the complete Whisper encoder. The gate requires zero Whisper internal checkpoint modules, exactly one outer
Whisper wrapper, no double-checkpoint candidate, all five units at `reshard_after_forward=true`, finite loss/grad norm,
nonzero gradients for Huginn LoRA + aligner + Whisper, and zero gradients for the frozen native Huginn backbone. It saves
only per-rank audits and `acceleration_stage1_report.json`, not a model checkpoint. This candidate passed and has now
been adopted by the checkpoint/resume smoke and formal-training launcher: Whisper internal per-layer checkpointing is
disabled, while the single outer FSDP activation-checkpoint wrapper around the complete Whisper encoder remains active.

Acceleration Stage 2 is isolated behind
`HUGINN_AUDIO_DYNAMIC30S_ACCELERATION_STAGE2_AUDIT_DIR` and builds on the Stage 1 checkpoint-dedup candidate. The FSDP
JSON still initializes every unit with `reshard_after_forward=true`; after FSDP2 setup and before the first model
forward, the Stage 2 callback verifies all five runtime states, calls
`HuginnRecurrentCoreFSDPUnit.set_reshard_after_forward(False)`, synchronizes all four ranks, and verifies that only the
recurrent core changed. Train-end auditing confirms that the setting persisted. No normal training launch is affected
unless the dedicated Stage 2 environment variable is present. The Stage 2 candidate passed on all four ranks at about
`172.07s` for the one-update B2/GA4 gate, with peak allocated memory `15.244 GiB` and peak reserved memory
`17.992 GiB`. It has now been adopted through the separate production switch
`HUGINN_AUDIO_DYNAMIC30S_RECURRENT_CORE_RESHARD_AFTER_FORWARD_FALSE=1` in both checkpoint/resume smoke and formal
training; Stage 0/1/2 audit modes remain unset in those production paths.

The Stage 2 gate generates one deterministic mono 16-kHz 30-second WAV and 64 synthetic manifest rows, then consumes
one complete B2/FSDP4/GA4 global batch of 32 samples. Every sample must produce exactly 125 audio content tokens and 127
valid prefix tokens including trainable audio BOS/EOS; every rank must observe eight 127-token prefixes. It retains
Whisper SDPA, one outer complete-Whisper checkpoint, zero internal Whisper checkpoint modules, all existing per-block
activation-checkpoint wrappers, trainable Whisper + aligner + Huginn-only LoRA, and response-only shifted NTP loss. The
gate records recurrent-core forward counts, finite gradients/losses, wall time, and peak CUDA memory. It fails if peak
allocated memory reaches 29 GiB or peak reserved memory reaches 30 GiB, and saves no model checkpoint. Passing this gate
establishes worst-case memory/correctness safety. The user explicitly waived a separate Stage 3 paired comparison, so
the next mandatory gate is the real-data four-GPU save/cold-resume smoke.

The checkpoint/resume smoke now exercises the adopted acceleration contract in two distinct four-rank process groups:
fresh steps `0..4`, full checkpoint save, complete process exit, then cold resume for steps `4..6`. It requires FSDP
activation checkpointing, zero Whisper internal checkpoint modules, exactly one outer complete-Whisper checkpoint
wrapper, and `reshard_after_forward=false` only for `HuginnRecurrentCoreFSDPUnit`. It also re-audits the effective LoRA
rank/alpha/dropout and Huginn-only ownership, Whisper/aligner/LoRA optimizer coverage at `1e-4`, nonzero finite gradients
including trainable audio BOS/EOS, frozen native Huginn parameters, dynamic prefix bounds, prefix-label `-100` masking,
assistant-response-only contiguous supervision, shifted next-token prediction, all five coarse FSDP units, full-model
DCP contents, optimizer/scheduler/per-rank RNG restoration, deterministic no-replacement data continuity, exclusion of
prefetched rows from statistics, cumulative sample/hour accounting, and AAC/ASR task-specific prompt mapping. Each
checkpoint receives `huginn_training_runtime_contract.json`; the offline checkpoint inspector rejects any contract or
trainable-state mismatch. Formal training must not be submitted until this updated smoke passes.

The first launch of this updated smoke stopped before step 1 because the new LoRA audit reported 34 target modules while
still observing exactly 66 LoRA A/B parameter tensors. This was an audit-only false positive: activation checkpointing's
`CheckpointWrapper` delegates `lora_A` to the wrapped recurrent adapter, so an attribute-based scan counted both the
wrapper and the real PEFT Linear. The audit now derives 33 targets from matched `lora_A`/`lora_B` parameter paths and
separately requires 33 modules that directly register `lora_A`; delegated wrappers are not counted. No model topology,
LoRA attachment, checkpoint state, or training behavior was changed by this correction.

Formal training is fixed by user instruction at exactly `20000` optimizer steps; no metadata-duration estimate is used
to choose `max_steps`. With FSDP4, per-device batch `2`, and gradient accumulation `4`, the global batch is `32` and the
run consumes exactly `640000` scheduled samples. It saves full model/optimizer/scheduler/RNG checkpoints every `5000`
steps and retains exactly four: `checkpoint-5000`, `checkpoint-10000`, `checkpoint-15000`, and `checkpoint-20000`.
The frozen plan records the deterministic per-pool sample counts at every checkpoint. Runtime and terminal audits verify
the no-replacement-v2 sampler policy, exact resume position, cumulative sample/duration accounting, and all four
checkpoint contracts. The terminal audit still reports whether actually decoded, 30-second-capped audio strictly
exceeded `3000` hours. If it did not, all four checkpoints remain saved and audited, then the job exits with an error;
it does not start an automatic continuation.

Formal cold resume is allowed only from `checkpoint-5000`, `checkpoint-10000`, or `checkpoint-15000`. The prior
checkpoint directory must contain one unbranched sequence through the selected step, and the new run root must be fresh:

```bash
HUGINN_AUDIO_DYNAMIC90S_FORMAL_RESUME_CHECKPOINT=/absolute/path/to/checkpoint-10000 \
HUGINN_AUDIO_DYNAMIC90S_FORMAL_RUN_ROOT=/absolute/path/to/new-resume-run \
bash code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh
```

The full Torch Profiler route is paused after a Kineto/native `SIGSEGV` on the old recurrent dynamic-90s workload and is
not part of the current launch sequence.

### Latest 2026-08-05 handoff for the next ACAVCAPS phase

The finite multiplier run remains active on four GPUs. The project is waiting for the real final
`checkpoint-46050`; the existence of a checkpoint directory or an intermediate checkpoint is not enough to declare
that run complete. The already-audited `checkpoint-25000` is useful as checkpoint-contract evidence only and is not the
planned ACAVCAPS initialization source.

The ACAVCAPS preparation work is complete. The private manifest and strict preflight establish one global tar order over
all `1071` tars from stage1/2/3, totaling `4,664,169` JSON/FLAC pairs. This route is intentionally not a three-stage
training schedule: the stage fields are provenance, while the training permutation is global. The preparation wrapper
and checkpoint-25000 audit wrapper were submitted to `pdgpu-4090`; their `_5090` filenames are compatibility names.

After `checkpoint-46050` is formally accepted, the next action is the eight-card model-only DCP warm-start smoke. It is
a fresh training state: only Whisper, aligner, and Huginn-only LoRA weights are loaded; optimizer, scheduler, global
step, RNG, and ACAVCAPS position are newly initialized. The smoke must perform real tar decoding, forward/backward,
save, process exit, and a separate eight-card resume before the formal all-tar ACAVCAPS run is submitted.

### Historical Huginn Whisper dynamic-90s status update (2026-07-29; superseded)

At that time, the active Huginn task was an isolated Whisper-large dynamic-90s route, not a LoSATok continuation and not
formal dataset training. The historical fixed-32 Whisper model/plugin and its dataset-specific training/evaluation scripts
were restored to their original paths. The experimental route was isolated under
`models/huginn-audio-whisper-dynamic90s-v1/` and
`code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_swift.py`.

This section is retained as an audit/history record only. The active contract is now the single dynamic-30s chunk,
240ms/token, recurrent-core `reshard_after_forward=false` route documented above; do not use the old 90-second values,
old duration planner, or old checkpoint artifacts for new work.

The active trainability contract changed on 2026-07-30 and supersedes every frozen-Whisper validation result below:

- the complete Whisper-large encoder is trainable at learning rate `1e-4` and remains one whole FSDP unit;
- the complete 14-tensor aligner is trainable at learning rate `1e-4`;
- the 66 Huginn-only LoRA tensors are trainable with rank `8`, alpha `16`, dropout `0.05`, and learning rate `1e-4`;
- the native Huginn backbone and LM head remain frozen; Whisper and the aligner still receive no LoRA modules;
- all old checkpoint-4/6 runs used a different optimizer/model contract and are invalid as resume evidence.

For this isolated route, `audio_encoder` is registered as Swift's `vision_tower`, not as its always-frozen
`generator` branch. Swift 4.1.3 `LoRALLMTuner` constructs LoRA only for the language-model target regex, then explicitly
unfreezes `vision_tower + aligner` before optimizer construction. This gives full-parameter Whisper/aligner training
without attaching LoRA to either module. Historical frozen-Whisper plugins retain their original `generator`
registration.

The first required gate for this contract is
`code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_memory90_fsdp4_5090.sh`. It performs one complete optimizer
update with every sample exactly 90 seconds long, `B=2` per rank, four ranks, and `GA=4`, giving global batch `32`.
Each local forward flattens six 30-second Whisper chunks into one Whisper call. The gate requires nonzero finite
Whisper/aligner/LoRA gradients, no Huginn-base gradients, two 752-position prefixes per rank, complete FSDP2 DTensor
sharding, and per-rank CUDA peak allocated/reserved memory markers. The initial fully-trainable-Whisper B2 attempt
reached `31.18 GiB` before the first recurrent-core all-gather and OOMed while requesting another `362 MiB`. The active
retry therefore preserves B2/GA4 but enables explicit Whisper gradient checkpointing, FSDP activation checkpointing,
and the expandable-segments CUDA allocator. The custom whole-Whisper FSDP unit proxies Swift's gradient-checkpointing
interface to the inner Hugging Face encoder and uses non-reentrant checkpointing for floating log-mel inputs; the marker
gate verifies all three memory controls are actually active. It saves no checkpoint.

Historically, the pre-grouping implementation passed Stage 0-2 remotely, including the production duration contract, real Swift
collator/prefix checks, effective rank-8/alpha-16/dropout-0.05 LoRA audit, frozen Whisper/base audit, and a real backward
pass. The first Stage 3-4 attempt then exposed incomplete wrapping (`64/80` trainable DTensors). The implementation now
uses five coarse callable FSDP units (Whisper whole, aligner whole, prelude 2 blocks, recurrent adapter + 4 blocks, coda
2 blocks), all with `reshard_after_forward=true`; LoRA remains Huginn-only. The revised merged Stage 3-4 FSDP4 gate has
now passed on all four ranks with `640` DTensor parameters and one optimizer update. The 20-step synthetic Stage 5
stability smoke also passed on all ranks with finite losses/gradient norms through `global_step=20`. Those results
predate Whisper unfreezing and remain architecture/history evidence only; the active gate order is the trainable-Whisper
90-second memory smoke followed by the replacement checkpoint/resume smoke documented below.

The former duration contract had no discard threshold and retained inputs through a 90-second cap. That behavior is no
longer active: current training caps every record at 30 seconds and never splits or concatenates chunks.

### Historical Huginn Whisper dynamic-90s data status update (2026-07-30; superseded)

The four full atomic pools are complete: WavCaps excluding BBC Sound Effects, AudioCaps-v2, Clotho-v2 train grouped by
audio, and GigaSpeech-L segment-level ASR. The deterministic indexed hierarchical mixture gate has passed with the
required `AAC=60%` / `ASR=40%` task split and `WavCaps=60%` / `AudioCaps=30%` / `Clotho=10%` inside AAC. Sampling is by
training occurrence, not by precomputed token count. The sampler has since been replaced in place by deterministic
per-pool no-replacement epochs: a pool cannot repeat an atomic record until that pool has been completely covered and
reshuffled for its next epoch. Clotho selects exactly one deterministic caption per occurrence.

The real data-chain gate is
`code/huginn_lora/run_inspect_huginn_whisper_dynamic90s_real_data_chain_5090.sh`. It registers the indexed mixture as a
Swift `IterableDataset`, checks deterministic non-zero-position restart, and decodes exactly one real source item per
pool without loading Whisper/Huginn or materializing converted audio. GigaSpeech Opus segments are decoded on demand
from the read-only public source with ffmpeg segment bounds. Dynamic token counts remain runtime statistics.

That real data-chain gate has now passed and remains valid because it audits data/index/decode behavior without loading
the model. The earlier eight-step real-model gate used frozen Whisper and is therefore historical rather than evidence
for the active trainability contract. Its entry point is
`code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_realdata_fsdp4_5090.sh`: eight real optimizer steps on four
GPUs with checkpoint saving disabled. The script has been updated to re-audit trainable Whisper, Huginn-only rank-8 LoRA,
trainable aligner, finite losses/gradient norms, and per-rank realized dynamic audio-token totals. Its deterministic
global sample window is positions `0..31`, which covers all four pools (`11` WavCaps, `6` AudioCaps, `2` Clotho, and
`13` GigaSpeech; `19` AAC and `13` ASR). Checkpoint save/restart remains the following separate gate.

The earlier real-mixture FSDP4 architecture gate passed, but its old with-replacement sample sequence is no longer the
active sampler contract. The replacement checkpoint gate is
`code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4_5090.sh`. Phase 1 uses one fresh
four-rank process group to consume global mixture positions `0..15`, train through step `4`, and save a complete
full-model FSDP DCP. PEFT's default adapter-only DCP cannot contain trainable Whisper weights, while wrapping the whole
Whisper encoder as `modules_to_save` would deep-copy it. The isolated plugin therefore opts this gate into Accelerate's
paired full-model FSDP2 save/load path. Phase 2 starts a distinct four-rank process group, restores checkpoint `4`, starts the stateless
mixture explicitly at position `16` with Trainer data skipping disabled, consumes positions `16..23`, and reaches step
`6`. The gate requires full Whisper model shards, exactly `66` Huginn LoRA tensors, the PEFT-owned trainable aligner,
restored optimizer state for every trainable Whisper/aligner/LoRA tensor, optimizer step `4`, scheduler epoch
`4`, exact per-rank Python/NumPy/CPU/CUDA RNG restoration, Trainer global step continuity, disjoint process-launch IDs,
nonzero gradients for all three trainable groups, actual Whisper/LoRA/aligner tensor changes, and exact equality of the
frozen Huginn backbone between checkpoints `4` and `6`. It additionally requires zero repeated
pool records across the save/resume boundary, exact forward-consumed positions on all four ranks, and cumulative
per-pool sample and effective-duration statistics. The statistics metadata is carried through the collator and is
committed only after an actual successful training forward, so Swift's duplicate template encoding and Accelerate's
unconsumed prefetch tail are excluded. Each checkpoint contains `audio_training_statistics.json`; cold resume restores
the cumulative counts, effective seconds, next global position, and per-pool epoch offsets. Constant LR is used only
for this short gate so the first phase cannot decay to zero before the resumed updates. All checkpoint-4/6 artifacts
from the previous with-replacement sampler are obsolete and must not be reused.

For checkpoint jobs, PEFT still receives the leaf `modules_to_save` names, while Swift's model-architecture aligner
registration uses their full `audio_aligner.<module>` paths. This distinction is required because Swift 4.1.3 builds
multimodal optimizer groups by matching real `named_parameters()` prefixes; using the leaf names alone leaves all 14
trainable aligner tensors outside the optimizer.

Run the active model gates in this order after syncing code. The no-replacement CPU sampler audit has already passed and
does not need to be repeated unless its code or report changes:

```bash
bash code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_memory90_fsdp4_5090.sh
bash code/huginn_lora/run_smoke_huginn_audio_whisper_dynamic90s_checkpoint_resume_fsdp4_5090.sh
```

The checkpoint job still refuses to start unless the no-replacement v2 report has passed. Its run root stores cumulative snapshots in
`training_statistics/training_statistics.jsonl` and `training_statistics/latest.json`; checkpoint `4` and checkpoint
`6` each store their own `audio_training_statistics.json`. Per-rank forward-consumption JSONL is enabled only for the
short smoke so the audit can prove four-rank aggregation, exact save/resume continuity, zero cross-checkpoint repeats,
and exclusion of the template/prefetch-only rows.

The following was the **superseded** formal-training plan for the former dynamic-90s route and is retained only as a
historical audit record. It must not be used to plan new work; the current dynamic-30s plans and checkpoint retention
are defined in the authoritative section above.

- train for more than `4000` realized source-audio hours;
- retain exactly two formal checkpoints, one at half of the final global-step count and one at completion;
- before formal training, pass a four-GPU FSDP checkpoint smoke that saves, exits all training processes, starts a new
  process group, resumes model/optimizer/scheduler/RNG/data position, and then performs additional finite updates.

The former four-GPU checkpoint save/cold-resume smoke passed. The old planner and its `17800`-step/90-second estimates
are historical:

- step planner: `code/huginn_lora/scripts/plan_huginn_whisper_dynamic90s_formal_training.py`;
- runtime: `code/huginn_lora/scripts/train_huginn_audio_whisper_dynamic90s_multitask_fsdp4.sh`;
- submit wrapper: `code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh`.

The planner reads the remote `pool_registry.json`, uses the fixed source-pool hours (`6500/136/24/2498.217`) and the
deterministic sampler's exact pool selections, adds a `5%` planning reserve above `4000` hours, and rounds the optimizer
step count upward to an even multiple of `100`. With the currently known pool sizes this is approximately `17800`
steps, so the expected checkpoints are approximately `checkpoint-8900` and `checkpoint-17800`; the generated
`formal_training_plan.json` is authoritative for the exact remote values. Duration estimates never replace runtime
accounting: the final gate requires `audio_training_statistics.json` to report more than `4000` actually decoded,
90-second-capped hours.

The former topology used the same four-GPU batch and trainability groups but had different dynamic-90s and checkpoint
settings. Current production uses dynamic-30s/240ms tokens, recurrent-core `reshard_after_forward=false`, no Whisper
internal checkpointing plus one outer Whisper wrapper, and the schedule-specific checkpoint limits documented above.

The formal first-forward audit requires response-only contiguous labels, full audio-prefix `-100` masking, shifted NTP
(`logits[:, :-1]` against `labels[:, 1:]`), the exact optimizer ownership split, all trainables sharded as DTensors,
active Whisper gradient checkpointing, and nonzero finite first-update gradients in Whisper, aligner, and LoRA only.
The final checkpoint audit requires Whisper/aligner/LoRA changes, an unchanged Huginn backbone, complete optimizer,
scheduler, RNG, and cumulative statistics state.

Fresh formal submission must use the wrapper:

```bash
bash code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh
```

To cold-resume from the generated halfway checkpoint, submit a new job and a new run root:

```bash
HUGINN_AUDIO_DYNAMIC90S_FORMAL_RESUME_CHECKPOINT=/absolute/path/to/checkpoint-HALF \
HUGINN_AUDIO_DYNAMIC90S_FORMAL_RUN_ROOT=/absolute/path/to/new-resume-run \
bash code/huginn_lora/run_train_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh
```

#### Historical/paused full-path Torch Profiler for the formal Whisper dynamic-90s route

Before changing the formal topology for throughput, use the isolated four-GPU profiler route:

- profiler hook: `code/huginn_lora/plugins/huginn_whisper_dynamic90s_torch_profiler.py`;
- combined real-mixture plugin: `code/huginn_lora/plugins/huginn_audio_whisper_dynamic90s_mixture_profiler_swift.py`;
- runtime: `code/huginn_lora/scripts/profile_huginn_audio_whisper_dynamic90s_multitask_fsdp4.sh`;
- four-rank result gate: `code/huginn_lora/scripts/inspect_huginn_whisper_dynamic90s_profiler_results.py`;
- submit wrapper: `code/huginn_lora/run_profile_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh`.

The profiler uses the real no-replacement mixture and the formal four-GPU `B2/GA4` model, optimizer, FSDP,
checkpointing, and dataloader configuration. It deliberately saves no checkpoint and reports to no online service. The
default eight optimizer steps provide `32` microbatches per rank: four wait, four warmup, eight active trace collection,
and sixteen post-active microbatches for lower-overhead timing. The default global forward window is positions `0..255`.
It captures CPU/CUDA operators, shapes, FLOP estimates, memory, DataLoaderDispatcher latency, NCCL/FSDP collectives,
coarse module ranges, recurrence draws and recomputation calls, dynamic prefix/segment lengths, local padding, cross-rank
length imbalance, GPU utilization samples, NCCL topology logs, and per-rank TensorBoard/Chrome traces. Python stacks are
off by default because they greatly increase trace size and perturb this recurrent workload; set
`HUGINN_TORCH_PROFILER_WITH_STACK=1` only for a targeted second run.

The aggregate report projects the `17700`-step formal duration from the post-active phase only and labels the projection
as a short-run estimate. Active trace timing must not be used as the formal ETA. Submit through:

```bash
bash code/huginn_lora/run_profile_huginn_audio_whisper_dynamic90s_multitask_fsdp4_5090.sh
```

The authoritative outputs are `profiler_aggregate.json`, four `profiler_summary_rank*.json` files, the per-rank trace
directories, `resource_samples.log`, `nvidia_topology.txt`, and the per-process `nccl.*.log` files below the generated
profiler run root.

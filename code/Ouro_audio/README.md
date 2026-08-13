# Ouro Audio Branch

This directory is the isolated Ouro research line.

The local repository contains only Ouro-specific wrappers, Swift plugins,
audits, and submission launchers. Model weights, tokenizer assets, and the
Hugging Face remote-code snapshot stay outside GitHub on the remote Linux
server, currently under:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B
```

The first milestone is a submitted, text-only native Ouro smoke test and a
text-only ms-swift registration. Whisper and the audio aligner are deliberately
not part of this first milestone.

## Text-only LoRA + gate smoke

The first training experiment is intentionally minimal: three assistant-only
records in one batch, one optimizer update, ordinary causal next-token cross
entropy, Ouro's configured four recurrent steps, and no KV cache. It trains
only explicit LoRA adapters and `early_exit_gate`; the embedding, decoder
backbone, norms, and `lm_head` remain frozen. The MLP `gate_proj` is an LoRA
target, but it is not the separate Ouro early-exit gate.

The tiny dataset is:

```text
code/Ouro_audio/data/ouro_lora_tiny.jsonl
```

The submitted smoke entry point is:

```text
code/Ouro_audio/run_smoke_ouro_swift_lora_4090.sh
```

It uses `pdgpu-4090` and writes a checkpoint plus a JSON audit report. The
audit checks the actual Swift trainer batch, ms-swift's compact-suffix
next-token label alignment, four forward and backward calls through the
shared decoder layer, four gate forward calls and three gate backward calls,
one optimizer step, optimizer membership, gradient capture before
`zero_grad`, frozen-parameter gradients and update probes, and checkpoint
contents. Ouro's last gate output is intentionally not differentiated: with
the default `early_exit_threshold=1.0`, the final exit probability is the
remaining probability mass, so the fourth gate is a forced-exit branch. This
smoke does
not implement the paper's entropy/KL regularizer; that will be a separate
controlled experiment after the ordinary-CE path is stable.

## Remote execution

All GPU work must go through the `vc submit` launcher. Login-node commands are
limited to filesystem and CPU-only checks.

The Ouro environment is expected at:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3/envs/swift_ouro
```

The model loader uses the pinned Hugging Face revision:

```text
574fa66cb8bf5abdc979642d01cf2b79b16bfab1
```

The Swift registration uses a custom `OuroTextLoader`. It installs the same
cache patch immediately after ms-swift loads the model, so normal Swift
inference and later SFT/LoRA entry points do not depend on an audit script
performing a second manual patch.

## KV-cache compatibility

Ouro-1.4B reuses 24 physical layers for 4 recurrent steps, so its native
cache uses 96 logical slots. The pinned Transformers 4.54.1 mask path also
expects `past_key_values.layers[layer_idx].get_mask_sizes(...)`, while the
published Ouro cache leaves `layers` empty. `compat/ouro_cache.py` supplies
the missing layered-cache adapters and preserves the original 96-slot update
layout. It is patched at runtime after loading the remote model code; the
downloaded model snapshot is not edited.

## Current branch status (2026-08-13)

This remains an isolated research branch, separate from the Huginn and
HRM-text lines. Current work has two parts:

1. a validated text-only Ouro-1.4B registration/LoRA path; and
2. an OWL/SAGE multimodal investigation whose data contract is being audited
   before multimodal training is implemented.

The branch uses `ByteDance/Ouro-1.4B`, not Ouro-2.6B. The complete model and
Transformers remote-code snapshot remains on the remote server:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B/
  config.json
  configuration_ouro.py
  modeling_ouro.py
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
```

The tokenizer is loaded from this same directory. The local Git tree contains
wrappers, compatibility code, audits, and launchers only; weights, datasets,
generated reports, and checkpoints stay on the remote Linux server.

### Validated text-only Ouro path

These paths have completed successfully through submitted GPU jobs:

- native Transformers load and generation via `inspect_ouro_native.py`;
- ms-swift registration and generation via `inspect_ouro_swift_inference.py`.

Both use Ouro's configured four recurrent steps. The runtime cache patch in
`compat/ouro_cache.py` is installed after model loading when generation uses
the native cache. The issue was an interface mismatch between the published
`UniversalTransformerCache` and the layered cache/mask API expected by
Transformers 4.54.1; the downloaded `modeling_ouro.py` is not edited.

The text-only LoRA smoke is a deliberately minimal one-update experiment:

```text
total_ut_steps=4
use_cache=false
LoRA rank=8
trainable: Ouro LoRA adapters + early_exit_gate
frozen: Ouro backbone, embeddings, norms, lm_head
loss: ordinary causal next-token cross entropy
```

The smoke has verified the trainable-parameter inventory and Swift trainer
loss path. With `early_exit_threshold=1.0`, the fourth recurrent gate is a
forced-exit branch, so the final gate output is not differentiated like the
first three. This is expected native behavior, not evidence that only three
recurrent steps executed. Entropy/KL regularization is not part of this
ordinary-CE smoke.

### OWL/SAGE branch contract

The current multimodal contract is:

```text
language model: Ouro-1.4B
audio encoder: pretrained OWL SAGE, frozen
audio projector: official OWL 8-layer Q-Former, 64 queries, trainable
Ouro backbone: frozen
Ouro early-exit gate: frozen
Ouro LoRA: rank 8 on q_proj/k_proj/v_proj only, trainable
total_ut_steps: 4
early_exit_threshold: 1.0
use_cache: false during training
Stage 1 and Stage 2: in scope
Stage 3: not currently in scope
```

The authoritative contract is:

```text
code/Ouro_audio/owl/configs/phase0_contract.json
```

The official OWL checkout is an architecture and dataset-loader reference,
not a model asset. It is currently on the remote server at:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL
```

It should not be copied into `models`.

### OWL assets currently available remotely

```text
SAGE checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE/finetuned.pth

BiDepth:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth/
  owl-questions/
  reverb.tar.gz
  reverb_extracted/mp3d_reverb/binaural/...

AudioSet, read-only public source:
/hpc_stor03/public/shared/data/raa/AudioSet
```

The public AudioSet tree is read-only and must never be used as an output
directory. Reports, logs, checkpoints, and generated manifests must be
written under the private `/hpc_stor03/sjtu_home/jinwei.zhang` tree.

### Current BiDepth audit conclusions

The official loader reads the complete file
`qa_data_root/<stage>/<split>.json`; it does not filter records by paper
Type I-IV labels. During training, the official pipeline loads `train.json`
and `val.json`; `test.json` is used separately for evaluation. The loader
resolves AudioSet paths from the AudioSet root, convolves each waveform with
the referenced RIR, pads to the configured ten-second waveform, and averages
the two rendered waveforms when both source pairs are present.

The current training-file audit reported:

```text
stage1-clsdoa:
  records=330714
  single-source=330714
  CLASSIFICATION=165357 -> Type I
  DOA=165357            -> Type II

stage2-single:
  records=599831
  single-source=330714
  dual-source=269117
  CLASSIFICATION=299901
  DOA=299930
  strict dual-source Yes/No answers=0

stage3-mixup:
  records=252138
  dual-source=252138
  treated as the Stage 3 / Type IV CoT partition
```

Stage 1's field mapping is fully validated: there are no unknown
`question_type` values, and examples match event classification versus
absolute direction/distance answers.

The current `stage2-single` file is not equivalent to the paper's Stage 2
Type III set. It contains both single- and dual-source Type I/II-style
records, and the strict normalized answer test found no dual-source answer
whose complete answer is `Yes` or `No`. It must not yet be used as the paper's
Type III curriculum stage.

The audit distinguishes two units:

```text
QA record    = question + answer + source fields
source tuple = audio_id + reverb_id + audio_id2 + reverb_id2
```

The earlier `stage2_delta_after_stage1` comparison was a complete JSON-record
difference, so it did not measure newly introduced acoustic configurations.
The audit code now reports exact-record delta and source-tuple delta
separately, along with normalized answer-form frequencies. The corrected
report must be rerun before a curriculum manifest is built.

The paper target remains approximately:

```text
Stage 1: approximately 270K single-source Type I/II
          + approximately 270K dual-source Type I/II
Stage 2: approximately 300K dual-source Type III Yes/No pairs
Stage 3: approximately 250K dual-source Type IV CoT pairs
```

The released data currently matches Stage 3's scale, but does not yet match
the expected Stage 1/Stage 2 semantic composition. This is a dataset-release
or partitioning issue to resolve, not something to silently repair with
keyword-based classification.

### Next steps

1. Rerun the corrected paper-type audit through the submitted `pdgpu-4090`
   entry point.
2. Inspect normalized Stage 2 answer forms, including possible alternatives
   such as `true/false`, before concluding that Type III is absent.
3. Compare Stage 1/Stage 2 by source tuple and QA record separately.
4. If the released files still lack approximately 300K dual-source Type III
   pairs, locate the correct BiDepth partition or construct a documented
   read-only-derived manifest; do not train Stage 2 from the filename alone.
5. Only then implement the SAGE waveform path, official Q-Former, Ouro audio
   token insertion, freeze audit, one-batch multimodal smoke, and Stage 1/2
   launchers.

All GPU execution must continue to use submitted jobs. The temporary
single-card smoke queue is `pdgpu-4090`; the later formal eight-card target is
`pdgpu-5090`. CPU-only inspection and upload commands may run on the login
node.

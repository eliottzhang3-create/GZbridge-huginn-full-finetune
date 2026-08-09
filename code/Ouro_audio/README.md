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

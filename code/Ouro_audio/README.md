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

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

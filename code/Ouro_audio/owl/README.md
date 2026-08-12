# OWL-SAGE + Ouro-1.4B branch

This directory is an isolated multimodal branch. It does not replace the
validated text-only Ouro registration under `../plugins/ouro_text_swift.py`.

The branch follows OWL's downstream training design:

- use the pretrained SAGE binaural encoder;
- keep SAGE frozen;
- use the official eight-layer, 64-query OWL Q-Former;
- project Q-Former outputs to Ouro-1.4B's 2048-dimensional hidden space;
- keep Ouro's native backbone and early-exit gate frozen;
- train only the Q-Former and rank-8 LoRA adapters on Ouro's Q/K/V modules;
- keep `total_ut_steps=4`, `early_exit_threshold=1.0`, and `use_cache=false` during training;
- execute OWL Stage 1 and Stage 2 only.

The first multimodal smoke uses one RTX 4090 with batch size 8. The later
formal target is eight RTX 5090 cards with one sample per card and global batch
size 8.

The authoritative Phase 0 contract is `configs/phase0_contract.json`.
Weights, datasets, and generated checkpoints remain on the remote Linux
server; they are not committed to this repository.

The existing native Ouro inspection also profiles one complete no-cache
forward. Its report distinguishes unique active physical parameters from the
four-step parameter-use count, reports linear-projection FLOPs and supported
PyTorch-profiler FLOPs, and records the peak CUDA allocation increment.

The downloaded Ouro model directory must retain its remote-code files,
including `configuration_ouro.py` and `modeling_ouro.py`. The local branch
loads them with Transformers `trust_remote_code=True`; copying a second copy
into the local `models` directory is intentionally not part of Phase 0.

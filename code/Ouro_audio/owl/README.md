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

## Phase 1 deep asset audit

After the initial metadata audit, submit the read-only deep audit through the
GPU job system:

```bash
bash code/Ouro_audio/owl/run_inspect_phase1_deep_assets_4090.sh
```

The job uses `pdgpu-4090`, streams `reverb.tar.gz` without extracting it, and
writes `phase1_deep_asset_audit.json`.  Optional environment variables are:

```bash
OWL_AUDIO_ROOT=/path/to/anechoic/root
OWL_SOURCE_ROOT=/path/to/official/OWL/checkout
OWL_PHASE1_DEEP_SHA256=1
```

The report intentionally marks audio references as unresolved when no audio
root is supplied; this is an audit finding, not an assumption that the audio
is present in `reverb.tar.gz`.

## Phase 1 decompressed RIR and official-loader audit

Once the remote archive has been extracted to
`/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth/reverb_extracted`, use the
direct-lookup audit:

```bash
bash code/Ouro_audio/owl/run_inspect_phase1_decompressed_assets_4090.sh
```

This job does not rescan the gzip archive. It checks every `reverb_id` and
`reverb_id2` reference against the extracted tree, loads representative NPY
files with `allow_pickle=false`, records shape/dtype/finite-value statistics,
and reports split-level coverage. It also reads the official remote OWL
checkout and statically audits the dataset loader plus the SAGE, Q-Former,
multimodal wrapper, and official training launcher. The report distinguishes
hard integrity issues from source reuse that may be intentional across the
curriculum stages.

The official OWL checkout is therefore not a model asset and should not be
copied into `models`. The submitted audit reads it from
`OWL_SOURCE_ROOT` (default:
`/hpc_stor03/sjtu_home/jinwei.zhang/code/OWL`). If an offline local copy is
needed for interactive source reading, place it under a code-only path such
as `code/Ouro_audio/owl/vendor/official_owl/`, not under `models/`.

## Phase 1 SAGE native audit

The SAGE audit imports the official OWL SAGE implementation, tries an exact
checkpoint load, runs a synthetic two-channel 32 kHz waveform, and optionally
runs representative real BiDepth audio samples:

```bash
bash code/Ouro_audio/owl/run_inspect_sage_native_4090.sh
```

Before submission, `OWL_SOURCE_ROOT` must point to an official OWL checkout
whose `src/slam_llm/models/SAGE/` directory is present. If the source audio
is available, set `OWL_AUDIO_ROOT` to one or more colon-separated roots. The
job still runs a synthetic forward when no audio root is configured, but the
report remains `incomplete` until real audio references can be resolved.

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

## Stage 1/2 train-contract audit

For the current training decision, use the focused audit instead of the
val/test overlap report:

```bash
bash code/Ouro_audio/owl/run_inspect_phase1_train_contract_4090.sh
```

It only analyzes `stage1-clsdoa/train.json` and `stage2-single/train.json`.
For each stage it reports the single/dual source shape inferred from
`audio_id2` and `reverb_id2`, unique anechoic and RIR references, RIR coverage,
representative NPY payloads, and whether an external `--audio-root` resolves
the `audio_id` references. It also records the exact path/convolution/fixed
length behavior found in the official loader and the official training
launcher. The intended current evaluation policy is `train` for optimization,
`val` for validation, and no use of `test`.

# BAT-Spatial-AST + Ouro-1.4B branch

> Current status: this directory preserves the historical OWL audit notes,
> but the active multimodal branch has switched from OWL/SAGE to BAT with the
> frozen pretrained Spatial-AST encoder. Active implementation code is under
> `../bat/` and the active plugin is under `../plugins/`.

## Current active plan and progress

The active target is BAT's three cumulative language-model stages with
Ouro-1.4B:

```text
AudioSet audio + binaural RIR
    -> BAT waveform renderer [2, 320000]
    -> frozen Spatial-AST FP32 tokens [515, 768]
    -> trainable BAT Q-Former (8 layers, 64 queries)
    -> 64 audio tokens [2048]
    -> Ouro-1.4B with trainable rank-8 LoRA
```

The active training contract is:

- Spatial-AST is frozen and is not trained;
- Q-Former is randomly initialized and trainable;
- Ouro native backbone is frozen;
- Ouro early-exit gate is frozen;
- `total_ut_steps=4`, `early_exit_threshold=1.0`, and `use_cache=false`;
- Ouro LoRA uses rank 8, alpha 32, dropout 0.05, targeting `q_proj` and
  `v_proj`;
- BAT paper optimizer/schedule: AdamW, betas `(0.9, 0.95)`, weight decay
  `0.05`, base learning rate `0.001`, half-cycle cosine decay, and two warmup
  epochs;
- per-device training batch size is 2 on eight 5090 cards, giving global
  batch size 16;
- the curriculum remains Stage I/II/III with epochs `2/2/3`.

The active remote assets are:

```text
Ouro:          /hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B
BAT QA:        /hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end
BAT RIR:       /hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb
Spatial-AST:   /hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST
AST checkpoint:/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth
AudioSet:      /hpc_stor03/public/shared/data/raa/AudioSet  (read-only)
```

### BAT data deduplication result

The three raw training files contain cumulative Stage records. The completed
manifest audit processed `1,665,880` raw rows and retained `872,312` unique
QA records:

```text
Type A: 139,392
Type B: 139,392
Type C: 118,000
Type D: 118,000
Type E: 357,528
Total:  872,312
```

There are `872,193` unique ordered source tuples:

```text
dual-source tuples:   593,528
single-source tuples: 278,665
```

The difference between QA records and source tuples is expected: multiple
different questions can reuse the same rendered audio. QA deduplication does
not collapse different questions merely because their source tuple is equal.

BAT `question_id` values are local/reused identifiers, not global primary
keys. For example, the same ID can occur for DOA, classification, and several
mixup reasoning questions. This is recorded as an audit warning and is not a
deduplication failure.

The unique manifests are stored remotely under:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/unique_union
```

The 16 source shards are under:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/unique_union_shards_16/source_shards
```

### Offline Spatial-AST BF16 feature cache

To avoid repeatedly reading the public AudioSet tree, loading RIR files,
performing FFT convolution, and running the frozen Spatial-AST during model
training, each unique source tuple is precomputed once:

```text
AudioSet + RIR
    -> binaural waveform
    -> Spatial-AST FP32 inference
    -> [515, 768]
    -> BF16 safetensors cache
```

The cache worker writes chunked safetensors and a per-shard `index.jsonl`.
Each shard is processed by one independent GPU process, without DDP/NCCL:

```text
shards 000-007 -> 8 x pdgpu-5090
shards 008-015 -> 8 x pdgpu-3090
```

The formal submission entry is:

```bash
BAT_SOURCE_SHARD_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/unique_union_shards_16/source_shards \
BAT_FEATURE_OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/cache/spatial_ast_bf16 \
BAT_PRECOMPUTE_BATCH_SIZE=4 \
BAT_PRECOMPUTE_CHUNK_SIZE=512 \
bash code/Ouro_audio/bat/run_precompute_bat_spatial_ast_16way_5090_3090.sh
```

The current smoke precompute completed on both RTX 5090 and RTX 3090 for 32
records. The first cache audit correctly opened the generated safetensors and
checked all `32 x 515 x 768` values as finite. Its only reported issue was an
audit-scope mismatch because the smoke cache contained 32 rows while the audit
was initially given the full 54K-row source shard. The audit now supports
`BAT_FEATURE_AUDIT_SOURCE_LIMIT=32`; the limited smoke audit should be rerun
after syncing the latest code.

The precompute worker previously had a chunk-overwrite bug, which has been
fixed: batches are now accumulated until `chunk_size` and then written as one
safetensors file. Smoke outputs created before that fix must not be reused as
formal cache outputs.

After all 16 jobs finish, the expected per-shard report is:

```text
status=ok
missing=0
errors=0
```

Then merge and audit the global index:

```bash
BAT_SOURCE_SHARD_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/unique_union_shards_16/source_shards \
BAT_FEATURE_OUTPUT_ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/cache/spatial_ast_bf16 \
BAT_SOURCE_SHARD_COUNT=16 \
bash code/Ouro_audio/bat/run_merge_bat_spatial_ast_features_5090.sh
```

The feature cache is private output. The public AudioSet directory remains
read-only input only. The full BF16 cache is expected to require roughly
700 GB including shard/index overhead.

### Next steps

1. Re-run the 32-row safetensors audits with the corrected source limit.
2. Monitor all 16 feature jobs and verify every per-shard report.
3. Merge and audit `global_index.jsonl`.
4. Add a cached-feature dataset/template path that uses `source_key` and the
   global feature index instead of online AudioSet/RIR rendering.
5. Run cached-feature multimodal forward/backward and Q-Former+LoRA smoke
   tests, including next-token CE alignment and frozen/trainable parameter
   audits.
6. Only after those checks, launch the formal BAT curriculum training.

The current formal training plugin still uses the online audio renderer; the
feature cache must be integrated and audited before changing the training
manifest or starting formal training.

This directory is an isolated multimodal branch. It does not replace the
validated text-only Ouro registration under `../plugins/ouro_text_swift.py`.

## Historical OWL/SAGE notes

The following sections document the earlier OWL/SAGE investigation. They are
kept for provenance and are not the active BAT training plan below.

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

The current train audit shows that `stage1-clsdoa/train.json` is entirely
single-source, while `stage2-single/train.json` contains the Stage 1
single-source records plus an additional dual-source portion. The audit uses
the source ID fields, rather than the filename alone, to determine this
composition.

The laboratory AudioSet tree is an external read-only asset. Because the JSON
references already contain prefixes such as
`balanced_train/audio/YOOj8XfZGR8c`, pass the AudioSet root itself:

```bash
OWL_AUDIO_ROOT=/hpc_stor03/public/shared/data/raa/AudioSet \
bash code/Ouro_audio/owl/run_inspect_phase1_train_contract_4090.sh
```

Do not pass `.../AudioSet/balanced_train` as the root; that would duplicate the
`balanced_train/audio/` prefix during resolution. The expected waveform for a
reference such as `balanced_train/audio/YOOj8XfZGR8c` is
`.../AudioSet/balanced_train/audio/YOOj8XfZGR8c.wav`.

# Phase 1 asset preparation

The GPU queue for later inspection and training jobs is `pdgpu-5090`.
Downloading and uploading assets does not use a GPU and does not require a
GPU job.

## 1. Download SAGE locally

Run this in the local PowerShell environment where `hf download` is available:

```powershell
New-Item -ItemType Directory -Force .\OWL-assets | Out-Null
hf download BASH-Lab/OWL `
  SAGE/finetuned.pth `
  --repo-type model `
  --revision main `
  --local-dir .\OWL-assets
```

Expected local file:

```text
OWL-assets\SAGE\finetuned.pth
```

Upload it to:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE/finetuned.pth
```

Do not download the other OWL model-zoo subdirectories. Ouro-1.4B remains the
language model for this branch.

## 2. Download BiDepth locally

The public dataset repository currently contains the `owl-questions` folder
and `reverb.tar.gz`; the repository is approximately 5.63 GB.

```powershell
hf download BASH-Lab/BiDepth `
  --repo-type dataset `
  --revision main `
  --local-dir .\BiDepth
```

Expected local structure:

```text
BiDepth\
  owl-questions\
  reverb.tar.gz
  README.md
  .gitattributes
```

Upload the directory while preserving the archive:

```text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth/
  owl-questions/
  reverb.tar.gz
```

Keep `reverb.tar.gz` intact during transfer. After the upload is verified, we
will inspect its contents and extract it on the remote server if needed. This
avoids transferring a potentially very large number of small files.

## 3. Remote directory preparation

On the remote login node, create the destination directories before upload:

```bash
mkdir -p /hpc_stor03/sjtu_home/jinwei.zhang/models/OWL/SAGE
mkdir -p /hpc_stor03/sjtu_home/jinwei.zhang/data/BiDepth
```

## 4. Submit the Phase 1 audit

The audit is CPU-only internally, but its supported entry point is a submitted
job so it follows the project's remote execution policy:

```bash
bash code/Ouro_audio/owl/run_inspect_phase1_remote_assets_4090.sh
```

This submits to `pdgpu-4090` and runs
`scripts/inspect_phase1_remote_assets.sh`. It does not instantiate SAGE on a
GPU; it only audits the checkpoint container and BiDepth JSON metadata.

## 5. Assets not to download yet

Do not download AudioSet-20K or another large anechoic-audio collection yet.
The released OWL launcher has an `anechoic_data_root` argument, but the public
BiDepth archive and the official dataset loader must first be inspected to
determine whether the archive already contains the required training audio or
whether a separate source-audio tree is needed.

Do not copy `configuration_ouro.py` or `modeling_ouro.py` into the local
`models` directory. They already belong to the remote Ouro model directory
and are loaded through `trust_remote_code=True`.

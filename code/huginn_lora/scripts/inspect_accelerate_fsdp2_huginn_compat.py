"""Inspect the installed FSDP2 implementation and Huginn's relevant model metadata.

This is intentionally source-only: it does not allocate the 4.2B parameter model
or start distributed training.  Its output explains an FSDP2 prepare failure before
we make an architecture-changing workaround such as untieing the output head.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path


def print_source(function, label: str) -> None:
    source_file = Path(inspect.getfile(function))
    source_lines, start_line = inspect.getsourcelines(function)
    print(f'========== {label} ==========', flush=True)
    print(f'[source] file={source_file}', flush=True)
    print(f'[source] start_line={start_line} line_count={len(source_lines)}', flush=True)
    for offset, line in enumerate(source_lines):
        print(f'{start_line + offset:5}: {line.rstrip()}', flush=True)


def find_fsdp2_config(swift_root: Path) -> None:
    candidates = sorted(swift_root.rglob('fsdp2.json'))
    print('========== SWIFT FSDP2 CONFIG ==========', flush=True)
    if not candidates:
        print('[fsdp2_config] no fsdp2.json found below Swift package root', flush=True)
        return
    for path in candidates:
        print(f'[fsdp2_config] path={path}', flush=True)
        try:
            content = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[fsdp2_config] read_error={exc!r}', flush=True)
        else:
            print(f'[fsdp2_config] content={json.dumps(content, sort_keys=True)}', flush=True)


def inspect_huginn_metadata(repo_root: Path) -> None:
    config_path = repo_root / 'models' / 'huginn-0125' / 'config.json'
    source_path = repo_root / 'models' / 'huginn-0125' / 'raven_modeling_minimal.py'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    source = source_path.read_text(encoding='utf-8').splitlines()

    print('========== HUGINN FSDP-RELEVANT METADATA ==========', flush=True)
    print(f'[huginn] config_path={config_path}', flush=True)
    print(f'[huginn] tie_embeddings={config.get("tie_embeddings")!r}', flush=True)
    print(f'[huginn] model_source={source_path}', flush=True)

    keywords = ('_tied_weights_keys', '_no_split_modules', 'self.tie_weights()', 'self.lm_head')
    for line_number, line in enumerate(source, start=1):
        if any(keyword in line for keyword in keywords):
            print(f'[huginn-source] {source_path}:{line_number}: {line.strip()}', flush=True)


def main() -> None:
    import accelerate
    import torch
    import swift
    from accelerate.utils import fsdp_utils

    repo_root = Path(__file__).resolve().parents[3]
    print('========== ACCELERATE FSDP2 HUGINN COMPAT INSPECT ==========', flush=True)
    print(f'[environment] python_torch={torch.__version__}', flush=True)
    print(f'[environment] accelerate={accelerate.__version__}', flush=True)
    print(f'[environment] accelerate_root={Path(accelerate.__file__).resolve().parent}', flush=True)
    print(f'[environment] swift={swift.__version__}', flush=True)
    print(f'[environment] repo_root={repo_root}', flush=True)

    print_source(fsdp_utils.fsdp2_load_full_state_dict, 'ACCELERATE FSDP2 LOAD FULL STATE DICT')
    print_source(fsdp_utils.fsdp2_prepare_model, 'ACCELERATE FSDP2 PREPARE MODEL')
    find_fsdp2_config(Path(swift.__file__).resolve().parent)
    inspect_huginn_metadata(repo_root)
    print('========== ACCELERATE FSDP2 HUGINN COMPAT INSPECT DONE ==========', flush=True)


if __name__ == '__main__':
    main()

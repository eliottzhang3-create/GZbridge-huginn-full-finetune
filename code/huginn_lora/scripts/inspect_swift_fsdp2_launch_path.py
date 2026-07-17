"""Print the installed Swift 4.x FSDP2 launch and wrapping path without loading a model."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path


KEYWORDS = ('fsdp2', 'fsdp', 'fully_shard', 'torchrun', 'NPROC_PER_NODE', 'accelerate')


def print_context(path: Path, line_number: int, radius: int = 5) -> None:
    lines = path.read_text(encoding='utf-8').splitlines()
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    for index in range(start, end):
        print(f'{path}:{index + 1}: {lines[index]}')


def print_function_source(function, label: str) -> None:
    source_path = Path(inspect.getfile(function))
    lines, start_line = inspect.getsourcelines(function)
    print(f'========== {label} ==========', flush=True)
    print(f'[source] name={function.__qualname__} file={source_path} start_line={start_line}', flush=True)
    for offset, line in enumerate(lines):
        print(f'{start_line + offset:5}: {line.rstrip()}', flush=True)


def main() -> None:
    import swift
    from swift.arguments.sft_args import SftArguments
    from swift.pipelines.train.sft import SwiftSft

    swift_root = Path(swift.__file__).resolve().parent
    print('========== SWIFT FSDP2 LAUNCH PATH INSPECT ==========', flush=True)
    print(f'[swift] version={swift.__version__}', flush=True)
    print(f'[swift] root={swift_root}', flush=True)
    print(f'[swift] sft_pipeline={inspect.getfile(SwiftSft)}', flush=True)

    print('========== FSDP ARGUMENTS ==========', flush=True)
    fields = {field.name: field for field in dataclasses.fields(SftArguments)}
    for name in ('fsdp', 'fsdp_config', 'deepspeed', 'ddp_backend', 'gradient_checkpointing', 'gradient_checkpointing_kwargs'):
        field = fields.get(name)
        if field is None:
            print(f'[arg] name={name} present=false', flush=True)
        else:
            default = '<missing>' if field.default is dataclasses.MISSING else repr(field.default)
            print(f'[arg] name={name} present=true default={default} type={field.type}', flush=True)

    fsdp_methods = [
        method for name, method in inspect.getmembers(SftArguments, inspect.isfunction) if 'fsdp' in name.lower()
    ]
    if not fsdp_methods:
        print('========== SFT ARGUMENT FSDP METHODS ==========', flush=True)
        print('[source] no SftArguments methods with fsdp in their name', flush=True)
    else:
        for method in fsdp_methods:
            print_function_source(method, 'SFT ARGUMENT FSDP METHOD')

    print('========== SWIFT FSDP2 PRESET REFERENCES ==========', flush=True)
    for source_path in sorted(swift_root.rglob('*.py')):
        if source_path.name != 'sft_args.py':
            continue
        lines = source_path.read_text(encoding='utf-8').splitlines()
        for line_number, line in enumerate(lines, start=1):
            if "'fsdp2'" not in line and 'fsdp2.json' not in line:
                continue
            print(f'[match] {source_path}:{line_number}: {line.strip()}', flush=True)
            print_context(source_path, line_number, radius=12)
    print('========== SWIFT FSDP2 LAUNCH PATH INSPECT DONE ==========', flush=True)


if __name__ == '__main__':
    main()

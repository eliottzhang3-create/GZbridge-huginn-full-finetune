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
    for name in ('fsdp', 'deepspeed', 'ddp_backend', 'gradient_checkpointing', 'gradient_checkpointing_kwargs'):
        field = fields.get(name)
        if field is None:
            print(f'[arg] name={name} present=false', flush=True)
        else:
            default = '<missing>' if field.default is dataclasses.MISSING else repr(field.default)
            print(f'[arg] name={name} present=true default={default} type={field.type}', flush=True)

    print('========== FSDP SOURCE REFERENCES ==========', flush=True)
    match_count = 0
    for source_path in sorted(swift_root.rglob('*.py')):
        relative = source_path.relative_to(swift_root)
        if not any(part in {'arguments', 'cli', 'pipelines', 'trainers', 'utils'} for part in relative.parts):
            continue
        try:
            lines = source_path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not any(keyword in line.lower() for keyword in KEYWORDS):
                continue
            match_count += 1
            print(f'[match] {source_path}:{line_number}: {line.strip()}', flush=True)
            print_context(source_path, line_number)
    print(f'[swift] fsdp_source_match_count={match_count}', flush=True)
    print('========== SWIFT FSDP2 LAUNCH PATH INSPECT DONE ==========', flush=True)


if __name__ == '__main__':
    main()

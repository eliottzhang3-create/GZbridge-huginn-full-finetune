"""Inspect installed Swift support for LoRA warm-start without Trainer-state resume."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path


FIELD_PATTERN = re.compile(r'adapter|resume|checkpoint|ignore_data|tuner', re.IGNORECASE)
SOURCE_SYMBOLS = (
    'resume_only_model',
    'resume_from_checkpoint',
    'restore_callback_states_from_checkpoint',
    'ignore_data_skip',
)


def print_context(path: Path, line_number: int, radius: int = 3) -> None:
    lines = path.read_text(encoding='utf-8').splitlines()
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    for index in range(start, end):
        print(f'{path}:{index + 1}: {lines[index]}')


def inspect_argument_fields() -> None:
    from swift.arguments.sft_args import SftArguments

    print('========== SWIFT WARM-START ARGUMENTS ==========')
    print(f'[args] class={SftArguments}')
    print(f'[args] module={inspect.getfile(SftArguments)}')
    for field in dataclasses.fields(SftArguments):
        if FIELD_PATTERN.search(field.name):
            default = '<missing>' if field.default is dataclasses.MISSING else repr(field.default)
            print(f'[args] name={field.name} default={default} type={field.type}')


def inspect_sources() -> None:
    import swift

    swift_root = Path(swift.__file__).resolve().parent
    print('========== SWIFT WARM-START SOURCE MATCHES ==========')
    print(f'[swift] version={swift.__version__}')
    print(f'[swift] root={swift_root}')
    match_counts = {symbol: 0 for symbol in SOURCE_SYMBOLS}
    for source_path in sorted(swift_root.rglob('*.py')):
        relative_path = source_path.relative_to(swift_root)
        if not any(part in {'arguments', 'pipelines', 'trainers'} for part in relative_path.parts):
            continue
        try:
            lines = source_path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            matched_symbols = [symbol for symbol in SOURCE_SYMBOLS if symbol in line]
            for symbol in matched_symbols:
                match_counts[symbol] += 1
                print(f'[match:{symbol}] {source_path}:{line_number}: {line.strip()}')
                print_context(source_path, line_number)
    print(f'[swift] source_match_counts={match_counts}')


def main() -> None:
    inspect_argument_fields()
    inspect_sources()
    print('========== SWIFT WARM-START INSPECT DONE ==========')


if __name__ == '__main__':
    main()

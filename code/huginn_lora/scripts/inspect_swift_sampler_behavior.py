from __future__ import annotations

import dataclasses
import inspect
import platform
import sys
from pathlib import Path


KEYWORDS = ("shuffle", "sampler", "get_train_dataloader", "RandomSampler", "SequentialSampler", "IterableDataset")


def print_header(title: str) -> None:
    print(f"========== {title} ==========", flush=True)


def print_context(path: Path, line_number: int, radius: int = 3) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    for index in range(start, end):
        print(f"{path}:{index + 1}: {lines[index]}")


def inspect_sft_argument_fields() -> None:
    from swift.arguments.sft_args import SftArguments

    print_header("SFT ARGUMENT FIELDS")
    print(f"[args] class={SftArguments}")
    for field in dataclasses.fields(SftArguments):
        field_name = field.name.lower()
        if any(keyword.lower() in field_name for keyword in KEYWORDS) or field_name in {
            "seed",
            "data_seed",
            "dataset_num_proc",
            "dataloader_num_workers",
        }:
            default = field.default
            if default is dataclasses.MISSING:
                default = "<missing>"
            print(f"[args] name={field.name} default={default} type={field.type}")

    print_header("SFT ARGUMENT FIELD ORIGIN")
    for field_name in ("dataset_shuffle", "train_dataloader_shuffle", "sortish_sampler"):
        for cls in SftArguments.__mro__:
            dataclass_fields = getattr(cls, "__dataclass_fields__", {})
            if field_name in dataclass_fields:
                print(f"[args-origin] field={field_name} class={cls} module={inspect.getfile(cls)}")
                break


def inspect_swift_sources() -> None:
    import swift

    swift_root = Path(swift.__file__).resolve().parent
    print_header("SWIFT SOURCE SEARCH")
    print(f"[swift] root={swift_root}")

    matches: list[tuple[Path, int, str]] = []
    for source_path in sorted(swift_root.rglob("*.py")):
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(keyword in line for keyword in KEYWORDS):
                matches.append((source_path, line_number, line.strip()))

    print(f"[swift] keyword_match_count={len(matches)}")
    for source_path, line_number, line in matches[:80]:
        print(f"[match] {source_path}:{line_number}: {line}")
        print_context(source_path, line_number)


def inspect_pipeline_source() -> None:
    from swift.pipelines.train.sft import SwiftSft

    print_header("SWIFT SFT PIPELINE")
    print(f"[pipeline] class={SwiftSft}")
    print(f"[pipeline] module={inspect.getfile(SwiftSft)}")
    source = inspect.getsource(SwiftSft)
    pipeline_keywords = KEYWORDS + ("trainer", "Trainer", "get_trainer", "trainer_cls", "build_trainer")
    for line_number, line in enumerate(source.splitlines(), start=1):
        if any(keyword in line for keyword in pipeline_keywords):
            print(f"[pipeline-match] {line_number}: {line}")


def inspect_transformers_trainer() -> None:
    import transformers
    from transformers import Trainer, TrainingArguments

    print_header("TRANSFORMERS TRAINER SAMPLER")
    print(f"[transformers] version={transformers.__version__}")
    print(f"[transformers] trainer_file={inspect.getfile(Trainer)}")
    training_arguments_fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    for field_name in ("train_dataloader_shuffle", "data_seed", "dataloader_num_workers", "sortish_sampler"):
        field = training_arguments_fields.get(field_name)
        if field is None:
            print(f"[transformers-args] name={field_name} present=False")
        else:
            print(f"[transformers-args] name={field_name} present=True default={field.default}")

    for method_name in ("_get_train_sampler", "get_train_dataloader"):
        method = getattr(Trainer, method_name)
        print(f"[transformers-method] name={method_name}")
        print(inspect.getsource(method))


def inspect_exact_field_references() -> None:
    import swift
    import transformers

    print_header("EXACT FIELD REFERENCES")
    roots = (Path(swift.__file__).resolve().parent, Path(transformers.__file__).resolve().parent)
    for root in roots:
        for source_path in sorted(root.rglob("*.py")):
            try:
                lines = source_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if "train_dataloader_shuffle" in line or "dataset_shuffle" in line:
                    print(f"[field-reference] {source_path}:{line_number}: {line.strip()}")
                    print_context(source_path, line_number)


def main() -> None:
    import swift

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    print_header("ENV")
    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] platform={platform.platform()}")
    print(f"[env] swift_version={swift.__version__}")
    print(f"[env] swift_file={swift.__file__}")

    inspect_sft_argument_fields()
    inspect_pipeline_source()
    inspect_transformers_trainer()
    inspect_exact_field_references()
    inspect_swift_sources()
    print_header("INSPECT DONE")


if __name__ == "__main__":
    main()

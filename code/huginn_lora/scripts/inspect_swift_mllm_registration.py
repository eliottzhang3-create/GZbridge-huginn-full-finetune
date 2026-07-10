from __future__ import annotations

import argparse
import importlib
import inspect
import platform
import sys
import textwrap
from pathlib import Path
from typing import Iterable


def print_header(title: str):
    print(f"\n========== {title} ==========")


def safe_signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # pragma: no cover - debug helper
        return f"<signature unavailable: {type(exc).__name__}: {exc}>"


def safe_source(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception as exc:  # pragma: no cover - debug helper
        return f"<source unavailable: {type(exc).__name__}: {exc}>"


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def search_swift_sources(swift_root: Path, patterns: list[str], limit: int):
    print_header("Swift Source Search")
    hit_count = 0
    for path in iter_python_files(swift_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in patterns):
                rel_path = path.relative_to(swift_root.parent)
                print(f"{rel_path}:{lineno}: {line.strip()}")
                hit_count += 1
                if hit_count >= limit:
                    print(f"... truncated after {limit} hits")
                    return
    if hit_count == 0:
        print("No matching lines found.")


def describe_symbol(module_name: str, symbol_name: str):
    print_header(f"{module_name}.{symbol_name}")
    module = importlib.import_module(module_name)
    obj = getattr(module, symbol_name)
    print(f"module_file={inspect.getfile(obj)}")
    print(f"signature={safe_signature(obj)}")
    source = safe_source(obj)
    print("source:")
    print(textwrap.indent(source, prefix="  "))
    return obj


def maybe_describe(module_name: str, symbol_name: str):
    try:
        return describe_symbol(module_name, symbol_name)
    except Exception as exc:
        print_header(f"{module_name}.{symbol_name}")
        print(f"unavailable: {type(exc).__name__}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Inspect the installed Swift MLLM registration API.")
    parser.add_argument("--search-limit", type=int, default=40)
    args = parser.parse_args()

    print_header("Environment")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    import swift

    swift_root = Path(swift.__file__).resolve().parent
    print(f"swift_version={getattr(swift, '__version__', 'unknown')}")
    print(f"swift_root={swift_root}")
    print(f"swift_init={Path(swift.__file__).resolve()}")

    multi_model_keys = maybe_describe("swift.model", "MultiModelKeys")
    if multi_model_keys is None:
        multi_model_keys = maybe_describe("swift.llm", "MultiModelKeys")

    maybe_describe("swift.model", "register_model_arch")
    maybe_describe("swift.model", "ModelMeta")
    maybe_describe("swift.model", "register_model")

    if multi_model_keys is not None:
        print_header("MultiModelKeys Constructor Trial")
        for mode, kwargs in [
            ("keyword_model_arch", {"model_arch": "dummy_arch"}),
            ("positional_model_arch", {}),
        ]:
            try:
                if mode == "keyword_model_arch":
                    obj = multi_model_keys(
                        language_model=["lm"],
                        aligner=["aligner"],
                        vision_tower=["tower"],
                        **kwargs,
                    )
                else:
                    obj = multi_model_keys(
                        "dummy_arch",
                        language_model=["lm"],
                        aligner=["aligner"],
                        vision_tower=["tower"],
                    )
                print(f"{mode}=ok repr={obj!r}")
            except Exception as exc:
                print(f"{mode}=fail {type(exc).__name__}: {exc}")

    search_swift_sources(
        swift_root,
        patterns=[
            "register_model_arch(",
            "MultiModelKeys(",
            "is_multimodal=True",
            "vision_tower=[",
            "aligner=[",
        ],
        limit=args.search_limit,
    )


if __name__ == "__main__":
    main()

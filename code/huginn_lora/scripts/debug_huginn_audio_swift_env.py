from __future__ import annotations

import importlib
import inspect
import platform
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_path = repo_root / "code" / "huginn_lora" / "plugins" / "huginn_audio_swift.py"

    print(f"[audio-swift-debug] python={sys.version.split()[0]}")
    print(f"[audio-swift-debug] platform={platform.platform()}")
    print(f"[audio-swift-debug] plugin_path={plugin_path}")

    import swift
    from swift.model import ModelMeta

    print(f"[audio-swift-debug] swift_version={getattr(swift, '__version__', 'unknown')}")
    print(f"[audio-swift-debug] ModelMeta.signature={inspect.signature(ModelMeta)}")

    try:
        from swift.model import MultiModelKeys

        multi_model_source = "swift.model"
    except ImportError:
        from swift.llm import MultiModelKeys  # type: ignore

        multi_model_source = "swift.llm"
    print(f"[audio-swift-debug] MultiModelKeys.source={multi_model_source}")
    print(f"[audio-swift-debug] MultiModelKeys.signature={inspect.signature(MultiModelKeys)}")

    sys.path.insert(0, str(plugin_path.parent))
    importlib.import_module(plugin_path.stem)
    print("[audio-swift-debug] plugin_import=ok")


if __name__ == "__main__":
    main()

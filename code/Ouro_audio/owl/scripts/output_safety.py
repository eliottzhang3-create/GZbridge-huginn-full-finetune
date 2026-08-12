"""Safety checks for OWL audit outputs.

The laboratory AudioSet tree is read-only shared input.  Reports, logs,
checkpoints, caches, and all other generated artifacts must stay in the user's
private storage area.
"""

from __future__ import annotations

from pathlib import Path


PUBLIC_PREFIXES = (
    "/hpc_stor03/public",
    "/hpc_stor03/public/",
)


def assert_private_output(path: Path) -> None:
    """Reject any generated output path under the shared public tree."""

    normalized = str(path.expanduser())
    if any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PUBLIC_PREFIXES
    ):
        raise PermissionError(
            "Refusing to write generated output under the read-only public "
            f"tree: {path}. Use /hpc_stor03/sjtu_home/jinwei.zhang/... instead."
        )

"""Compatibility patch for Ouro's UniversalTransformerCache.

Ouro is a Universal Transformer: its 24 physical layers are reused for four
recurrent steps. Its cache therefore has ``4 * 24 = 96`` logical slots rather
than the usual one slot per physical layer. The model's published cache class
also predates the layered-cache API used by Transformers 4.54.1:

* ``Cache.key_cache`` and ``Cache.value_cache`` are read-only properties in
  the pinned Transformers version; and
* causal-mask construction calls
  ``past_key_values.layers[layer_idx].get_mask_sizes(cache_position)``.

This module preserves Ouro's cache indexing while providing those two
compatibility surfaces. It intentionally does not modify the downloaded
Hugging Face model files.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache


class _OuroCacheLayer:
    """Small adapter for Transformers' layered-cache mask interface.

    Transformers 4.54.1 asks a cache layer for the KV length before Ouro has
    entered the current recurrent step. The corresponding Ouro slot contains
    the previous sequence for that physical layer, so the correct mask length
    is ``past_length + query_length``. The actual tensors remain owned by the
    parent cache and are still updated through Ouro's global slot index.
    """

    def __init__(self, owner: "OuroUniversalTransformerCache", layer_idx: int):
        self.owner = owner
        self.layer_idx = layer_idx

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = int(cache_position.shape[0])
        past_length = self.owner.get_seq_length(self.layer_idx)
        return past_length + query_length, 0


class OuroUniversalTransformerCache(Cache):
    """Transformers-4.54-compatible implementation of Ouro's cache contract."""

    _ouro_cache_compat = True

    def __init__(self, max_cache_size: Optional[int] = None):
        # Do not call Cache.__init__: the upstream Ouro implementation also
        # intentionally avoids it because this cache owns a non-standard
        # recurrent-step layout.
        self._ouro_key_cache: list[Optional[Any]] = []
        self._ouro_value_cache: list[Optional[Any]] = []
        # Ouro passes max_cache_size=total_ut_steps*num_hidden_layers (96 for
        # Ouro-1.4B). Keep one adapter per logical cache slot. A default is
        # useful for callers that instantiate the cache manually, while update
        # still grows the list for a larger explicitly configured model.
        layer_count = max_cache_size if max_cache_size is not None else 96
        self.layers: list[_OuroCacheLayer] = [
            _OuroCacheLayer(self, layer_idx) for layer_idx in range(layer_count)
        ]
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def _ensure_layer_capacity(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(_OuroCacheLayer(self, len(self.layers)))

    @property
    def key_cache(self) -> list[Optional[Any]]:
        return self._ouro_key_cache

    @key_cache.setter
    def key_cache(self, value: list[Optional[Any]]) -> None:
        self._ouro_key_cache = value

    @property
    def value_cache(self) -> list[Optional[Any]]:
        return self._ouro_value_cache

    @value_cache.setter
    def value_cache(self, value: list[Optional[Any]]) -> None:
        self._ouro_value_cache = value

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[Any, Any]:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds configured max_cache_size={self.max_cache_size}. "
                "Check total_ut_steps and num_hidden_layers."
            )

        self._ensure_layer_capacity(layer_idx)
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]
        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            if cached_value is None:
                raise RuntimeError(f"Missing cached value tensor at layer_idx={layer_idx}")
            self.key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.key_cache[layer_idx]
        result_value = self.value_cache[layer_idx]
        if result_key is None or result_value is None:
            raise RuntimeError(f"Cache update produced an empty entry at layer_idx={layer_idx}")
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx:
            return 0
        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0
        return int(cached.shape[2])

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: Any) -> None:
        for idx, (key_entry, value_entry) in enumerate(zip(self.key_cache, self.value_cache)):
            if key_entry is None:
                continue
            if value_entry is None:
                raise RuntimeError(f"Missing cached value tensor at layer_idx={idx}")
            device = key_entry.device
            self.key_cache[idx] = key_entry.index_select(0, beam_idx.to(device))
            self.value_cache[idx] = value_entry.index_select(0, beam_idx.to(device))

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0


def patch_ouro_cache(model: Any) -> dict[str, str]:
    """Replace the remote module's cache class before the first forward."""

    module_name = model.__class__.__module__
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)

    original = getattr(module, "UniversalTransformerCache", None)
    if original is None:
        raise RuntimeError(f"Ouro module has no UniversalTransformerCache: {module_name}")
    if original is not OuroUniversalTransformerCache:
        setattr(module, "UniversalTransformerCache", OuroUniversalTransformerCache)
    return {
        "module": module_name,
        "original_class": f"{original.__module__}.{original.__name__}",
        "patched_class": f"{OuroUniversalTransformerCache.__module__}.{OuroUniversalTransformerCache.__name__}",
    }

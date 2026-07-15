"""Evaluate Swift Huginn audio checkpoints with Clotho audio-text retrieval."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINTS = [
    '/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/'
    'outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-5604',
    '/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/'
    'outputs/huginn_audio_audiocaps_v2_train_e5_b8ga4_5090/v0-20260713-155848/checkpoint-8406',
]
DEFAULT_DATASET_DIR = '/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_caption_huginn'
DEFAULT_PLUGIN_PATH = (
    '/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/'
    'code/huginn_lora/plugins/huginn_audio_swift.py'
)
ALIGNER_PREFIXES = (
    'temporal_compressor.',
    'audio_projector.',
    'audio_boundary_embeddings.',
    'audio_bos',
    'audio_eos',
)
SKIP_STATE_TOKENS = ('optimizer', 'scheduler', 'rng', 'trainer_state', 'training_args')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', default=None, help='Repeat for each checkpoint directory.')
    parser.add_argument('--dataset_dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--eval_manifest', default='test_expand.jsonl')
    parser.add_argument('--plugin_path', default=DEFAULT_PLUGIN_PATH)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_count', default='all', help="Use 'all' or a positive integer.")
    parser.add_argument('--audio_batch_size', type=int, default=8)
    parser.add_argument('--text_batch_size', type=int, default=64)
    parser.add_argument('--max_text_length', type=int, default=192)
    parser.add_argument('--failure_sample_count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=74)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def import_plugin(plugin_path: str) -> ModuleType:
    path = Path(plugin_path)
    if not path.is_file():
        raise FileNotFoundError(f'Plugin not found: {path}')
    module_name = 'huginn_audio_swift_retrieval_plugin'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import plugin from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def maybe_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def extract_references(record: dict[str, Any]) -> list[str]:
    references: list[str] = []
    for key in ('references', 'captions', 'caption_list', 'ref_captions', 'caption', 'text'):
        references.extend(maybe_text_list(record.get(key)))
    return list(dict.fromkeys(references))


def load_clotho_groups(dataset_dir: str, manifest_name: str) -> list[tuple[str, list[str]]]:
    root = Path(dataset_dir)
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Clotho evaluation manifest not found: {manifest_path}')
    if manifest_path.suffix == '.json':
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(payload, list):
            raise ValueError(f'Expected a JSON array: {manifest_path}')
        records = payload
    else:
        records = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    grouped: dict[str, list[str]] = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        raw_audio_path = record.get('audio_path')
        if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
            raise ValueError(f'{manifest_path}:{line_number} has no audio_path')
        audio_path = Path(raw_audio_path)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        refs = extract_references(record)
        if not refs:
            raise ValueError(f'{manifest_path}:{line_number} has no caption references')
        grouped[str(audio_path)].extend(refs)

    output = []
    for audio_path, refs in sorted(grouped.items()):
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f'Clotho audio file not found: {audio_path}')
        output.append((audio_path, list(dict.fromkeys(refs))))
    if not output:
        raise ValueError(f'No Clotho evaluation groups found in {manifest_path}')
    return output


def choose_groups(groups: list[tuple[str, list[str]]], sample_count: str, seed: int) -> list[tuple[str, list[str]]]:
    if sample_count == 'all':
        return groups
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be 'all' or a positive integer")
    if count >= len(groups):
        return groups
    indices = sorted(random.Random(seed).sample(range(len(groups)), count))
    return [groups[index] for index in indices]


def state_dict_from_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file

        payload = load_file(str(path), device='cpu')
    else:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
            payload = payload['state_dict']
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)}


def candidate_target_keys(source_key: str) -> list[str]:
    candidates = {source_key}
    changed = True
    while changed:
        changed = False
        for key in list(candidates):
            for prefix in ('base_model.model.', 'base_model.', 'model.', 'module.'):
                if key.startswith(prefix):
                    stripped = key[len(prefix):]
                    if stripped not in candidates:
                        candidates.add(stripped)
                        changed = True
    normalized = set()
    for key in candidates:
        normalized.add(key)
        normalized.add(key.replace('.modules_to_save.default.', '.'))
        normalized.add(key.replace('.original_module.', '.'))
    return list(normalized)


def load_aligner_state(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, Any]:
    target_state = model.state_dict()
    canonical_target_keys: dict[str, str] = {}
    for target_key in target_state:
        for candidate in candidate_target_keys(target_key):
            if candidate.startswith(ALIGNER_PREFIXES):
                canonical_target_keys.setdefault(candidate, target_key)
    selected: dict[str, torch.Tensor] = {}
    source_keys: list[str] = []
    for path in sorted(checkpoint_dir.rglob('*')):
        if not path.is_file() or path.suffix not in {'.safetensors', '.bin', '.pt', '.pth'}:
            continue
        if any(token in path.name.lower() for token in SKIP_STATE_TOKENS):
            continue
        for source_key, tensor in state_dict_from_file(path).items():
            for target_key in candidate_target_keys(source_key):
                actual_target_key = canonical_target_keys.get(target_key)
                if actual_target_key is not None:
                    if target_state[actual_target_key].shape != tensor.shape:
                        continue
                    selected[actual_target_key] = tensor
                    source_keys.append(source_key)
                    break
    if not selected:
        raise RuntimeError(
            f'No aligner tensors could be recovered from {checkpoint_dir}. '
            f'Available target aligner aliases include: {sorted(canonical_target_keys)[:20]}'
        )
    load_result = model.load_state_dict(selected, strict=False)
    return {
        'loaded_aligner_tensor_count': len(selected),
        'source_key_preview': source_keys[:20],
        'missing_key_count': len(load_result.missing_keys),
        'unexpected_key_count': len(load_result.unexpected_keys),
    }


def load_checkpoint(plugin: ModuleType, checkpoint_dir: str, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_path}')
    base_model = plugin.build_huginn_audio_model(str(plugin.AUDIO_MODEL_DIR))
    # This retrieval definition never runs Huginn recurrent blocks. It uses only
    # the projector-side audio tokens and the frozen raw text embedding table, so
    # LoRA weights cannot affect its result and are deliberately not restored.
    aligner_report = load_aligner_state(base_model, checkpoint_path)
    if any(parameter.requires_grad for parameter in base_model.audio_encoder.parameters()):
        raise RuntimeError('Audio encoder unexpectedly became trainable during evaluation restore')

    processor = plugin.build_huginn_audio_processor()
    base_model.to(device=device, dtype=torch.bfloat16)
    base_model.eval()
    return base_model, processor, {
        'checkpoint_dir': str(checkpoint_path),
        'lora_restored': False,
        'lora_restore_reason': 'raw token embeddings and projector-side audio embeddings do not traverse LoRA modules',
        'audio_encoder_trainable_parameter_count': sum(
            parameter.numel() for parameter in base_model.audio_encoder.parameters() if parameter.requires_grad
        ),
        'aligner_restore': aligner_report,
    }


def mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
    return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def compute_audio_embeddings(
    plugin: ModuleType,
    audio_model: torch.nn.Module,
    processor: Any,
    groups: list[tuple[str, list[str]]],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    feature_extractor = processor.feature_extractor
    sample_rate = int(getattr(feature_extractor, 'sampling_rate', plugin.DEFAULT_SAMPLE_RATE))
    for start in range(0, len(groups), batch_size):
        batch_groups = groups[start:start + batch_size]
        waveforms = [
            plugin.load_audio_file(Path(audio_path), sample_rate, plugin.DEFAULT_MAX_AUDIO_SECONDS)
            for audio_path, _ in batch_groups
        ]
        inputs = feature_extractor(waveforms, sampling_rate=sample_rate, return_tensors='pt')
        features = inputs['input_features'].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            encoded = audio_model.audio_encoder(input_features=features, return_dict=True).last_hidden_state
            projected = audio_model.audio_projector(audio_model.temporal_compressor(encoded))
            embeddings.append(mean_pool(projected).float())
        print(f'[retrieval] audio_batches={start + len(batch_groups)}/{len(groups)}', flush=True)
    return torch.cat(embeddings, dim=0)


def compute_text_embeddings(
    audio_model: torch.nn.Module,
    tokenizer: Any,
    reference_groups: list[list[str]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts: list[str] = []
    owners: list[int] = []
    positions: list[int] = []
    max_refs = max(len(refs) for refs in reference_groups)
    for owner, refs in enumerate(reference_groups):
        for position, text in enumerate(refs):
            texts.append(text if text.strip() else (tokenizer.eos_token or '.'))
            owners.append(owner)
            positions.append(position)

    all_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokenized = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
                add_special_tokens=False,
            )
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            # This is intentionally the original Huginn input embedding table, not recurrent hidden states.
            text_tokens = audio_model.get_input_embeddings()(input_ids)
            all_embeddings.append(mean_pool(text_tokens, attention_mask).float())
    flat = torch.cat(all_embeddings, dim=0)
    grouped = torch.zeros((len(reference_groups), max_refs, flat.shape[-1]), device=device, dtype=flat.dtype)
    mask = torch.zeros((len(reference_groups), max_refs), device=device, dtype=torch.bool)
    for index, embedding in enumerate(flat):
        grouped[owners[index], positions[index]] = embedding
        mask[owners[index], positions[index]] = True
    return grouped, mask


def retrieval_metrics(similarity: torch.Tensor) -> dict[str, Any]:
    target = torch.arange(similarity.shape[0]).unsqueeze(1)

    def ranks(sorted_indices: torch.Tensor) -> torch.Tensor:
        return (sorted_indices == target).nonzero(as_tuple=False)[:, 1] + 1

    ranks_a2t = ranks(torch.argsort(similarity, dim=1, descending=True))
    ranks_t2a = ranks(torch.argsort(similarity, dim=0, descending=True).t())

    def metrics_for(rank_values: torch.Tensor) -> dict[str, float]:
        return {
            'recall@1': float((rank_values <= 1).float().mean().item()),
            'recall@5': float((rank_values <= 5).float().mean().item()),
            'recall@10': float((rank_values <= 10).float().mean().item()),
            'mrr': float((1.0 / rank_values.float()).mean().item()),
        }

    diagonal = torch.diagonal(similarity)
    off_diagonal = similarity[~torch.eye(similarity.shape[0], dtype=torch.bool)]
    return {
        'audio_to_text': metrics_for(ranks_a2t),
        'text_to_audio': metrics_for(ranks_t2a),
        'positive_mean': float(diagonal.mean().item()),
        'negative_mean': float(off_diagonal.mean().item()),
        'gap': float((diagonal.mean() - off_diagonal.mean()).item()),
        'ranks_audio_to_text': ranks_a2t.tolist(),
        'ranks_text_to_audio': ranks_t2a.tolist(),
    }


def failure_examples(similarity: torch.Tensor, groups: list[tuple[str, list[str]]], count: int, seed: int) -> list[dict[str, Any]]:
    top_indices = torch.argsort(similarity, dim=1, descending=True)
    failures = [index for index in range(len(groups)) if int(top_indices[index, 0]) != index]
    if len(failures) > count:
        failures = random.Random(seed).sample(failures, count)
    output = []
    for index in failures:
        output.append({
            'audio_path': groups[index][0],
            'ground_truth_references': groups[index][1],
            'top5': [
                {
                    'candidate_audio_path': groups[candidate][0],
                    'candidate_references': groups[candidate][1],
                    'similarity': float(similarity[index, candidate].item()),
                }
                for candidate in top_indices[index, :5].tolist()
            ],
        })
    return output


def checkpoint_slug(checkpoint_dir: str) -> str:
    path = Path(checkpoint_dir)
    return f'{path.parent.parent.name}_{path.name}'.replace('/', '_')


def evaluate_one_checkpoint(
    plugin: ModuleType,
    checkpoint_dir: str,
    groups: list[tuple[str, list[str]]],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    audio_model, processor, restore = load_checkpoint(plugin, checkpoint_dir, device)
    audio_embeddings = F.normalize(
        compute_audio_embeddings(plugin, audio_model, processor, groups, device, args.audio_batch_size), dim=-1
    )
    text_embeddings, text_mask = compute_text_embeddings(
        audio_model,
        processor.tokenizer,
        [references for _, references in groups],
        device,
        args.text_batch_size,
        args.max_text_length,
    )
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    per_reference_similarity = torch.einsum('nd,mrd->nmr', audio_embeddings, text_embeddings)
    similarity = per_reference_similarity.masked_fill(~text_mask.unsqueeze(0), float('-inf')).max(dim=-1).values.cpu()
    metrics = retrieval_metrics(similarity)
    slug = checkpoint_slug(checkpoint_dir)
    np.save(output_dir / f'similarity_matrix_{slug}.npy', similarity.numpy())
    examples = failure_examples(similarity, groups, args.failure_sample_count, args.seed)
    (output_dir / f'retrieval_failures_{slug}.json').write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    del audio_model
    gc.collect()
    torch.cuda.empty_cache()
    return {'checkpoint_dir': checkpoint_dir, 'restore': restore, 'metrics': metrics}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this retrieval evaluator')
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = choose_groups(load_clotho_groups(args.dataset_dir, args.eval_manifest), args.sample_count, args.seed)
    plugin = import_plugin(args.plugin_path)
    checkpoints = args.checkpoint or DEFAULT_CHECKPOINTS

    print('========== SWIFT HUGINN AUDIO-TEXT RETRIEVAL ==========')
    print(f'[retrieval] dataset_dir={args.dataset_dir}')
    print(f'[retrieval] eval_manifest={args.eval_manifest}')
    print(f'[retrieval] group_count={len(groups)}')
    print('[retrieval] audio_embedding=mean_pool(audio_encoder->temporal_compressor->audio_projector)')
    print('[retrieval] text_embedding=masked_mean(raw_huginn_input_token_embeddings)')
    print('[retrieval] audio_boundary_embeddings=excluded')

    results = [evaluate_one_checkpoint(plugin, checkpoint, groups, args, device, output_dir) for checkpoint in checkpoints]
    payload = {
        'dataset_dir': args.dataset_dir,
        'eval_manifest': args.eval_manifest,
        'group_count': len(groups),
        'sample_count': args.sample_count,
        'embedding_definition': {
            'audio': 'mean pool of projected audio tokens; excludes audio_bos/audio_eos',
            'text': 'masked mean of raw Huginn input token embeddings; no recurrent hidden states',
        },
        'results': results,
    }
    metrics_path = output_dir / 'retrieval_metrics.json'
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[retrieval] metrics_path={metrics_path}')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_AQA_SYSTEM = "You are a helpful assistant that can understand audio and answer questions about it."
DEFAULT_AQA_USER_PREFIX = "Listen to the audio and answer the question.\nQuestion: "
DEFAULT_CAPTION_SYSTEM = "You are a helpful assistant that can understand audio and describe it."
DEFAULT_CAPTION_USER = "Listen to the audio and describe it."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local Huginn audio data into Swift multimodal JSONL.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--task", choices=["aqa", "caption"], required=True)
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Stable source name written into metadata. Defaults to the dataset directory name.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--verify_audio_paths",
        action="store_true",
        help="Require every resolved audio path to exist before committing the converted manifest.",
    )
    return parser.parse_args()


def load_records(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        if manifest_path.suffix.lower() == ".json":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {manifest_path}")
            return payload
        return [json.loads(line) for line in f if line.strip()]


def resolve_audio_path(dataset_dir: Path, record: dict[str, Any]) -> str:
    audio_path = record.get("audio_path") or record.get("audio")
    if not audio_path:
        raise KeyError("Record is missing audio_path/audio")
    path = Path(audio_path)
    if not path.is_absolute():
        path = dataset_dir / path
    return str(path.resolve())


def convert_record(
    dataset_dir: Path,
    record: dict[str, Any],
    task: str,
    dataset_name: str,
    source_manifest: Path,
    source_record_index: int,
    verify_audio_paths: bool,
) -> dict[str, Any]:
    audio_path = resolve_audio_path(dataset_dir, record)
    if verify_audio_paths and not Path(audio_path).is_file():
        raise FileNotFoundError(
            f"Source audio does not exist at record {source_record_index}: {audio_path}"
        )
    if task == "aqa":
        if "question" in record and "answer" in record:
            question = str(record["question"]).strip()
            assistant_content = str(record["answer"]).strip()
            user_content = f"{DEFAULT_AQA_USER_PREFIX}{question}"
        elif "query" in record and "response" in record:
            question = str(record["query"]).strip()
            user_content = question
            assistant_content = str(record["response"]).strip()
        else:
            raise KeyError("AQA record must contain question/answer or query/response")
        if not question or not assistant_content:
            raise ValueError(f"AQA record {source_record_index} has empty question or answer")
        system_content = DEFAULT_AQA_SYSTEM
    else:
        if "caption" in record:
            assistant_content = str(record["caption"]).strip()
        elif "response" in record:
            assistant_content = str(record["response"]).strip()
        else:
            raise KeyError("Caption record must contain caption or response")
        if not assistant_content:
            raise ValueError(f"Caption record {source_record_index} is empty")
        user_content = DEFAULT_CAPTION_USER
        system_content = DEFAULT_CAPTION_SYSTEM

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "audios": [audio_path],
        "metadata": {
            "dataset": dataset_name,
            "task": task,
            "source_manifest": str(source_manifest),
            "source_record_index": source_record_index,
            "source_audio_path": audio_path,
            "source_target": assistant_content,
        },
    }


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / args.input_manifest
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    dataset_name = args.dataset_name or dataset_dir.name

    converted = []
    for source_record_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(
                f"Source record {source_record_index} must be an object, got {type(record).__name__}"
            )
        converted.append(
            convert_record(
                dataset_dir,
                record,
                args.task,
                dataset_name,
                manifest_path,
                source_record_index,
                args.verify_audio_paths,
            ))
    with output_manifest.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[prepare-huginn-audio-dataset] input={manifest_path}")
    print(f"[prepare-huginn-audio-dataset] output={output_manifest}")
    print(
        f"[prepare-huginn-audio-dataset] samples={len(converted)} task={args.task} "
        f"dataset_name={dataset_name} verify_audio_paths={args.verify_audio_paths}"
    )
    print("[prepare-huginn-audio-dataset] first_sample=")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

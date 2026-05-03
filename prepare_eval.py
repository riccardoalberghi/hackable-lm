from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "winogrande",
    "boolq",
]


SMOKE_FIXTURES = {
    "lambada_openai": [
        {"context": "The capital of France is", "target": " Paris"},
        {"context": "The opposite of hot is", "target": " cold"},
    ],
    "hellaswag": [
        {"ctx": "A person opens an umbrella because", "endings": [" it is raining.", " the oven is hot.", " a book fell.", " the phone rang."], "label": 0},
        {"ctx": "The runner crosses the finish line and", "endings": [" sits in a car.", " completes the race.", " paints a wall.", " freezes water."], "label": 1},
    ],
    "piqa": [
        {"goal": "How do you keep a door open?", "sol1": "Use a doorstop.", "sol2": "Turn off the lights.", "label": 0},
        {"goal": "How do you dry wet hands?", "sol1": "Put them in water.", "sol2": "Use a towel.", "label": 1},
    ],
    "arc_easy": [
        {"question": "What do plants need to grow?", "choices": ["sunlight", "plastic", "glass", "sandpaper"], "label": 0},
        {"question": "Which object is used to measure temperature?", "choices": ["thermometer", "ruler", "scale", "clock"], "label": 0},
    ],
    "arc_challenge": [
        {"question": "A metal spoon feels cold because it transfers heat from your hand by", "choices": ["conduction", "reflection", "evaporation", "condensation"], "label": 0},
        {"question": "Which change most likely increases the rate of evaporation?", "choices": ["cooling the liquid", "covering the liquid", "heating the liquid", "freezing the liquid"], "label": 2},
    ],
    "openbookqa": [
        {"question_stem": "A magnet will most likely attract", "choices": ["an iron nail", "a paper cup", "a wooden pencil", "a glass jar"], "label": 0},
        {"question_stem": "Water in a freezer changes into", "choices": ["steam", "ice", "sand", "oil"], "label": 1},
    ],
    "winogrande": [
        {"sentence": "The trophy does not fit in the suitcase because _ is too large.", "option1": "the trophy", "option2": "the suitcase", "answer": "1"},
        {"sentence": "The trophy does not fit in the suitcase because _ is too small.", "option1": "the trophy", "option2": "the suitcase", "answer": "2"},
    ],
    "boolq": [
        {"passage": "Paris is the capital city of France.", "question": "is paris the capital of france", "answer": True},
        {"passage": "The Pacific Ocean is larger than the Atlantic Ocean.", "question": "is the atlantic larger than the pacific", "answer": False},
    ],
}


HF_SPECS = {
    "hellaswag": {"path": "hellaswag", "name": None, "split": "validation"},
    "piqa": {"path": "piqa", "name": None, "split": "validation"},
    "arc_easy": {"path": "ai2_arc", "name": "ARC-Easy", "split": "validation"},
    "arc_challenge": {"path": "ai2_arc", "name": "ARC-Challenge", "split": "validation"},
    "openbookqa": {"path": "openbookqa", "name": "main", "split": "validation"},
    "winogrande": {"path": "winogrande", "name": "winogrande_xl", "split": "validation"},
    "boolq": {"path": "boolq", "name": None, "split": "validation"},
}


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _require_datasets():
    from datasets import load_dataset

    return load_dataset


def _hf_choice_label(answer_key: Any, labels: list[Any]) -> int:
    key = str(answer_key).strip()
    normalized = [str(label).strip() for label in labels]
    if key in normalized:
        return normalized.index(key)
    if key.isdigit():
        one_based = int(key) - 1
        if 0 <= one_based < len(normalized):
            return one_based
    if len(key) == 1 and "A" <= key.upper() <= "Z":
        return ord(key.upper()) - ord("A")
    raise ValueError(f"cannot map answer key {answer_key!r} to labels {labels!r}")


def _convert_hf_row(task: str, row: dict[str, Any]) -> dict[str, Any]:
    if task == "hellaswag":
        return {"ctx": row["ctx"], "endings": list(row["endings"]), "label": int(row["label"])}
    if task == "piqa":
        return {"goal": row["goal"], "sol1": row["sol1"], "sol2": row["sol2"], "label": int(row["label"])}
    if task in {"arc_easy", "arc_challenge"}:
        choices = row["choices"]
        return {
            "question": row["question"],
            "choices": list(choices["text"]),
            "choice_labels": list(choices["label"]),
            "answer_key": row["answerKey"],
            "label": _hf_choice_label(row["answerKey"], list(choices["label"])),
        }
    if task == "openbookqa":
        choices = row["choices"]
        return {
            "question_stem": row["question_stem"],
            "choices": list(choices["text"]),
            "choice_labels": list(choices["label"]),
            "answer_key": row["answerKey"],
            "label": _hf_choice_label(row["answerKey"], list(choices["label"])),
        }
    if task == "winogrande":
        return {
            "sentence": row["sentence"],
            "option1": row["option1"],
            "option2": row["option2"],
            "answer": str(row["answer"]),
        }
    if task == "boolq":
        return {"passage": row["passage"], "question": row["question"], "answer": bool(row["answer"])}
    raise ValueError(f"HF conversion is not implemented for task {task!r}")


def _load_hf_task(task: str, limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
    if task not in HF_SPECS:
        raise ValueError(f"no HF dataset spec for {task!r}; available: {sorted(HF_SPECS)}")
    load_dataset = _require_datasets()
    spec = HF_SPECS[task]
    if spec["name"] is None:
        dataset = load_dataset(spec["path"], split=spec["split"])
        source = f"{spec['path']}:{spec['split']}"
    else:
        dataset = load_dataset(spec["path"], spec["name"], split=spec["split"])
        source = f"{spec['path']}/{spec['name']}:{spec['split']}"
    rows = [_convert_hf_row(task, dict(row)) for row in dataset]
    if limit is not None:
        rows = rows[:limit]
        source = f"{source}:first_{limit}"
    return rows, source


def prepare_eval_data(
    output: str | Path,
    input_dir: str | Path | None = None,
    tasks: list[str] | None = None,
    smoke: bool = False,
    source: str = "smoke",
    limit_per_task: int | None = None,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if input_dir is not None:
        source = "local"
    if source not in {"smoke", "local", "hf"}:
        raise ValueError("source must be one of: smoke, local, hf")
    if source == "local" and input_dir is None:
        raise ValueError("source='local' requires input_dir")
    source_mode = source
    tasks = tasks or DEFAULT_TASKS
    entries = {}
    for task in tasks:
        out_path = output / f"{task}.jsonl"
        if source_mode == "local":
            src = Path(input_dir) / f"{task}.jsonl"
            if not src.exists():
                raise FileNotFoundError(f"missing local eval file for {task}: {src}")
            shutil.copyfile(src, out_path)
            entry_source = str(src)
            smoke_task = False
        elif source_mode == "hf":
            rows, dataset_source = _load_hf_task(task, limit_per_task)
            write_jsonl(out_path, rows)
            entry_source = dataset_source
            smoke_task = False
        else:
            if task not in SMOKE_FIXTURES:
                raise ValueError(f"no smoke fixture for {task}")
            rows = SMOKE_FIXTURES[task]
            if limit_per_task is not None:
                rows = rows[:limit_per_task]
            write_jsonl(out_path, rows)
            entry_source = "built_in_smoke_fixture"
            smoke_task = True
        rows = sum(1 for _ in out_path.open("r", encoding="utf-8"))
        entries[task] = {
            "task_version": "local_jsonl_v1",
            "source_dataset_identifier": entry_source,
            "file": str(out_path),
            "sha256": hash_file(out_path),
            "examples": rows,
            "license_notes": {
                "local": "user-provided local JSONL",
                "hf": "prepared from Hugging Face datasets; review upstream dataset licenses",
                "smoke": "smoke fixture only; not research evidence",
            }[source_mode],
            "smoke_only": smoke_task or smoke,
        }
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "tasks": entries,
    }
    write_json(output / "eval_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local JSONL eval data and manifest.")
    parser.add_argument("--output", default="eval_data")
    parser.add_argument("--input-dir", help="directory containing task-name.jsonl files")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--source", choices=["smoke", "hf"], default="smoke", help="use smoke fixtures or download supported validation splits through HF datasets")
    parser.add_argument("--limit-per-task", type=int, help="write only the first N examples per task")
    args = parser.parse_args()
    manifest = prepare_eval_data(
        args.output,
        args.input_dir,
        [t.strip() for t in args.tasks.split(",") if t.strip()],
        source=args.source,
        limit_per_task=args.limit_per_task,
    )
    print(json.dumps({"output": args.output, "tasks": list(manifest["tasks"])}))


if __name__ == "__main__":
    main()

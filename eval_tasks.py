from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MULTIPLE_CHOICE_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "winogrande",
    "boolq",
]


@dataclass
class TaskResult:
    task: str
    metric: str
    value: float
    n: int
    details: dict[str, Any]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def deterministic_subset(examples: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(examples):
        return examples
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(examples), generator=gen).tolist()
    return [examples[i] for i in perm[:limit]]


@torch.no_grad()
def continuation_logprob(model, tokenizer, context: str, continuation: str, device: torch.device) -> tuple[float, int]:
    prefix = tokenizer.encode(context)
    full = tokenizer.encode(context + continuation)
    if len(full) <= len(prefix):
        return 0.0, 0
    if len(full) < 2:
        return 0.0, 0
    block_size = int(getattr(getattr(model, "config", None), "block_size", len(full) - 1))
    offset = max(0, len(full) - 1 - block_size)
    idx = torch.tensor(full[offset:-1], dtype=torch.long, device=device)[None, :]
    logits, _ = model(idx)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    labels = torch.tensor(full[offset + 1 :], dtype=torch.long, device=device)
    start = max(0, len(prefix) - 1 - offset)
    vals = logp[torch.arange(start, len(labels), device=device), labels[start:]]
    return float(vals.sum().item()), int(vals.numel())


def _as_int_label(label: Any, choices: list[Any] | None = None) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        label = label.strip()
        if choices is not None:
            labels = [str(choice.get("label", "") if isinstance(choice, dict) else choice).strip() for choice in choices]
            if label in labels:
                return labels.index(label)
        if label.isdigit():
            return int(label)
        if len(label) == 1 and "A" <= label.upper() <= "Z":
            return ord(label.upper()) - ord("A")
    raise ValueError(f"cannot parse choice label {label!r}")


def _choices_from_any(value: Any) -> list[str]:
    if isinstance(value, dict):
        if "text" in value:
            return [str(item) for item in value["text"]]
        return [str(item) for item in value.values()]
    return [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value]


def _label_from_answer_key(answer_key: Any, choices: Any) -> int:
    if isinstance(choices, dict):
        labels = [str(label).strip() for label in choices.get("label", [])]
    else:
        labels = [str(choice.get("label", "")).strip() for choice in choices if isinstance(choice, dict)]
    key = str(answer_key).strip()
    if key in labels:
        return labels.index(key)
    if key.isdigit():
        one_based = int(key) - 1
        if 0 <= one_based < len(labels):
            return one_based
    return _as_int_label(key)


def _choice_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _answer_choices(choices: list[str]) -> list[str]:
    return [choice if choice[:1].isspace() else f" {choice}" for choice in choices]


def _continuations_after(context: str, choices: list[str]) -> list[str]:
    return [
        choice if not context or context[-1].isspace() or choice[:1].isspace() or choice[:1] in ".,;:!?)]}" else f" {choice}"
        for choice in choices
    ]


def _extract_hellaswag(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    context = ex.get("ctx") or ex.get("context") or ""
    return context, _continuations_after(context, _choices_from_any(ex["endings"])), _as_int_label(ex["label"])


def _extract_piqa(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    if "choices" in ex:
        choices = _choices_from_any(ex["choices"])
    else:
        choices = [str(ex["sol1"]), str(ex["sol2"])]
    return _choice_prompt(str(ex.get("goal") or ex.get("question") or "")), _answer_choices(choices), _as_int_label(ex["label"])


def _extract_arc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    raw_choices = ex["choices"]
    choices = _choices_from_any(raw_choices)
    label = ex.get("label")
    if label is None:
        label = _label_from_answer_key(ex.get("answerKey") or ex.get("answer_key"), raw_choices)
    else:
        label = _as_int_label(label, raw_choices if isinstance(raw_choices, list) else None)
    return _choice_prompt(str(ex["question"])), _answer_choices(choices), label


def _extract_openbookqa(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    raw_choices = ex["choices"]
    choices = _choices_from_any(raw_choices)
    question = ex.get("question_stem") or ex.get("question") or ex.get("stem") or ""
    label = ex.get("label")
    if label is None:
        label = _label_from_answer_key(ex.get("answerKey") or ex.get("answer_key"), raw_choices)
    else:
        label = _as_int_label(label, raw_choices if isinstance(raw_choices, list) else None)
    return _choice_prompt(str(question)), _answer_choices(choices), label


def _extract_winogrande(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    sentence = str(ex["sentence"])
    options = [str(ex["option1"]), str(ex["option2"])]
    answer = ex.get("answer", ex.get("label"))
    label = int(answer) - 1 if isinstance(answer, str) and answer.strip() in {"1", "2"} else _as_int_label(answer)
    if "_" in sentence:
        before, after = sentence.split("_", 1)
        choices = [f"{option}{after}" for option in options]
        return before, choices, label
    return _choice_prompt(sentence), _answer_choices(options), label


def _extract_boolq(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    passage = str(ex.get("passage", "")).strip()
    question = str(ex.get("question", "")).strip()
    context = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    answer = ex.get("answer", ex.get("label"))
    return context, [" no", " yes"], _as_int_label(answer)


CHOICE_EXTRACTORS = {
    "hellaswag": _extract_hellaswag,
    "piqa": _extract_piqa,
    "arc_easy": _extract_arc,
    "arc_challenge": _extract_arc,
    "openbookqa": _extract_openbookqa,
    "winogrande": _extract_winogrande,
    "boolq": _extract_boolq,
}


def evaluate_multiple_choice(task: str, model, tokenizer, examples: list[dict[str, Any]], device: torch.device) -> TaskResult:
    extractor = CHOICE_EXTRACTORS[task]
    norm_correct = 0
    raw_correct = 0
    skipped = 0
    total_tokens = 0
    for ex in examples:
        context, choices, label = extractor(ex)
        if not 0 <= label < len(choices):
            skipped += 1
            continue
        scores = []
        norm_scores = []
        for choice in choices:
            score, count = continuation_logprob(model, tokenizer, context, choice, device)
            scores.append(score)
            norm_scores.append(score / count)
            total_tokens += count
        raw_pred = max(range(len(scores)), key=lambda i: scores[i])
        norm_pred = max(range(len(norm_scores)), key=lambda i: norm_scores[i])
        raw_correct += int(raw_pred == label)
        norm_correct += int(norm_pred == label)
    n = len(examples) - skipped
    return TaskResult(
        task,
        "accuracy_norm",
        norm_correct / n,
        n,
        {
            "accuracy_raw": raw_correct / n,
            "score": "continuation_logprob_per_token",
            "scored_tokens": total_tokens,
            "skipped": skipped,
        },
    )


def evaluate_lambada(model, tokenizer, examples: list[dict[str, Any]], device: torch.device) -> TaskResult:
    correct = 0
    total = 0
    nll = 0.0
    ntok = 0
    for ex in examples:
        context = ex.get("context", "")
        target = ex.get("target") or ex.get("answer") or ""
        score, count = continuation_logprob(model, tokenizer, context, target, device)
        nll -= score
        ntok += count
        # Exact-token prediction for the first target token is the classic LAMBADA-style fast signal.
        prefix_ids = tokenizer.encode(context)
        target_ids = tokenizer.encode(context + target)[len(prefix_ids) :]
        if prefix_ids and target_ids:
            idx = torch.tensor(prefix_ids, dtype=torch.long, device=device)[None, :]
            logits, _ = model(idx)
            pred = int(logits[0, -1].argmax().item())
            correct += int(pred == target_ids[0])
            total += 1
    return TaskResult("lambada_openai", "accuracy", correct / total, total, {"nll_per_token": nll / ntok, "tokens": ntok})


def evaluate_hellaswag(model, tokenizer, examples: list[dict[str, Any]], device: torch.device) -> TaskResult:
    return evaluate_multiple_choice("hellaswag", model, tokenizer, examples, device)


TASK_EVALUATORS = {
    "lambada_openai": evaluate_lambada,
    "hellaswag": evaluate_hellaswag,
    "piqa": lambda model, tokenizer, examples, device: evaluate_multiple_choice("piqa", model, tokenizer, examples, device),
    "arc_easy": lambda model, tokenizer, examples, device: evaluate_multiple_choice("arc_easy", model, tokenizer, examples, device),
    "arc_challenge": lambda model, tokenizer, examples, device: evaluate_multiple_choice("arc_challenge", model, tokenizer, examples, device),
    "openbookqa": lambda model, tokenizer, examples, device: evaluate_multiple_choice("openbookqa", model, tokenizer, examples, device),
    "winogrande": lambda model, tokenizer, examples, device: evaluate_multiple_choice("winogrande", model, tokenizer, examples, device),
    "boolq": lambda model, tokenizer, examples, device: evaluate_multiple_choice("boolq", model, tokenizer, examples, device),
}


def evaluate_task(task: str, model, tokenizer, examples: list[dict[str, Any]], device: torch.device, limit: int | None, seed: int) -> TaskResult:
    if task not in TASK_EVALUATORS:
        raise ValueError(f"unknown task {task!r}; available tasks: {sorted(TASK_EVALUATORS)}")
    subset = deterministic_subset(examples, limit, seed)
    return TASK_EVALUATORS[task](model, tokenizer, subset, device)

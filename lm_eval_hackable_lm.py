from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import config_from_dict
from kernels import apply_precision_policy
from model import LanguageModel
from repro import load_trusted_checkpoint
from tokenizer import EOS_TOKEN, load_tokenizer

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - exercised only when lm-eval is absent.
    raise ImportError("lm_eval_hackable_lm.py requires `uv sync --locked` or an editable lm-evaluation-harness checkout") from exc


@dataclass
class _Scored:
    tokens: list[int]
    cont_start: int
    cont_len: int


@register_model("hackable_lm")
class SimpleLMHarness(LM):
    """Minimal EleutherAI lm-evaluation-harness adapter for this repo."""

    AUTO_MODEL_CLASS = None

    def __init__(
        self,
        checkpoint: str,
        tokenizer: str | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        dtype: str = "bfloat16",
        compile: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._dtype = getattr(torch, dtype) if dtype else None
        if self._device.type == "cuda" and self._dtype not in {None, torch.bfloat16}:
            raise RuntimeError("hackable-lm checkpoints use bf16 weights on CUDA; pass dtype=bfloat16")

        ckpt = load_trusted_checkpoint(checkpoint, map_location="cpu")
        self._config = config_from_dict(ckpt["config"])
        self.model = LanguageModel(self._config.model)
        self.model = apply_precision_policy(self.model, self._config.precision)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self._device)
        self.model.eval()
        if compile:
            self.model = torch.compile(self.model)

        manifest_data = ckpt.get("run_manifest", {}).get("data", {})
        tokenizer_path = tokenizer or str(Path(manifest_data.get("data_dir", ".")).joinpath("tokenizer.json"))
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        self._eot_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        if self._eot_token_id is None:
            self._eot_token_id = 0

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return int(self._config.model.block_size)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str, **_: Any) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int], **_: Any) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str) -> _Scored:
        context_enc = self.tok_encode(context)
        continuation_enc = self.tok_encode(continuation)
        if not context_enc:
            context_enc = [self.eot_token_id]
        tokens = context_enc + continuation_enc
        if len(tokens) > self.max_length + 1:
            overflow = len(tokens) - (self.max_length + 1)
            tokens = tokens[overflow:]
            cont_start = max(0, len(context_enc) - overflow)
        else:
            cont_start = len(context_enc)
        return _Scored(tokens=tokens, cont_start=cont_start, cont_len=len(continuation_enc))

    @torch.no_grad()
    def _loglikelihood_tokens(self, items: list[_Scored]) -> list[tuple[float, bool]]:
        out: list[tuple[float, bool]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            max_len = max(len(item.tokens) for item in batch)
            idx = torch.full((len(batch), max_len - 1), self.eot_token_id, dtype=torch.long, device=self.device)
            labels = torch.full((len(batch), max_len - 1), -100, dtype=torch.long, device=self.device)
            for row, item in enumerate(batch):
                inp = item.tokens[:-1]
                tgt = item.tokens[1:]
                idx[row, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=self.device)
                labels[row, : len(tgt)] = torch.tensor(tgt, dtype=torch.long, device=self.device)

            logits, _ = self.model(idx)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            greedy = logits.argmax(dim=-1)
            for row, item in enumerate(batch):
                pos0 = max(0, item.cont_start - 1)
                pos1 = min(pos0 + item.cont_len, labels.shape[1])
                if pos1 <= pos0:
                    out.append((0.0, True))
                    continue
                row_labels = labels[row, pos0:pos1]
                row_log_probs = log_probs[row, pos0:pos1]
                ll = row_log_probs.gather(1, row_labels[:, None]).sum().item()
                is_greedy = bool(torch.equal(greedy[row, pos0:pos1], row_labels))
                out.append((float(ll), is_greedy))
        return out

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        items = [self._encode_pair(*request.args) for request in requests]
        return self._loglikelihood_tokens(items)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        scored: list[_Scored] = []
        owner: list[int] = []
        totals = [0.0 for _ in requests]
        for req_i, request in enumerate(requests):
            (text,) = request.args
            tokens = [self.eot_token_id] + self.tok_encode(text)
            step = self.max_length
            for start in range(0, max(0, len(tokens) - 1), step):
                chunk = tokens[start : start + self.max_length + 1]
                if len(chunk) > 1:
                    scored.append(_Scored(tokens=chunk, cont_start=1, cont_len=len(chunk) - 1))
                    owner.append(req_i)
        for req_i, (ll, _) in zip(owner, self._loglikelihood_tokens(scored), strict=True):
            totals[req_i] += ll
        return [(value,) for value in totals]

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs: list[str] = []
        for request in requests:
            context, kwargs = request.args
            until = kwargs.get("until") or []
            until = [until] if isinstance(until, str) else list(until)
            max_gen_toks = int(kwargs.get("max_gen_toks", self.max_gen_toks))
            tokens = self.tok_encode(context) or [self.eot_token_id]
            prompt_len = len(tokens)
            for _ in range(max_gen_toks):
                idx = torch.tensor(tokens[-self.max_length :], dtype=torch.long, device=self.device)[None, :]
                logits, _ = self.model(idx)
                next_id = int(logits[0, -1].argmax().item())
                tokens.append(next_id)
                text = self.tok_decode(tokens[prompt_len:])
                if any(stop and stop in text for stop in until):
                    break
            text = self.tok_decode(tokens[prompt_len:])
            for stop in until:
                if stop and stop in text:
                    text = text.split(stop, 1)[0]
            outputs.append(text)
        return outputs

# Agent Notes

This repo is intentionally flat. Keep architecture changes local to `model.py`,
optimizer changes local to `optim.py`, data sampling changes local to `data.py`,
and tokenizer/preprocessing changes local to `tokenizer.py` and
`prepare_data.py`.

The golden training path is CUDA + bf16 + Muon/AdamW + FlashAttention 2 +
Liger kernels where available + `torch.compile`. CPU paths exist only in
`tests/` for correctness smoke checks.

For fair comparisons, keep the manifest fields aligned: data hashes, tokenizer
hash, split seed, sequence length, global batch tokens, optimizer grouping,
precision, kernel backend, scaling policy, and seed.

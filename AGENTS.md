# Agent Notes

This repo is intentionally flat. Keep architecture changes local to `model.py`,
optimizer changes local to `optim.py`, data sampling changes local to `data.py`,
and tokenizer/preprocessing changes local to `tokenizer.py` and
`prepare_data.py`.

The golden training path is CUDA + local tensorwise FP8 for eligible linear
layers + Muon/AdamW + FlashAttention 2 + `torch.compile`. CPU paths exist only in
`tests.py` for correctness smoke checks.

For fair comparisons, keep the manifest fields aligned: data hashes, tokenizer
hash, split seed, sequence length, global batch tokens, optimizer grouping,
precision, kernel backend, scaling policy, and seed.

# Agent Notes

Always implement the simplest solution that is easy to read. Prefer direct,
obvious code over abstractions, indirection, clever helpers, defensive wrappers,
or speculative generality.

Assume the user, developer, and local callers are well behaved. Do not add
try/catch blocks, assertions, validation layers, fallback paths, or error
handling for unrealistic misuse or impossible states. Handle real errors that
the current code path can actually encounter; ignore hypotheticals.

This project is early. Backward compatibility is not required. If a direct
breaking change makes the code simpler, make the breaking change and keep the
surrounding code consistent.

The repo is intentionally flat. Keep architecture changes local to `model.py`,
optimizer changes local to `optim.py`, data sampling changes local to `data.py`,
and tokenizer/preprocessing changes local to `tokenizer.py` and
`prepare_data.py`.

CPU paths exist only in `tests/` for correctness smoke checks.

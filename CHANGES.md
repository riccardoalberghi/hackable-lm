Changes in the custom FP8 path:

- Added `fp8.py` with a nanochat-style `Float8Linear` built on `torch._scaled_mm`.
- Removed the torchao dependency and replaced the old precision mode with `fp8`.
- Removed CUDA autocast from training and validation; `model.py` now casts activations explicitly.
- Added a local `Linear` subclass so non-FP8 linear layers cast weights to the activation dtype.
- Updated manifests to record the custom FP8 implementation instead of a torchao version.
- Kept Liger only for fused LM-head + cross-entropy; RMSNorm uses the local torch path.
- Kept `lm_head` out of FP8 conversion so it runs as BF16 compute.

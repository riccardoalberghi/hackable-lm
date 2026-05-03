# Custom Triton Kernels

This directory is reserved for opt-in custom kernels. Each production kernel
added here should include:

- a Torch reference implementation for tests only
- forward correctness tests
- backward correctness tests where applicable
- a microbenchmark against the Torch reference
- dtype, shape, layout, and determinism notes

The base training path uses local tensorwise FP8 linears, Liger fused linear CE, and Torch norm/MLP nonlinear ops.

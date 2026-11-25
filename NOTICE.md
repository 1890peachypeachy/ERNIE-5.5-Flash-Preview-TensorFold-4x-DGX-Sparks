# Notice

- **TensorFold** (https://github.com/ashhart/TensorFold), Apache License 2.0. `patches/tensorfold-ernie55.patch`
  modifies TensorFold commit `549b55a`: it adds an ERNIE 5.5 family (`zig/src/families/ernie55/`,
  `zig/kernels/cuda/ernie55/`, `zig/build/ernie55.zig`, ERNIE chat/tool parsers) and edits shared build, server,
  tokenizer, CUDA and native registry files. The changed files carry the changes described in `docs/LOG.md`.
  TensorFold's own NOTICE and third-party notices apply to the patched tree.
- **ERNIE-5.5-Flash-Preview** weights (baidu/ERNIE-5.5-Flash-Preview) are not included; their license governs their use.
- **vLLM-ERNIE** (baidu/vLLM-ERNIE) and vLLM were read as references for the DSpark drafter and the bench harness.
- Original work in this repo (kernels in `kernels/harness/`, scripts, docs) is released under Apache License 2.0 (`LICENSE`).

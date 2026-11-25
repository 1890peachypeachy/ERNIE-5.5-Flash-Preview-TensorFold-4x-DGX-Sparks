# ERNIE-5.5-Flash-Preview on TensorFold — 4x DGX Spark, TP4

Running Baidu's **ERNIE-5.5-Flash-Preview** (280B total / 14B active; MXFP4 routed experts, block-FP8 dense, mHC, HCA +
MLA-DSA attention, bundled 3-layer DSpark drafter) on **[TensorFold-native](https://github.com/ashhart/TensorFold)**
at tensor-parallel 4 across four NVIDIA DGX Spark (GB10, sm_121) nodes over RoCE. TensorFold has no ERNIE support
upstream; this repo carries the port as a patch plus the kernels, harnesses, scripts and measurements.

The goal is to beat the [vLLM recipe for the same model and hardware](https://github.com/1890peachypeachy/ERNIE-5.5-Flash-Preview-4x-DGX-Sparks).
Status: **working and correct, faster than vLLM on agent-style repeat prompts (prompt cache), behind on cold prefill and
concurrency.** Experimental.

## Current numbers (2026-11-24)

Same harness as the vLLM recipe (`scripts/bench/bench.py` there: thinking on, temperature 0.6, 768-token replies,
unique prompts), same four nodes, one lane at a time.

| | TensorFold (this repo) | vLLM recipe |
|---|---:|---:|
| c1 decode, tok/s per stream (median) | 40.5 | 46.6 |
| c2 aggregate tok/s | 41.8 | 74.7 |
| c4 aggregate tok/s | 55.2 | 114.4 |
| TTFT, short prompt | 0.43 s | 0.21 s |
| Prefill, 6K / 24K prompt | 882 / 872 tok/s | ~1,500 |
| Prefill, 66K prompt (needle found) | 780 tok/s | ~1,450 (88K) |
| Repeat / next-turn prompt, 6.6K cached | 0.68 s | prefix cache |

From the starting point of the port (the first build that stayed up): cold prefill 273 -> 780-890 tok/s and flat to 66K
(was 312 tok/s at 66K), a 6.6K cold prompt 22 s -> 7.6 s, prompt cache added (0.7 s warm), c4 27 -> 55 tok/s.
Raw results: `results/`. Full history with every measurement: `docs/LOG.md`.

## What the patch adds to TensorFold (`patches/tensorfold-ernie55.patch`, against `549b55a`)

- The ERNIE 5.5 family: weights (block-FP8 + MXFP4 loading, TP4 sharding), forward (mHC, HCA with C128 compressor,
  MLA-DSA with the lightning indexer, latent MoE with SITU, VHA, MultiMax), DSpark drafter with its Markov head,
  sampler, TP follow protocol, ERNIE chat template and tool parser.
- Fixes: a depth-0 draft request that killed worker ranks; a division-by-zero in the drafter's prefill absorb; a
  scratch overflow in the DSA indexer past ~65K context.
- Prompt cache (`snap_*` hooks over the TP follow protocol, 256-token grid).
- Tensor-core kernels for prefill: expert-major MXFP4 MoE, DSA indexer scores, `absorb_q` / `expand_v`; tiled mHC
  pre-mix; 1024-row prompt chunks; NCCL ring all-reduce for large messages.
- Batched verify: every stream's window in one forward (concurrent greedy outputs equal solo ones).

## Layout

| Path | What |
|---|---|
| `patches/tensorfold-ernie55.patch` | The port, `git apply` on TensorFold `549b55a` |
| `kernels/harness/` | Standalone parity + timing harnesses for every new kernel (`*_test.cu`) and the kernel sources as first written (`*.cuh`) |
| `scripts/` | `start/stop/deploy` (cluster from `cluster.env`), `build_fatbins.sh`, `suite.py` + `ebench.py` (correctness + speed) |
| `results/` | JSON from every measured build |
| `docs/LOG.md` | Work log: profiles, decisions, before/after tables |
| `CONTRIBUTING.md` / `AGENTS.md` | How to work here; pitfalls |

## Build and run

Needs: 4x DGX Spark (or GB10) with CUDA 13, RoCE between them, the checkpoint at the same path on every node (NFS works),
Zig 0.17, nvcc.

```bash
git clone https://github.com/ashhart/TensorFold tensorfold-ernie && cd tensorfold-ernie
git checkout 549b55a129a45656a5eeef6c320ead37f833a73a
git apply /path/to/this/repo/patches/tensorfold-ernie55.patch
TF=$PWD OUT=/var/tmp/ernie55/fatbins bash /path/to/this/repo/scripts/build_fatbins.sh   # ERNIE fatbins
# the other fatbins come from TensorFold's own build (zig build fatbins) or a release archive
zig build native -Dfatbins=/var/tmp/ernie55/fatbins -Doptimize=ReleaseSafe
cp /path/to/this/repo/scripts/cluster.env.example /path/to/this/repo/scripts/cluster.env   # fill in
bash /path/to/this/repo/scripts/deploy.sh   # copies the binary to every rank, starts, waits for /v1/models (~5 min)
python3 /path/to/this/repo/scripts/suite.py mytag   # ERNIE_URL / ERNIE_MODEL env
```

`deploy.sh` expects `bin/tensorfold-native` and `lib/libnccl.so.2` under `$ROOT` on every rank.

## Next work (highest value first)

1. **Concurrency, part 2:** drafts still run one stream at a time (`Lanes.draftFn` -> `Engine.block`); batch them like
   verify. Re-time `gb10_shared_costs` (the host learns row costs, but the prior matters). Target: vLLM's c4 114 tok/s.
2. **Decode c1:** profile shows FP8G GEMVs ~26 ms of a ~57 ms round, the bf16 router GEMV at ~70 GB/s, `hca_attn`
   108 us/layer, ~140 RoCE gathers a forward. Fuse same-input projections (q_down+kv_proj+compressor; fc1_latent +
   shared w1/w3), a faster router and decode HCA attention.
3. **Prefill:** `dsa_attn` (sparse MLA, CUDA cores, ~10%), NCCL all-reduce (~15-19%; overlap with compute),
   `mhc_pre` / `mhc_post`.
4. **Drafter acceptance:** compare per-position acceptance with vLLM's (75 / 45 / 25% at k=3) on the same workload;
   the confidence head is not used yet.
5. Missing: `libtfgrammar.so` (structured output and forced tool calls return HTTP 400), API key on non-loopback binds.

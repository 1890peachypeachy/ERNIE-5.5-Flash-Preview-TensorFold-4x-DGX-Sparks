# Working on this repo (humans and coding agents)

This repo is the shared workspace for running **ERNIE-5.5-Flash-Preview** on **TensorFold-native** (ashhart/TensorFold)
at TP=4 across four DGX Spark (GB10) nodes. The model port lives in `patches/tensorfold-ernie55.patch` against
TensorFold commit `549b55a129a45656a5eeef6c320ead37f833a73a`. Keep that relationship: the patch is the source of truth.

## Rules

1. **Measure before and after every change.** `scripts/suite.py <tag>` (correctness + speed) and the vLLM recipe's
   `scripts/bench/bench.py` (like-for-like with vLLM). Put the JSON in `results/` and a row in `docs/LOG.md`.
2. **Kernels go through a standalone harness first** (`kernels/harness/*_test.cu`): parity against the kernel it
   replaces (bit-equal, or rel-L2 and the effect on downstream choices, e.g. selected top-k sets), then timing. Only
   then wire it in.
3. **Correctness gates on every deploy:** greedy output of a cached prompt equals the cold one; concurrent greedy
   streams equal the same prompts run alone; long-context needle (33K and 66K) passes; tool call returns `tool_calls`.
4. **No PII or infrastructure detail** in commits: no IPs, hostnames, user names, tokens, home paths. Cluster specifics
   live in `scripts/cluster.env` (git-ignored; template `scripts/cluster.env.example`).
5. Regenerate the patch from the working tree, never hand-edit it:
   `cd <tensorfold checkout> && export GIT_INDEX_FILE=/tmp/idx && git read-tree HEAD && git add -A zig && git diff --cached --binary HEAD -- zig > patches/tensorfold-ernie55.patch; unset GIT_INDEX_FILE`
   and check it applies to a clean `549b55a` worktree (`git apply --check`).
6. Fatbins: rebuild with `scripts/build_fatbins.sh` (TensorFold's own nvcc flags; `ops.cu` keeps strict IEEE flags).

## Known pitfalls (each cost time once)

- nsys's default graph tracing hides kernels inside CUDA graphs: profile with `--cuda-graph-trace=node`, or decode
  looks 17% busy when it is ~97% busy.
- `pkill -f <pattern>` over ssh kills the ssh session when the pattern appears in the remote command line: match the
  binary path (`^$ROOT/bin/tensorfold-native`), and wait until no rank holds the binary before copying ("Text file busy").
- Zig 0.17 narrows `@min(x, <comptime constant>)` to a tiny int type: annotate (`var n: usize = @min(...)`).
- The rank-0 log prints one `done chatcmpl ...` line a request (rounds, accepted/drafted, ttft, tok/s): read it before
  trusting a client-side number, and count `POST /v1` lines to make sure no other client shared the run.

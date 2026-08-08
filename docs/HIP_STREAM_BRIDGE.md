# HIP stream bridge (PR-A phase 1)

**Branch:** `exp/hip-stream-bridge`  
**Status:** Phase 1 API landed — **host** `hipStreamSynchronize` then replay  
**Consumers:** lemon-mlx-engine OWN_RMSNORM / dual-queue engines  
**Contract:** lemonade-sdk `P13_STREAM_BRIDGE_PR.md`

## API

| Symbol | Role |
|--------|------|
| `rl_feature_bits()` | bit `RL_FEATURE_HIP_STREAM_WAIT` (1) if this build exports wait APIs |
| `rl_gpu_wait_hip_stream(void* hip_stream)` | host-join HIP stream (null = no-op) |
| `rl_pm4_replay_after_hip_stream(ib, hip_stream)` | wait stream + `rl_pm4_replay` |

Errors: `RL_ERR_HIP` (-8) if libamdhip64 / symbol missing or HIP status ≠ 0.

## Phase 1 honesty

This is **ABI + centralization**, not a gen-t/s win by itself:

- Host still joins producers (same cost as lemon-mlx `PRE_SYNC=stream/force`).
- Product HIP same-stream RMSNorm does **not** host-join; we still do.

## Phase 2 (required for PRE tax removal)

Device-side wait: enqueue dependency on HIP stream completion **without** blocking the CPU for the whole producer interval (HSA signal / queue wait packet / same-queue submit). See lemon-mlx P13 options S/E/Q.

## Build / install (gfx1150 host)

```bash
cd /home/antmi/redline
export PATH=/opt/rocm/core/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
cargo build -p redline-capi --release
mkdir -p /tmp/redline-warpfront-target/release
cp -a target/release/libredline_dispatch.so target/release/libredline_dispatch.a \
  /tmp/redline-warpfront-target/release/
nm -D /tmp/redline-warpfront-target/release/libredline_dispatch.so | grep rl_pm4_replay_after
```

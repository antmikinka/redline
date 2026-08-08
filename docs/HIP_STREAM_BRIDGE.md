# HIP stream bridge (PR-A phase 1 + phase 2)

**Branch:** `exp/hip-stream-bridge` on **https://github.com/antmikinka/redline** (pwilkin base)  
**Consumers:** lemon-mlx-engine OWN_RMSNORM  
**Contract:** lemonade-sdk `P13_STREAM_BRIDGE_PR.md`

## API

| Symbol | Phase | Role |
|--------|-------|------|
| `rl_feature_bits()` | 1+2 | `PRESENT\|WAIT\|PHASE2` (never equals bare `rl_abi_version`) |
| `rl_gpu_wait_hip_stream` | 1 | host `hipStreamSynchronize` |
| `rl_pm4_replay_after_hip_stream` | 1 | sync stream + replay |
| `rl_pm4_replay_after_hip_stream_phase2` | **2** | `WriteValue32` milestone + host poll fence + replay (**no** StreamSynchronize) |
| `rl_pm4_submit` / `rl_pm4_wait` | 2 | split submit vs completion wait |

Errors: `RL_ERR_HIP` (-8).

## Phase 1 honesty

Host still joins producers via StreamSynchronize — same class of tax as lemon-mlx PRE.

## Phase 2 honesty (current) — **not default for gen**

- Uses `hipStreamWriteValue32` + **host DtoH poll** of fence, then replay.
- Measured **slower** than phase1 on gfx1150 (~13 ms vs ~2.3 ms host for n=31) — poll is a bad join.
- lemon-mlx uses phase2 only if `MLX_REDLINE_PHASE2=1`; default stays phase1.
- `rl_pm4_submit` / `rl_pm4_wait` available for 2b async work.

## Phase 2b (next — true host free)

1. PM4 `WAIT_REG_MEM` prefix on HSA queue (no host poll)  
2. Consumer fence + `hipStreamWaitValue32` on product stream  
3. Host returns without `wait_signal` (async OWN_RMSNORM)

## Build / install

```bash
cd /home/antmi/redline
git checkout exp/hip-stream-bridge
export PATH=/opt/rocm/core/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
cargo build -p redline-capi --release
cp -a target/release/libredline_dispatch.so /tmp/redline-warpfront-target/release/
```

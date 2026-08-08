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
| `rl_pm4_replay_after_hip_stream_phase2` | **2b** | `WriteValue32` + PM4 `WAIT_REG_MEM` prefix + retained replay (**no** StreamSynchronize) |
| `rl_pm4_submit` / `rl_pm4_wait` | 2 | split submit vs completion wait |

Errors: `RL_ERR_HIP` (-8).

## Phase 1 honesty

Host still joins producers via StreamSynchronize — same class of tax as lemon-mlx PRE.

## Phase 2 / 2b honesty — **not default for gen**

- **2b (current default of the phase2 symbol):** `hipStreamWriteValue32` + ROCr-executable PM4 `WAIT_REG_MEM` prefix on the HSA queue, then retained IB. Host waits only on Redline completion (covers wait+kernel). No StreamSynchronize; no DtoH poll.
- **Host poll:** only if `REDLINE_PHASE2_HOST_POLL=1` (known **slower** than phase1 on gfx1150 ~13 ms vs ~2.3 ms for n=31). Not product default.
- **Fallback:** if ROCr wait-IB init / prefix submit fails → phase1 StreamSynchronize.
- lemon-mlx uses phase2 only if `MLX_REDLINE_PHASE2=1`; default stays phase1.
- `rl_pm4_submit` / `rl_pm4_wait` for async split; consumer `hipStreamWaitValue32` still follow-on.

## Phase 2b remaining (async + consumer)

1. ~~PM4 `WAIT_REG_MEM` prefix on HSA queue (no host poll)~~ **landed in symbol**  
2. Consumer fence + `hipStreamWaitValue32` on product stream  
3. Host returns without `wait_signal` (async OWN_RMSNORM via submit/wait)

## Build / install

```bash
cd /home/antmi/redline
git checkout exp/hip-stream-bridge
export PATH=/opt/rocm/core/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
cargo build -p redline-capi --release
cp -a target/release/libredline_dispatch.so /tmp/redline-warpfront-target/release/
```

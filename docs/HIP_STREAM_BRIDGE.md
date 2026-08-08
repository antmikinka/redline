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
| `rl_pm4_replay_after_hip_stream_phase2` | **2b** | `WriteValue32` + PM4 `WAIT_REG_MEM` prefix + retained replay (**no** StreamSynchronize; host still waits Redline) |
| `rl_pm4_submit_after_hip_stream_phase2` | **2b async** | same WAIT prefix + submit + WRITE_DATA consumer fence; **no** host `wait_signal` |
| `rl_gpu_consumer_wait_hip_stream` | **2b async** | `hipStreamWaitValue32` on consumer fence (product stream) |
| `rl_pm4_submit` / `rl_pm4_wait` | 2 | split submit vs completion wait |

Errors: `RL_ERR_HIP` (-8).

## Phase 1 honesty

Host still joins producers via StreamSynchronize — same class of tax as lemon-mlx PRE.

## Phase 2 / 2b honesty — **not default for gen**

- **2b sync (`rl_pm4_replay_after_hip_stream_phase2`):** `hipStreamWriteValue32` + ROCr-executable PM4 `WAIT_REG_MEM` prefix on the HSA queue, then retained IB. Host waits only on Redline completion (covers wait+kernel). No StreamSynchronize; no DtoH poll.
- **2b async (`rl_pm4_submit_after_hip_stream_phase2`):** WAIT prefix + doorbell + PM4 `WRITE_DATA` consumer fence (double-buffered IBs). Host returns without `wait_signal`. Product: `rl_gpu_consumer_wait_hip_stream`. **Must** `rl_pm4_wait` before IB reuse / `set_kernargs`.
- **Host poll:** only if `REDLINE_PHASE2_HOST_POLL=1` (known **slower** than phase1 on gfx1150 ~13 ms vs ~2.3 ms for n=31). Not product default.
- **Fallback:** if ROCr wait-IB init / prefix submit fails → phase1 StreamSynchronize.
- lemon-mlx uses phase2 only if `MLX_REDLINE_PHASE2=1`; default stays phase1.

## Phase 2b remaining

1. ~~PM4 `WAIT_REG_MEM` prefix on HSA queue (no host poll)~~ **landed**  
2. ~~Consumer fence + `hipStreamWaitValue32`~~ **landed** (`WRITE_DATA` + `rl_gpu_consumer_wait_hip_stream`)  
3. ~~Host returns without `wait_signal`~~ **landed** (`rl_pm4_submit_after_hip_stream_phase2`)  
4. **Fix (20260808):** WAIT_REG_MEM `mem_space` is GFX9+ bits`[5:4]` (not SI-era bit 8). Wrong encoding hung CP 5s → OWN fail-open (no `phase2-used`).  
5. lemon-mlx remeasure B1 phase2 for real `phase2-used` + honest t/s (flags still opt-in; no ≥2% claim without measure)

## When phase 2b (or any bridge work) is done — **commit + push required**

Remote is **your fork** (not warpfront):

```bash
cd /home/antmi/redline
git remote -v   # origin MUST be https://github.com/antmikinka/redline.git
git checkout exp/hip-stream-bridge
git add <paths>
git commit -m "feat(capi): <what landed>"
git push origin exp/hip-stream-bridge
# never --force
```

If lemon-mlx wire changed, also commit+push `exp/redline-kernel-launch` on lemonade-sdk/lemon-mlx-engine.

## Build / install

```bash
cd /home/antmi/redline
git checkout exp/hip-stream-bridge
export PATH=/opt/rocm/core/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
cargo build -p redline-capi --release
cp -a target/release/libredline_dispatch.so /tmp/redline-warpfront-target/release/
```

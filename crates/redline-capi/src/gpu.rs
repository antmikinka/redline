// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Real-GPU retained-PM4 replay over the C ABI.
//!
//! This is the fast path (a single retained PM4 indirect buffer, replayed
//! with one doorbell — `SingleQueuePm4Ib`) exposed for an inference engine to
//! drive with its *own* kernels and kernargs on gfx10, gfx11, and gfx12:
//!
//! 1. `rl_gpu_new(ordinal)` — bind a ROCr GPU (honours `ROCR_VISIBLE_DEVICES`).
//! 2. `rl_gpu_load_module_radiowave(...)` — load a code object plus its hashed
//!    Radiowave manifest. Raw [`rl_gpu_load_module`] remains fail-closed.
//! 3. `rl_pm4_builder_new(gpu)`, then `rl_pm4_dispatch(...)` per launch with the
//!    engine's kernarg segment bytes (which carry its device pointers), and
//!    `rl_pm4_wait_rmw(...)` to serialize with the minimal cache boundary
//!    certified for the next consumer. Raw [`rl_pm4_wait_idle`] remains the
//!    generic same-agent fallback.
//! 4. `rl_pm4_finalize(gpu, builder)` — lower to one retained PM4 IB.
//! 5. `rl_pm4_replay(ib)` — submit + wait; call it every token.
//!
//! The engine supplies each kernel's **full kernarg segment** (query the size
//! with `rl_module_kernarg_size`); Redline copies it verbatim, so hidden args
//! the engine already populated (grid dims, etc.) are preserved.
//!
//! Lifetime contract (C caller owns it): a module must outlive every IB built
//! from it; an IB must outlive its replays. Free with the matching `*_free`.

use std::ffi::{CStr, c_char};
use std::sync::Arc;

use radiowave::{CodeObjectCertification, MutableReadCache, SchedulerProfile};
use redline_dispatch::aql::{
    Executable, Gfx10Pm4CommandBuffer, Gfx12Pm4CommandBuffer, GpuDevice, GpuSelector,
    KernargBuffer, KernargPool, MultiQueuePm4Ib, QueuePolicy, Runtime, SingleQueuePm4Ib,
    load_symbols,
};

use crate::{
    RL_ERR_CERTIFICATION, RL_ERR_COMPILE, RL_ERR_HANDLE, RL_ERR_HIP, RL_ERR_NULL, RL_ERR_RECORD,
    RL_ERR_REPLAY, RL_ERR_UTF8, RL_OK,
};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// HIP runtime symbols via dlopen (no hard link).
struct HipFns {
    stream_synchronize: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
    stream_write_value32:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, u32, u32) -> i32,
    stream_wait_value32:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, u32, u32, u32) -> i32,
    malloc: unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize) -> i32,
    free: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
    memcpy: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const std::ffi::c_void,
        usize,
        i32,
    ) -> i32,
}

fn hip_fns() -> Option<&'static HipFns> {
    static CELL: OnceLock<Option<HipFns>> = OnceLock::new();
    CELL.get_or_init(|| {
        let lib = unsafe {
            libloading::Library::new("libamdhip64.so")
                .or_else(|_| libloading::Library::new("libamdhip64.so.7"))
                .or_else(|_| libloading::Library::new("/opt/rocm/core/lib/libamdhip64.so"))
        }
        .ok()?;
        let lib = Box::leak(Box::new(lib));
        unsafe {
            Some(HipFns {
                stream_synchronize: *lib.get(b"hipStreamSynchronize\0").ok()?,
                stream_write_value32: *lib.get(b"hipStreamWriteValue32\0").ok()?,
                stream_wait_value32: *lib.get(b"hipStreamWaitValue32\0").ok()?,
                malloc: *lib.get(b"hipMalloc\0").ok()?,
                free: *lib.get(b"hipFree\0").ok()?,
                memcpy: *lib.get(b"hipMemcpy\0").ok()?,
            })
        }
    })
    .as_ref()
}

// hipMemcpyDeviceToHost
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
// hipStreamWaitValueGte
const HIP_STREAM_WAIT_VALUE_GTE: u32 = 0x0;
// hipStreamWaitValueEq  
const HIP_STREAM_WAIT_VALUE_EQ: u32 = 0x1;

static PIPELINE_SEQ: AtomicU32 = AtomicU32::new(1);
/// Last seq written by WRITE_DATA consumer fence (for hipStreamWaitValue32).
static LAST_CONSUMER_SEQ: AtomicU32 = AtomicU32::new(0);
/// Device memory: [prod u32 @ +0][cons u32 @ +4].
static PIPELINE_FENCE: OnceLock<u64> = OnceLock::new();

/// Process-lifetime ROCr-executable PM4 IB slot (WAIT_REG_MEM or WRITE_DATA).
struct Phase2bPm4Ib {
    addr: *mut std::ffi::c_void,
    bytes: *mut u8,
    len: usize,
    dwords: u32,
}

// SAFETY: rewritten under PHASE2B_LOCK; pointees are process-lifetime (forgotten).
unsafe impl Send for Phase2bPm4Ib {}
unsafe impl Sync for Phase2bPm4Ib {}

/// Double-buffered wait + done IBs so async submit can return without holding
/// the rewrite lock across GPU wait (next call uses the other slot).
struct Phase2bPair {
    wait: Phase2bPm4Ib,
    done: Phase2bPm4Ib,
}

static PHASE2B_SLOTS: OnceLock<Option<[Phase2bPair; 2]>> = OnceLock::new();
static PHASE2B_TURN: AtomicU32 = AtomicU32::new(0);
static PHASE2B_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn pipeline_fence_addr() -> Option<u64> {
    let hip = hip_fns()?;
    Some(*PIPELINE_FENCE.get_or_init(|| {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let st = unsafe { (hip.malloc)(&mut p, 8) };
        if st != 0 || p.is_null() {
            return 0;
        }
        // zero prod + cons
        let z = [0u8; 8];
        unsafe {
            (hip.memcpy)(p, z.as_ptr().cast(), 8, 1); // HostToDevice = 1
        }
        p as u64
    }))
    .filter(|a| *a != 0)
}

fn consumer_fence_addr() -> Option<u64> {
    pipeline_fence_addr().map(|a| a + 4)
}

/// Build a tiny PM4 IB: WAIT_REG_MEM equal on device u32.
fn pm4_wait_reg_mem_eq(addr: u64, value: u32) -> Vec<u32> {
    const PACKET3_WAIT_REG_MEM: u32 = 0x3c;
    // equal(3) | memory space(1<<8)
    let function = 3u32 | (1 << 8);
    vec![
        (3u32 << 30) | (5 << 16) | (PACKET3_WAIT_REG_MEM << 8),
        function,
        addr as u32,
        (addr >> 32) as u32,
        value,
        u32::MAX,
        4, // poll interval
    ]
}

/// Build a tiny PM4 IB: WRITE_DATA of one u32 to device memory (consumer fence).
///
/// dst_sel=MEMORY (5), wr_confirm=1 — visible to hipStreamWaitValue32 after
/// the retained kernel IB (AQL barrier order).
fn pm4_write_data_u32(addr: u64, value: u32) -> Vec<u32> {
    const PACKET3_WRITE_DATA: u32 = 0x37;
    // dst_sel MEMORY (5<<8) | wr_confirm (1<<20)
    let control = (5u32 << 8) | (1 << 20);
    vec![
        (3u32 << 30) | (3 << 16) | (PACKET3_WRITE_DATA << 8),
        control,
        addr as u32,
        (addr >> 32) as u32,
        value,
    ]
}

fn alloc_phase2b_pm4_ib(pool: &KernargPool, dwords: &[u32]) -> Option<Phase2bPm4Ib> {
    let byte_len = dwords.len() * 4;
    let mut ib = pool.allocate_executable_bytes(byte_len).ok()?;
    let bytes: Vec<u8> = dwords.iter().flat_map(|d| d.to_le_bytes()).collect();
    ib.write_exact(&bytes).ok()?;
    let state = Phase2bPm4Ib {
        addr: ib.address(),
        bytes: ib.address().cast::<u8>(),
        len: byte_len,
        dwords: dwords.len() as u32,
    };
    std::mem::forget(ib);
    Some(state)
}

/// Double-buffered WAIT_REG_MEM + WRITE_DATA IBs (process lifetime).
fn phase2b_slots() -> Option<&'static [Phase2bPair; 2]> {
    PHASE2B_SLOTS
        .get_or_init(|| {
            let runtime = Runtime::initialize(load_symbols().ok()?).ok()?;
            let device = runtime.select_gpu(GpuSelector::Ordinal(0)).ok()?;
            let pool = KernargPool::discover(&device).ok()?;
            let wait_dwords = pm4_wait_reg_mem_eq(0, 0);
            let done_dwords = pm4_write_data_u32(0, 0);
            let mut pairs = Vec::with_capacity(2);
            for _ in 0..2 {
                let wait = alloc_phase2b_pm4_ib(&pool, &wait_dwords)?;
                let done = alloc_phase2b_pm4_ib(&pool, &done_dwords)?;
                pairs.push(Phase2bPair { wait, done });
            }
            let arr: [Phase2bPair; 2] = pairs.try_into().ok()?;
            // Keep pool/device/runtime mappings process-lifetime.
            std::mem::forget(pool);
            std::mem::forget(device);
            std::mem::forget(runtime);
            Some(arr)
        })
        .as_ref()
}

/// Next double-buffer slot (exclusive under PHASE2B_LOCK while rewritten).
fn phase2b_take_slot() -> Option<&'static Phase2bPair> {
    let slots = phase2b_slots()?;
    let i = (PHASE2B_TURN.fetch_add(1, Ordering::Relaxed) & 1) as usize;
    Some(&slots[i])
}

/// Patch executable IB body (caller holds PHASE2B_LOCK).
unsafe fn rewrite_pm4_ib(ib: &Phase2bPm4Ib, dwords: &[u32]) -> bool {
    let bytes: Vec<u8> = dwords.iter().flat_map(|d| d.to_le_bytes()).collect();
    if bytes.len() != ib.len {
        return false;
    }
    // SAFETY: exclusive under lock; process-lifetime executable IB.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ib.bytes, ib.len);
    }
    true
}

fn next_pipeline_seq() -> u32 {
    let seq = PIPELINE_SEQ.fetch_add(1, Ordering::Relaxed);
    if seq == 0 {
        // skip 0 so eq never matches cleared memory accidentally
        PIPELINE_SEQ.store(1, Ordering::Relaxed);
        1
    } else {
        seq
    }
}

/// Emergency host DtoH poll (known slower). Only if REDLINE_PHASE2_HOST_POLL=1.
fn phase2_host_poll_enabled() -> bool {
    match std::env::var("REDLINE_PHASE2_HOST_POLL") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"),
        Err(_) => false,
    }
}

fn phase2_host_poll_fence(hip: &HipFns, fence: u64, seq: u32) -> i32 {
    let mut spins = 0u32;
    loop {
        let mut host_val = 0u32;
        let rc = unsafe {
            (hip.memcpy)(
                (&mut host_val as *mut u32).cast(),
                fence as *const std::ffi::c_void,
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != 0 {
            return RL_ERR_HIP;
        }
        if host_val == seq {
            return RL_OK;
        }
        spins = spins.wrapping_add(1);
        if spins > 100_000_000 {
            return RL_ERR_HIP;
        }
        std::thread::yield_now();
    }
}

/// Wait until work previously submitted on `hip_stream` completes (host join).
///
/// **Phase 1:** `hipStreamSynchronize`. Returns `RL_OK` or `RL_ERR_HIP`.
///
/// # Safety
/// `hip_stream` is null (no-op) or a valid `hipStream_t` for the current device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_wait_hip_stream(hip_stream: *mut std::ffi::c_void) -> i32 {
    if hip_stream.is_null() {
        return RL_OK;
    }
    let Some(hip) = hip_fns() else {
        return RL_ERR_HIP;
    };
    let st = unsafe { (hip.stream_synchronize)(hip_stream) };
    if st != 0 {
        return RL_ERR_HIP;
    }
    RL_OK
}

/// Phase 1: host-join `hip_stream` then [`rl_pm4_replay`].
///
/// # Safety
/// Same as [`rl_pm4_replay`] for `ib`; `hip_stream` null or valid `hipStream_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay_after_hip_stream(
    ib: *mut RlPm4Ib,
    hip_stream: *mut std::ffi::c_void,
) -> i32 {
    let wait = unsafe { rl_gpu_wait_hip_stream(hip_stream) };
    if wait != RL_OK {
        return wait;
    }
    unsafe { rl_pm4_replay(ib) }
}

/// Phase 2 / 2b: order Redline after HIP producers **without** host
/// `hipStreamSynchronize`.
///
/// **Phase 2b (default when ROCr wait IB is available):**
/// 1. `hipStreamWriteValue32(stream, fence, seq)` — GPU milestone after producers
/// 2. PM4 `WAIT_REG_MEM` (equal) prefix on the HSA queue for that fence/seq
/// 3. retained IB submit + host wait on Redline completion only
///
/// Host is **not** spinning on DtoH and **not** in HIP StreamSynchronize while
/// producers run; the CP waits on device memory. Call wall-time still includes
/// producer tail once WAIT unblocks (GPU critical path) but removes host join tax.
///
/// **Fallback order:** phase2b → (opt-in) host DtoH poll if
/// `REDLINE_PHASE2_HOST_POLL=1` → phase1 StreamSynchronize. Host poll is known
/// slower and must not be product default.
///
/// # Safety
/// Same as [`rl_pm4_replay_after_hip_stream`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay_after_hip_stream_phase2(
    ib: *mut RlPm4Ib,
    hip_stream: *mut std::ffi::c_void,
) -> i32 {
    if hip_stream.is_null() {
        return unsafe { rl_pm4_replay(ib) };
    }
    let Some(hip) = hip_fns() else {
        return unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) };
    };
    let Some(fence) = pipeline_fence_addr() else {
        return unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) };
    };
    let Some(ib_ref) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let seq = next_pipeline_seq();

    // Milestone after all prior work on the product stream.
    let st = unsafe {
        (hip.stream_write_value32)(hip_stream, fence as *mut std::ffi::c_void, seq, 0)
    };
    if st != 0 {
        return unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) };
    }

    // Phase 2b: device-side WAIT_REG_MEM prefix then retained replay.
    // Hold lock only across rewrite+submit; wait under lock for the *sync*
    // path so a concurrent rewrite cannot clobber the in-flight single slot
    // (async path uses double-buffer + returns before wait).
    if let Some(slot) = phase2b_take_slot() {
        let dwords = pm4_wait_reg_mem_eq(fence, seq);
        let _guard = match PHASE2B_LOCK.lock() {
            Ok(g) => g,
            Err(_) => return unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) },
        };
        if unsafe { rewrite_pm4_ib(&slot.wait, &dwords) } {
            match unsafe {
                ib_ref
                    .ib
                    .replay_and_wait_with_pm4_ib_prefix(slot.wait.addr, slot.wait.dwords)
            } {
                Ok(()) => return RL_OK,
                Err(_) => {
                    // Fall through to host poll / phase1 after dropping lock.
                }
            }
        }
        drop(_guard);
    }

    // Opt-in host poll only (measured slower than phase1 — not default).
    if phase2_host_poll_enabled() {
        let poll = phase2_host_poll_fence(hip, fence, seq);
        if poll != RL_OK {
            return unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) };
        }
        return match unsafe { ib_ref.ib.replay_and_wait() } {
            Ok(()) => RL_OK,
            Err(_) => RL_ERR_REPLAY,
        };
    }

    // Default fallback: phase1 host StreamSynchronize + replay.
    unsafe { rl_pm4_replay_after_hip_stream(ib, hip_stream) }
}

/// Phase 2b **async**: WriteValue milestone + WAIT_REG_MEM prefix + retained
/// submit + WRITE_DATA consumer fence; **no** host `wait_signal`.
///
/// Host may return immediately. Product HIP consumers should call
/// [`rl_gpu_consumer_wait_hip_stream`] (hipStreamWaitValue32). Caller **must**
/// [`rl_pm4_wait`] before reusing the same IB for `set_kernargs` / next submit.
///
/// Falls back to [`rl_pm4_replay_after_hip_stream_phase2`] (host wait) on error.
///
/// # Safety
/// Same as [`rl_pm4_replay_after_hip_stream_phase2`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_submit_after_hip_stream_phase2(
    ib: *mut RlPm4Ib,
    hip_stream: *mut std::ffi::c_void,
) -> i32 {
    if hip_stream.is_null() {
        return unsafe { rl_pm4_submit(ib) };
    }
    let Some(hip) = hip_fns() else {
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    };
    let Some(prod) = pipeline_fence_addr() else {
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    };
    let Some(cons) = consumer_fence_addr() else {
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    };
    let Some(ib_ref) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let Some(slot) = phase2b_take_slot() else {
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    };
    let seq = next_pipeline_seq();

    let st = unsafe {
        (hip.stream_write_value32)(hip_stream, prod as *mut std::ffi::c_void, seq, 0)
    };
    if st != 0 {
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    }

    let wait_dw = pm4_wait_reg_mem_eq(prod, seq);
    let done_dw = pm4_write_data_u32(cons, seq);
    let _guard = match PHASE2B_LOCK.lock() {
        Ok(g) => g,
        Err(_) => return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) },
    };
    if !unsafe { rewrite_pm4_ib(&slot.wait, &wait_dw) }
        || !unsafe { rewrite_pm4_ib(&slot.done, &done_dw) }
    {
        drop(_guard);
        return unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) };
    }
    match unsafe {
        ib_ref.ib.submit_with_pm4_ib_prefix_suffix(
            slot.wait.addr,
            slot.wait.dwords,
            slot.done.addr,
            slot.done.dwords,
        )
    } {
        Ok(()) => {
            // Seq the product stream should WaitValue32 for (WRITE_DATA after kernel).
            LAST_CONSUMER_SEQ.store(seq, Ordering::Release);
            RL_OK
        }
        Err(_) => {
            drop(_guard);
            unsafe { rl_pm4_replay_after_hip_stream_phase2(ib, hip_stream) }
        }
    }
}

/// Enqueue `hipStreamWaitValue32` (eq) on the last phase2b consumer fence so
/// product HIP work waits for Redline kernel + WRITE_DATA without host join.
///
/// Pair after [`rl_pm4_submit_after_hip_stream_phase2`]. No-op success if no
/// consumer seq has been published yet.
///
/// # Safety
/// `hip_stream` null or valid `hipStream_t` for the active device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_consumer_wait_hip_stream(
    hip_stream: *mut std::ffi::c_void,
) -> i32 {
    if hip_stream.is_null() {
        return RL_OK;
    }
    let seq = LAST_CONSUMER_SEQ.load(Ordering::Acquire);
    if seq == 0 {
        return RL_OK;
    }
    let Some(hip) = hip_fns() else {
        return RL_ERR_HIP;
    };
    let Some(cons) = consumer_fence_addr() else {
        return RL_ERR_HIP;
    };
    let st = unsafe {
        (hip.stream_wait_value32)(
            hip_stream,
            cons as *mut std::ffi::c_void,
            seq,
            HIP_STREAM_WAIT_VALUE_EQ,
            0,
        )
    };
    if st != 0 {
        return RL_ERR_HIP;
    }
    RL_OK
}

/// Phase 2 async: submit retained IB without host wait. Pair with [`rl_pm4_wait`].
///
/// Does **not** establish HIP producer ordering by itself — use
/// [`rl_pm4_submit_after_hip_stream_phase2`] or
/// [`rl_pm4_replay_after_hip_stream_phase2`] for ordered OWN_RMSNORM.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_submit(ib: *mut RlPm4Ib) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    match unsafe { ib.ib.submit_only() } {
        Ok(()) => RL_OK,
        Err(_) => RL_ERR_REPLAY,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_wait(ib: *mut RlPm4Ib) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    match unsafe { ib.ib.wait_only() } {
        Ok(()) => RL_OK,
        Err(_) => RL_ERR_REPLAY,
    }
}

/// A GPU binding: ROCr runtime + selected device + kernarg pool.
pub struct RlGpu {
    pub(crate) device: GpuDevice,
    pub(crate) pool: KernargPool,
    _runtime: Runtime,
}

/// A loaded code object; kernels are looked up from it by symbol.
pub struct RlModule {
    pub(crate) executable: Executable,
    device_agent: u64,
    certification: Option<CodeObjectCertification>,
}

/// Mutable-resource read-cache classification from a verified Radiowave
/// manifest. Missing modules/kernels return `RlMutableScalarOrUnknown` and therefore
/// retain the fail-closed scalar-cache invalidation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlMutableReadCache {
    RlMutableScalarOrUnknown = 0,
    RlMutableVmemOnly = 1,
}

/// Scheduler profile recorded by a verified Radiowave manifest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlSchedulerProfile {
    RlSchedulerUnknown = 0,
    RlSchedulerDefault = 1,
    RlSchedulerMaxIlp = 2,
    RlSchedulerIterativeIlp = 3,
    RlSchedulerMemoryClause = 4,
    RlSchedulerPipelineIlp = 5,
}

/// Public-queue fan-out policy for independent retained PM4 work.
///
/// `RlQueueAuto` uses the architecture table certified by the #6409 sweeps:
/// gfx1100 and gfx12 use at most two lanes, other gfx11 devices use at most
/// four, and unmeasured families retain one. The resolved count never exceeds
/// `independent_width`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlQueuePolicy {
    RlQueueAuto = 0,
    RlQueueOne = 1,
    RlQueueTwo = 2,
    RlQueueThree = 3,
    RlQueueFour = 4,
}

impl From<RlQueuePolicy> for QueuePolicy {
    fn from(value: RlQueuePolicy) -> Self {
        match value {
            RlQueuePolicy::RlQueueAuto => Self::Auto,
            RlQueuePolicy::RlQueueOne => Self::One,
            RlQueuePolicy::RlQueueTwo => Self::Two,
            RlQueuePolicy::RlQueueThree => Self::Three,
            RlQueuePolicy::RlQueueFour => Self::Four,
        }
    }
}

/// A builder accumulating dispatches into one PM4 command buffer.
pub struct RlPm4Builder {
    family: Pm4Family,
    device_agent: u64,
    cmd: Pm4Commands,
    kernargs: Vec<KernargBuffer>,
    modules: Vec<Executable>,
    pool: KernargPool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4Family {
    Gfx10,
    Gfx11,
    Gfx12,
}

impl Pm4Family {
    fn from_name(name: &str) -> Option<Self> {
        if name.starts_with("gfx10") {
            Some(Self::Gfx10)
        } else if name.starts_with("gfx11") {
            Some(Self::Gfx11)
        } else if name.starts_with("gfx12") {
            Some(Self::Gfx12)
        } else {
            None
        }
    }
}

enum Pm4Commands {
    Legacy(Gfx10Pm4CommandBuffer),
    Gfx12(Gfx12Pm4CommandBuffer),
}

impl Pm4Commands {
    fn stateful_with_leading_acquire(family: Pm4Family) -> Self {
        match family {
            Pm4Family::Gfx10 | Pm4Family::Gfx11 => {
                let mut commands = Gfx10Pm4CommandBuffer::new_stateful();
                // Leading same-agent acquire: invalidate scalar/vector read
                // caches at the start of every replay, so in-place kernarg
                // mutation (rl_pm4_ib_set_kernargs) between replays is observed
                // fresh instead of read stale from the scalar cache. Required
                // for the per-token decode update pattern.
                commands.acquire_system();
                Self::Legacy(commands)
            }
            Pm4Family::Gfx12 => {
                let mut commands = Gfx12Pm4CommandBuffer::new_stateful();
                // Leading same-agent acquire: invalidate scalar/vector read
                // caches at the start of every replay, so in-place kernarg
                // mutation (rl_pm4_ib_set_kernargs) between replays is observed
                // fresh instead of read stale from the scalar cache. Required
                // for the per-token decode update pattern.
                commands.acquire_inter_node_gfx12();
                Self::Gfx12(commands)
            }
        }
    }

    fn dispatch(
        &mut self,
        kernel: &redline_dispatch::aql::Kernel,
        geometry: redline_dispatch::aql::LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut std::ffi::c_void,
    ) -> Result<(), ()> {
        match self {
            Self::Legacy(commands) => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|_| ()),
            Self::Gfx12(commands) => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|_| ()),
        }
    }

    fn dependency_rmw_same_agent(&mut self) {
        match self {
            Self::Legacy(commands) => commands.dependency_rmw_same_agent(),
            Self::Gfx12(commands) => commands.dependency_rmw_same_agent_gfx12(),
        }
    }

    fn dependency_rmw_vmem(&mut self) {
        match self {
            Self::Legacy(commands) => commands.dependency_rmw_vmem(),
            Self::Gfx12(commands) => commands.dependency_rmw_hip_llvm_vmem_gfx12(),
        }
    }
}

/// A finalized, retained PM4 indirect buffer, replayable end to end.
pub struct RlPm4Ib {
    ib: SingleQueuePm4Ib,
    kernargs: Vec<KernargBuffer>,
    _modules: Vec<Executable>,
}

/// Finalized retained PM4 indirect buffers, one per independent public queue.
pub struct RlPm4MultiIb {
    ib: MultiQueuePm4Ib,
    kernargs: Vec<Vec<KernargBuffer>>,
    _modules: Vec<Executable>,
}

/// Bind ROCr GPU `device_ordinal` (of the `ROCR_VISIBLE_DEVICES` set). Returns
/// null on failure. Free with [`rl_gpu_free`].
#[unsafe(no_mangle)]
pub extern "C" fn rl_gpu_new(device_ordinal: i32) -> *mut RlGpu {
    let build = || -> Option<RlGpu> {
        let runtime = Runtime::initialize(load_symbols().ok()?).ok()?;
        let ordinal = usize::try_from(device_ordinal).ok()?;
        let device = runtime.select_gpu(GpuSelector::Ordinal(ordinal)).ok()?;
        let pool = KernargPool::discover(&device).ok()?;
        Some(RlGpu {
            device,
            pool,
            _runtime: runtime,
        })
    };
    match build() {
        Some(gpu) => Box::into_raw(Box::new(gpu)),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `gpu` must be a pointer from [`rl_gpu_new`], or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_free(gpu: *mut RlGpu) {
    if !gpu.is_null() {
        drop(unsafe { Box::from_raw(gpu) });
    }
}

/// Resolve a PM4 queue policy for this GPU and an independent antichain width.
/// Returns zero when `gpu` is null. Every valid GPU returns at least one lane.
///
/// # Safety
/// `gpu` must be a live pointer from [`rl_gpu_new`], or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_pm4_queue_count(
    gpu: *const RlGpu,
    policy: RlQueuePolicy,
    independent_width: usize,
) -> usize {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return 0;
    };
    QueuePolicy::from(policy).resolve(gpu.device.name(), independent_width)
}

/// Load a code object (HSACO bytes). Writes the module pointer to `out`.
///
/// # Safety
/// `gpu`/`out` valid or null; `code` valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_load_module(
    gpu: *const RlGpu,
    code: *const u8,
    len: usize,
    out: *mut *mut RlModule,
) -> i32 {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if code.is_null() || out.is_null() {
        return RL_ERR_NULL;
    }
    let bytes: Arc<[u8]> = unsafe { std::slice::from_raw_parts(code, len) }.into();
    match Executable::load(&gpu.device, bytes) {
        Ok(executable) => {
            unsafe {
                *out = Box::into_raw(Box::new(RlModule {
                    executable,
                    device_agent: gpu.device.agent_handle(),
                    certification: None,
                }))
            };
            RL_OK
        }
        Err(_) => RL_ERR_COMPILE,
    }
}

/// Load a code object only after verifying its Radiowave JSON manifest binds
/// to these exact bytes and contains code-object inspection evidence.
///
/// # Safety
/// `gpu`/`out` valid or null; `code` and `manifest` valid for their stated
/// lengths. The manifest need not be NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_gpu_load_module_radiowave(
    gpu: *const RlGpu,
    code: *const u8,
    len: usize,
    manifest: *const u8,
    manifest_len: usize,
    out: *mut *mut RlModule,
) -> i32 {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if code.is_null() || manifest.is_null() || out.is_null() {
        return RL_ERR_NULL;
    }
    let code = unsafe { std::slice::from_raw_parts(code, len) };
    let manifest = unsafe { std::slice::from_raw_parts(manifest, manifest_len) };
    let Ok(manifest) = std::str::from_utf8(manifest) else {
        return RL_ERR_UTF8;
    };
    let Ok(certification) = CodeObjectCertification::from_json(code, manifest) else {
        return RL_ERR_CERTIFICATION;
    };
    let bytes: Arc<[u8]> = code.into();
    match Executable::load(&gpu.device, bytes) {
        Ok(executable) => {
            unsafe {
                *out = Box::into_raw(Box::new(RlModule {
                    executable,
                    device_agent: gpu.device.agent_handle(),
                    certification: Some(certification),
                }))
            };
            RL_OK
        }
        Err(_) => RL_ERR_COMPILE,
    }
}

/// # Safety
/// `module` from [`rl_gpu_load_module`], or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_free(module: *mut RlModule) {
    if !module.is_null() {
        drop(unsafe { Box::from_raw(module) });
    }
}

/// The kernarg segment size (bytes) the engine must supply for `symbol`, or -1.
///
/// # Safety
/// `module`/`symbol` valid or null; `symbol` NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_kernarg_size(
    module: *const RlModule,
    symbol: *const c_char,
) -> i64 {
    let Some(module) = (unsafe { module.as_ref() }) else {
        return -1;
    };
    if symbol.is_null() {
        return -1;
    }
    let Ok(sym) = (unsafe { CStr::from_ptr(symbol) }).to_str() else {
        return -1;
    };
    match module.executable.kernel(sym) {
        Ok(kernel) => i64::from(kernel.metadata().kernarg_segment_size),
        Err(_) => -1,
    }
}

/// Whether `module` was loaded with and verified against a Radiowave manifest.
///
/// # Safety
/// `module` from a module-load function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_radiowave_certified(module: *const RlModule) -> bool {
    unsafe { module.as_ref() }.is_some_and(|module| module.certification.is_some())
}

/// Scheduler profile in the verified manifest, or `RlSchedulerUnknown`.
///
/// # Safety
/// `module` from a module-load function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_scheduler_profile(
    module: *const RlModule,
) -> RlSchedulerProfile {
    let Some(profile) = (unsafe { module.as_ref() })
        .and_then(|module| module.certification.as_ref())
        .map(|certification| certification.manifest().scheduler_profile)
    else {
        return RlSchedulerProfile::RlSchedulerUnknown;
    };
    match profile {
        SchedulerProfile::Default => RlSchedulerProfile::RlSchedulerDefault,
        SchedulerProfile::MaxIlp => RlSchedulerProfile::RlSchedulerMaxIlp,
        SchedulerProfile::IterativeIlp => RlSchedulerProfile::RlSchedulerIterativeIlp,
        SchedulerProfile::MemoryClause => RlSchedulerProfile::RlSchedulerMemoryClause,
        SchedulerProfile::PipelineIlp => RlSchedulerProfile::RlSchedulerPipelineIlp,
    }
}

/// Wavefront width in the verified manifest, or zero for an uncertified/null
/// module.
///
/// # Safety
/// `module` from a module-load function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_wavefront_size(module: *const RlModule) -> u32 {
    unsafe { module.as_ref() }
        .and_then(|module| module.certification.as_ref())
        .map_or(0, |certification| {
            certification.manifest().wavefront.width()
        })
}

fn module_read_cache(module: &RlModule, symbol: &str) -> MutableReadCache {
    module
        .certification
        .as_ref()
        .map_or(MutableReadCache::ScalarOrUnknown, |certification| {
            certification.mutable_read_cache(symbol)
        })
}

/// Return the verified mutable-resource cache class for `symbol`. Unknown raw
/// modules, missing symbols, invalid UTF-8, and null pointers fail closed.
///
/// # Safety
/// `module`/`symbol` valid or null; `symbol` NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_module_kernel_read_cache(
    module: *const RlModule,
    symbol: *const c_char,
) -> RlMutableReadCache {
    let Some(module) = (unsafe { module.as_ref() }) else {
        return RlMutableReadCache::RlMutableScalarOrUnknown;
    };
    if symbol.is_null() {
        return RlMutableReadCache::RlMutableScalarOrUnknown;
    }
    let Ok(symbol) = (unsafe { CStr::from_ptr(symbol) }).to_str() else {
        return RlMutableReadCache::RlMutableScalarOrUnknown;
    };
    match module_read_cache(module, symbol) {
        MutableReadCache::ScalarOrUnknown => RlMutableReadCache::RlMutableScalarOrUnknown,
        MutableReadCache::VmemOnly => RlMutableReadCache::RlMutableVmemOnly,
    }
}

/// New PM4 builder bound to `gpu`. Free with [`rl_pm4_builder_free`] (or hand it
/// to [`rl_pm4_finalize`], which consumes it). Returns null on failure.
///
/// # Safety
/// `gpu` valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_builder_new(gpu: *const RlGpu) -> *mut RlPm4Builder {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return std::ptr::null_mut();
    };
    let Some(family) = Pm4Family::from_name(gpu.device.name()) else {
        return std::ptr::null_mut();
    };
    // Match the certified Hipfire retained-tape policy: preserve SH-register
    // state within the IB and omit writes whose values have not changed.
    let cmd = Pm4Commands::stateful_with_leading_acquire(family);
    Box::into_raw(Box::new(RlPm4Builder {
        family,
        device_agent: gpu.device.agent_handle(),
        cmd,
        kernargs: Vec::new(),
        modules: Vec::new(),
        pool: gpu.pool.clone(),
    }))
}

/// Record one dispatch: `symbol` from `module`, `grid`/`block` in **workitems**
/// (grid = gridDim × blockDim), `dynamic_group_bytes`, and `kernarg`/`kernarg_len`
/// = the engine's kernarg segment (zero-padded / truncated to the kernel's size).
///
/// # Safety
/// Pointers valid or null; `symbol` NUL-terminated; `kernarg` valid for
/// `kernarg_len`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rl_pm4_dispatch(
    builder: *mut RlPm4Builder,
    module: *const RlModule,
    symbol: *const c_char,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    dynamic_group_bytes: u32,
    kernarg: *const u8,
    kernarg_len: usize,
) -> i32 {
    let Some(builder) = (unsafe { builder.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let Some(module) = (unsafe { module.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if builder.device_agent != module.device_agent {
        return RL_ERR_HANDLE;
    }
    if symbol.is_null() {
        return RL_ERR_NULL;
    }
    let Ok(sym) = (unsafe { CStr::from_ptr(symbol) }).to_str() else {
        return RL_ERR_UTF8;
    };
    let Ok(kernel) = module.executable.kernel(sym) else {
        return RL_ERR_RECORD;
    };
    let Ok(block) = u16::try_from(block_x)
        .and_then(|x| Ok([x, u16::try_from(block_y)?, u16::try_from(block_z)?]))
    else {
        return RL_ERR_RECORD;
    };
    let Ok(geometry) = redline_dispatch::aql::LaunchGeometry::new([grid_x, grid_y, grid_z], block)
    else {
        return RL_ERR_RECORD;
    };
    let Ok(mut karg) = builder.pool.allocate_for(kernel.metadata()) else {
        return RL_ERR_RECORD;
    };
    {
        let dst = karg.as_mut_bytes();
        dst.fill(0);
        if !kernarg.is_null() && kernarg_len > 0 {
            let src = unsafe { std::slice::from_raw_parts(kernarg, kernarg_len) };
            let n = src.len().min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }
    }
    if builder
        .cmd
        .dispatch(&kernel, geometry, dynamic_group_bytes, karg.address())
        .is_err()
    {
        return RL_ERR_RECORD;
    }
    builder.kernargs.push(karg);
    builder.modules.push(module.executable.clone()); // keep the code object loaded
    RL_OK
}

/// Insert a dependency boundary after the dispatches recorded so far: wait for
/// the writer to retire, then invalidate scalar/vector read caches so the next
/// dispatch sees its L2-committed output. Required for a non-atomic read-modify-
/// write chain (decode); still the minimal same-agent fence (L2/MALL stays coherent).
///
/// # Safety
/// `builder` valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_wait_idle(builder: *mut RlPm4Builder) {
    if let Some(builder) = unsafe { builder.as_mut() } {
        builder.cmd.dependency_rmw_same_agent();
    }
}

/// Insert a same-agent RMW dependency selected for the next consumer. A
/// verified VMEM-only Radiowave consumer omits the unrelated scalar-cache
/// invalidation; every raw, missing, or ambiguous consumer uses the generic
/// fail-closed boundary.
///
/// # Safety
/// `builder`/`module` valid or null; `consumer_symbol` NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_wait_rmw(
    builder: *mut RlPm4Builder,
    module: *const RlModule,
    consumer_symbol: *const c_char,
) -> i32 {
    let Some(builder) = (unsafe { builder.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let Some(module) = (unsafe { module.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if consumer_symbol.is_null() {
        return RL_ERR_NULL;
    }
    let Ok(consumer_symbol) = (unsafe { CStr::from_ptr(consumer_symbol) }).to_str() else {
        return RL_ERR_UTF8;
    };
    match module_read_cache(module, consumer_symbol) {
        MutableReadCache::VmemOnly => builder.cmd.dependency_rmw_vmem(),
        MutableReadCache::ScalarOrUnknown => builder.cmd.dependency_rmw_same_agent(),
    }
    RL_OK
}

/// # Safety
/// `builder` from [`rl_pm4_builder_new`] and NOT already finalized, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_builder_free(builder: *mut RlPm4Builder) {
    if !builder.is_null() {
        drop(unsafe { Box::from_raw(builder) });
    }
}

/// Lower the builder into a retained PM4 IB. **Consumes `builder`** (do not use
/// or free it after this). Writes the IB pointer to `out`.
///
/// # Safety
/// `gpu`/`out` valid or null; `builder` a live, un-finalized builder pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_finalize(
    gpu: *const RlGpu,
    builder: *mut RlPm4Builder,
    out: *mut *mut RlPm4Ib,
) -> i32 {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if builder.is_null() || out.is_null() {
        return RL_ERR_NULL;
    }
    let builder = unsafe { Box::from_raw(builder) };
    match finalize_ib(gpu, builder.family, &builder.cmd, false) {
        Ok(ib) => {
            let boxed = Box::new(RlPm4Ib {
                ib,
                kernargs: builder.kernargs,
                _modules: builder.modules,
            });
            unsafe { *out = Box::into_raw(boxed) };
            RL_OK
        }
        Err(()) => RL_ERR_COMPILE,
    }
}

/// Profiled form of [`rl_pm4_finalize`]. The resulting IB can be replayed with
/// [`rl_pm4_replay_profiled`] to obtain the GPU execution span in microseconds.
///
/// # Safety
/// Identical to [`rl_pm4_finalize`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_finalize_profiled(
    gpu: *const RlGpu,
    builder: *mut RlPm4Builder,
    out: *mut *mut RlPm4Ib,
) -> i32 {
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if builder.is_null() || out.is_null() {
        return RL_ERR_NULL;
    }
    let builder = unsafe { Box::from_raw(builder) };
    match finalize_ib(gpu, builder.family, &builder.cmd, true) {
        Ok(ib) => {
            let boxed = Box::new(RlPm4Ib {
                ib,
                kernargs: builder.kernargs,
                _modules: builder.modules,
            });
            unsafe { *out = Box::into_raw(boxed) };
            RL_OK
        }
        Err(()) => RL_ERR_COMPILE,
    }
}

/// Lower one non-empty builder per independent lane into retained PM4 IBs.
///
/// All builders must come from the supplied GPU. Builders are consumed once
/// validation succeeds, including when PM4 compilation subsequently fails.
///
/// # Safety
/// `gpu`/`builders`/`out` valid or null; `builders` points to `lane_count`
/// distinct, live, un-finalized builder pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_finalize_multi(
    gpu: *const RlGpu,
    builders: *const *mut RlPm4Builder,
    lane_count: usize,
    out: *mut *mut RlPm4MultiIb,
) -> i32 {
    unsafe { finalize_multi(gpu, builders, lane_count, out, false) }
}

/// Profiled form of [`rl_pm4_finalize_multi`]. Replay reports the GPU makespan
/// from the earliest lane start through the latest lane end.
///
/// # Safety
/// Identical to [`rl_pm4_finalize_multi`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_finalize_multi_profiled(
    gpu: *const RlGpu,
    builders: *const *mut RlPm4Builder,
    lane_count: usize,
    out: *mut *mut RlPm4MultiIb,
) -> i32 {
    unsafe { finalize_multi(gpu, builders, lane_count, out, true) }
}

unsafe fn finalize_multi(
    gpu: *const RlGpu,
    builders: *const *mut RlPm4Builder,
    lane_count: usize,
    out: *mut *mut RlPm4MultiIb,
    profiled: bool,
) -> i32 {
    if out.is_null() {
        return RL_ERR_NULL;
    }
    unsafe { *out = std::ptr::null_mut() };
    let Some(gpu) = (unsafe { gpu.as_ref() }) else {
        return RL_ERR_NULL;
    };
    if builders.is_null() {
        return RL_ERR_NULL;
    }
    if lane_count == 0 {
        return RL_ERR_RECORD;
    }
    let builders = unsafe { std::slice::from_raw_parts(builders, lane_count) };
    if builders.contains(&std::ptr::null_mut()) {
        return RL_ERR_NULL;
    }
    for (index, builder) in builders.iter().enumerate() {
        if builders[..index].contains(builder) {
            return RL_ERR_HANDLE;
        }
    }

    let first = unsafe { &*builders[0] };
    let family = first.family;
    let device_agent = gpu.device.agent_handle();
    for builder in builders {
        let builder = unsafe { &**builder };
        if builder.device_agent != device_agent
            || builder.family != family
            || builder.kernargs.is_empty()
        {
            return RL_ERR_RECORD;
        }
    }

    let builders = builders
        .iter()
        .map(|builder| unsafe { *Box::from_raw(*builder) })
        .collect::<Vec<_>>();
    match build_multi_ib(gpu, family, builders, profiled) {
        Ok(ib) => {
            unsafe { *out = Box::into_raw(Box::new(ib)) };
            RL_OK
        }
        Err(()) => RL_ERR_COMPILE,
    }
}

fn build_multi_ib(
    gpu: &RlGpu,
    family: Pm4Family,
    builders: Vec<RlPm4Builder>,
    profiled: bool,
) -> Result<RlPm4MultiIb, ()> {
    let mut legacy_commands = Vec::with_capacity(builders.len());
    let mut gfx12_commands = Vec::with_capacity(builders.len());
    let mut kernargs = Vec::with_capacity(builders.len());
    let mut modules = Vec::new();
    for builder in builders {
        let RlPm4Builder {
            cmd,
            kernargs: lane_kernargs,
            modules: lane_modules,
            ..
        } = builder;
        match cmd {
            Pm4Commands::Legacy(commands) => legacy_commands.push(commands),
            Pm4Commands::Gfx12(commands) => gfx12_commands.push(commands),
        }
        kernargs.push(lane_kernargs);
        modules.extend(lane_modules);
    }

    let ib = match (family, profiled) {
        (Pm4Family::Gfx10, false) => {
            MultiQueuePm4Ib::create_gfx10(&gpu.device, &gpu.pool, &legacy_commands)
        }
        (Pm4Family::Gfx10, true) => {
            MultiQueuePm4Ib::create_profiled_gfx10(&gpu.device, &gpu.pool, &legacy_commands)
        }
        (Pm4Family::Gfx11, false) => {
            MultiQueuePm4Ib::create_gfx11(&gpu.device, &gpu.pool, &legacy_commands)
        }
        (Pm4Family::Gfx11, true) => {
            MultiQueuePm4Ib::create_profiled_gfx11(&gpu.device, &gpu.pool, &legacy_commands)
        }
        (Pm4Family::Gfx12, false) => {
            MultiQueuePm4Ib::create(&gpu.device, &gpu.pool, &gfx12_commands)
        }
        (Pm4Family::Gfx12, true) => {
            MultiQueuePm4Ib::create_profiled(&gpu.device, &gpu.pool, &gfx12_commands)
        }
    }
    .map_err(|_| ())?;
    Ok(RlPm4MultiIb {
        ib,
        kernargs,
        _modules: modules,
    })
}

/// Replay every independent lane once and wait for all lanes to complete.
///
/// # Safety
/// `ib` from [`rl_pm4_finalize_multi`], or null. Every lane's device memory
/// footprint must be independent of every other lane.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay_multi(ib: *mut RlPm4MultiIb) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    match unsafe { ib.ib.replay_and_wait() } {
        Ok(()) => RL_OK,
        Err(_) => RL_ERR_REPLAY,
    }
}

/// Profiled replay of every independent lane. Writes the cross-queue GPU
/// makespan in microseconds to `out_gpu_us`.
///
/// # Safety
/// `ib` from [`rl_pm4_finalize_multi_profiled`]; pointer and independence
/// requirements match [`rl_pm4_replay_multi`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay_multi_profiled(
    ib: *mut RlPm4MultiIb,
    out_gpu_us: *mut f64,
) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    if out_gpu_us.is_null() {
        return RL_ERR_NULL;
    }
    match unsafe { ib.ib.replay_and_wait_profiled() } {
        Ok(timing) => {
            unsafe { *out_gpu_us = timing.span_microseconds() };
            RL_OK
        }
        Err(_) => RL_ERR_REPLAY,
    }
}

/// Number of independent public-queue lanes retained by `ib`, or zero for null.
///
/// # Safety
/// `ib` from a multi-finalize function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_multi_ib_lane_count(ib: *const RlPm4MultiIb) -> usize {
    unsafe { ib.as_ref() }.map_or(0, |ib| ib.ib.queue_count())
}

/// Number of dispatches retained in `lane_index`, or zero for null/invalid lane.
///
/// # Safety
/// `ib` from a multi-finalize function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_multi_ib_dispatch_count(
    ib: *const RlPm4MultiIb,
    lane_index: usize,
) -> usize {
    unsafe { ib.as_ref() }
        .and_then(|ib| ib.kernargs.get(lane_index))
        .map_or(0, Vec::len)
}

/// Patch one retained kernarg segment selected by lane and dispatch index.
///
/// # Safety
/// `ib` from a multi-finalize function, or null; `kernarg` valid for `len`
/// bytes. Any device pointer written here must remain live through replay.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_multi_ib_set_kernargs(
    ib: *mut RlPm4MultiIb,
    lane_index: usize,
    dispatch_index: usize,
    byte_offset: usize,
    kernarg: *const u8,
    len: usize,
) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let Some(buffer) = ib
        .kernargs
        .get_mut(lane_index)
        .and_then(|lane| lane.get_mut(dispatch_index))
    else {
        return RL_ERR_HANDLE;
    };
    let Some(end) = byte_offset.checked_add(len) else {
        return RL_ERR_RECORD;
    };
    if end > buffer.len() {
        return RL_ERR_RECORD;
    }
    if len == 0 {
        return RL_OK;
    }
    if kernarg.is_null() {
        return RL_ERR_NULL;
    }
    let src = unsafe { std::slice::from_raw_parts(kernarg, len) };
    buffer.as_mut_bytes()[byte_offset..end].copy_from_slice(src);
    RL_OK
}

/// # Safety
/// `ib` from a multi-finalize function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_multi_ib_free(ib: *mut RlPm4MultiIb) {
    if !ib.is_null() {
        drop(unsafe { Box::from_raw(ib) });
    }
}

fn finalize_ib(
    gpu: &RlGpu,
    family: Pm4Family,
    commands: &Pm4Commands,
    profiled: bool,
) -> Result<SingleQueuePm4Ib, ()> {
    match (family, commands, profiled) {
        (Pm4Family::Gfx10, Pm4Commands::Legacy(commands), false) => {
            SingleQueuePm4Ib::create_gfx10(&gpu.device, &gpu.pool, commands).map_err(|_| ())
        }
        (Pm4Family::Gfx10, Pm4Commands::Legacy(commands), true) => {
            SingleQueuePm4Ib::create_profiled_gfx10(&gpu.device, &gpu.pool, commands)
                .map_err(|_| ())
        }
        (Pm4Family::Gfx11, Pm4Commands::Legacy(commands), false) => {
            SingleQueuePm4Ib::create_gfx11(&gpu.device, &gpu.pool, commands).map_err(|_| ())
        }
        (Pm4Family::Gfx11, Pm4Commands::Legacy(commands), true) => {
            SingleQueuePm4Ib::create_profiled_gfx11(&gpu.device, &gpu.pool, commands)
                .map_err(|_| ())
        }
        (Pm4Family::Gfx12, Pm4Commands::Gfx12(commands), false) => {
            SingleQueuePm4Ib::create(&gpu.device, &gpu.pool, commands).map_err(|_| ())
        }
        (Pm4Family::Gfx12, Pm4Commands::Gfx12(commands), true) => {
            SingleQueuePm4Ib::create_profiled(&gpu.device, &gpu.pool, commands).map_err(|_| ())
        }
        _ => unreachable!("PM4 command family is selected from the same device family"),
    }
}

/// Replay the retained IB once (submit + wait for completion). Returns `RL_OK`.
///
/// # Safety
/// `ib` from [`rl_pm4_finalize`], or null; every device pointer encoded in the
/// engine's kernargs must remain valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay(ib: *mut RlPm4Ib) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    // SAFETY: the retained IB owns its kernargs; the caller upholds pointee validity.
    match unsafe { ib.ib.replay_and_wait() } {
        Ok(()) => RL_OK,
        Err(_) => RL_ERR_REPLAY,
    }
}

/// Replay a profiled retained IB and write its GPU execution span in
/// microseconds to `out_gpu_us`.
///
/// # Safety
/// `ib` must come from [`rl_pm4_finalize_profiled`]; `out_gpu_us` must be a
/// valid writable pointer. Pointee lifetimes match [`rl_pm4_replay`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_replay_profiled(ib: *mut RlPm4Ib, out_gpu_us: *mut f64) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    if out_gpu_us.is_null() {
        return RL_ERR_NULL;
    }
    // SAFETY: the retained IB owns its kernargs; the caller upholds pointee validity.
    match unsafe { ib.ib.replay_and_wait_profiled() } {
        Ok(timing) => {
            unsafe { *out_gpu_us = timing.span_microseconds() };
            RL_OK
        }
        Err(_) => RL_ERR_REPLAY,
    }
}

/// Overwrite `len` bytes at `byte_offset` of the retained kernarg segment bound
/// to dispatch `dispatch_index` (in `rl_pm4_dispatch` record order), in place.
///
/// The PM4 packet keeps the same kernarg address, so the next [`rl_pm4_replay`]
/// observes the new values with **no IB rebuild** — this is the per-token update
/// path for a retained decode graph: build the IB once, then each token patch
/// only the scalars/pointers that changed (position, KV-cache slot, ...) and
/// replay. Between a completed replay and the next this is race-free, since
/// [`rl_pm4_replay`] waits for wave retirement before returning.
///
/// Returns `RL_OK`; `RL_ERR_NULL` (null `ib`, or null `kernarg` with `len > 0`);
/// `RL_ERR_HANDLE` (`dispatch_index` past the recorded dispatch count); or
/// `RL_ERR_RECORD` (`byte_offset + len` exceeds this dispatch's kernarg segment).
///
/// # Safety
/// `ib` from [`rl_pm4_finalize`], or null; `kernarg` valid for `len` bytes. Any
/// device pointer written here must stay GPU-live through the next replay.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_ib_set_kernargs(
    ib: *mut RlPm4Ib,
    dispatch_index: usize,
    byte_offset: usize,
    kernarg: *const u8,
    len: usize,
) -> i32 {
    let Some(ib) = (unsafe { ib.as_mut() }) else {
        return RL_ERR_NULL;
    };
    let Some(buffer) = ib.kernargs.get_mut(dispatch_index) else {
        return RL_ERR_HANDLE;
    };
    let Some(end) = byte_offset.checked_add(len) else {
        return RL_ERR_RECORD;
    };
    if end > buffer.len() {
        return RL_ERR_RECORD;
    }
    if len == 0 {
        return RL_OK;
    }
    if kernarg.is_null() {
        return RL_ERR_NULL;
    }
    // SAFETY: the caller guarantees `kernarg` is valid for `len` bytes.
    let src = unsafe { std::slice::from_raw_parts(kernarg, len) };
    buffer.as_mut_bytes()[byte_offset..end].copy_from_slice(src);
    RL_OK
}

/// # Safety
/// `ib` from [`rl_pm4_finalize`], or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pm4_ib_free(ib: *mut RlPm4Ib) {
    if !ib.is_null() {
        drop(unsafe { Box::from_raw(ib) });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Gfx10Pm4CommandBuffer, Gfx12Pm4CommandBuffer, Pm4Commands, Pm4Family, QueuePolicy,
        RL_ERR_NULL, RlQueuePolicy, rl_pm4_finalize_multi, rl_pm4_multi_ib_dispatch_count,
        rl_pm4_multi_ib_free, rl_pm4_multi_ib_lane_count, rl_pm4_multi_ib_set_kernargs,
        rl_pm4_replay_multi, rl_pm4_replay_multi_profiled,
    };

    #[test]
    fn rdna_generations_select_their_pm4_family() {
        assert_eq!(Pm4Family::from_name("gfx1010"), Some(Pm4Family::Gfx10));
        assert_eq!(Pm4Family::from_name("gfx1100"), Some(Pm4Family::Gfx11));
        assert_eq!(Pm4Family::from_name("gfx1151"), Some(Pm4Family::Gfx11));
        assert_eq!(Pm4Family::from_name("gfx1201"), Some(Pm4Family::Gfx12));
        assert_eq!(Pm4Family::from_name("gfx900"), None);
    }

    #[test]
    fn queue_policy_three_preserves_its_public_abi_value() {
        assert_eq!(RlQueuePolicy::RlQueueThree as u32, 3);
        assert_eq!(
            QueuePolicy::from(RlQueuePolicy::RlQueueThree),
            QueuePolicy::Three
        );
    }

    #[test]
    fn command_buffer_variant_matches_family() {
        assert!(matches!(
            Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx10),
            Pm4Commands::Legacy(_)
        ));
        assert!(matches!(
            Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx11),
            Pm4Commands::Legacy(_)
        ));
        assert!(matches!(
            Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx12),
            Pm4Commands::Gfx12(_)
        ));
    }

    #[test]
    fn leading_acquire_matches_generation_policy() {
        let mut expected_legacy = Gfx10Pm4CommandBuffer::new_stateful();
        expected_legacy.acquire_system();
        match Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx10) {
            Pm4Commands::Legacy(cmd) => assert_eq!(cmd.dwords(), expected_legacy.dwords()),
            Pm4Commands::Gfx12(_) => panic!("gfx10 must use the legacy command buffer"),
        }

        let mut expected_gfx11 = Gfx10Pm4CommandBuffer::new_stateful();
        expected_gfx11.acquire_system();
        match Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx11) {
            Pm4Commands::Legacy(cmd) => assert_eq!(cmd.dwords(), expected_gfx11.dwords()),
            Pm4Commands::Gfx12(_) => panic!("gfx11 must use the legacy command buffer"),
        }

        let mut expected_gfx12 = Gfx12Pm4CommandBuffer::new_stateful();
        expected_gfx12.acquire_inter_node_gfx12();
        match Pm4Commands::stateful_with_leading_acquire(Pm4Family::Gfx12) {
            Pm4Commands::Gfx12(cmd) => assert_eq!(cmd.dwords(), expected_gfx12.dwords()),
            Pm4Commands::Legacy(_) => panic!("gfx12 must use the gfx12 command buffer"),
        }
    }
    #[test]
    fn multiqueue_c_abi_is_null_safe() {
        unsafe {
            assert_eq!(
                rl_pm4_finalize_multi(std::ptr::null(), std::ptr::null(), 0, std::ptr::null_mut(),),
                RL_ERR_NULL
            );
            assert_eq!(rl_pm4_replay_multi(std::ptr::null_mut()), RL_ERR_NULL);
            assert_eq!(
                rl_pm4_replay_multi_profiled(std::ptr::null_mut(), std::ptr::null_mut()),
                RL_ERR_NULL
            );
            assert_eq!(rl_pm4_multi_ib_lane_count(std::ptr::null()), 0);
            assert_eq!(rl_pm4_multi_ib_dispatch_count(std::ptr::null(), 0), 0);
            assert_eq!(
                rl_pm4_multi_ib_set_kernargs(std::ptr::null_mut(), 0, 0, 0, std::ptr::null(), 0,),
                RL_ERR_NULL
            );
            rl_pm4_multi_ib_free(std::ptr::null_mut());
        }
    }
}

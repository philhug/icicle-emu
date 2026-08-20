//! KVM-backed execution engine (x86-64 first).
//!
//! This is the second execution backend behind [`Vm::run`](crate::Vm::run),
//! selected by [`Backend::Kvm`](crate::Backend::Kvm). Where the JIT lifts to
//! p-code and runs blocks in-process, this engine runs the guest natively on
//! the host through KVM and dispatches vmexits back into the existing
//! `IoMemory`/`VmExit` model.
//!
//! ## Layout
//!
//! Construction ([`Vcpu::new`]) opens `/dev/kvm`, creates the VM and one vCPU,
//! and mmaps the `kvm_run` region. Memory and vmexit dispatch are layered on
//! top in the tasks that follow (memslots mirror the `Mmu` region map, MMIO is
//! routed into the existing `IoMemory` path).

use std::os::fd::IntoRawFd;
use std::os::unix::io::RawFd;

use icicle_cpu::VmExit;

/// KVM ioctl numbers for the construction set are plain `_IO` (no data
/// structure), so they reduce to `(KVMIO<<8) | nr`, KVMIO = 0xAE.
const KVM_GET_API_VERSION: libc::c_ulong = 0xAE00;
const KVM_CREATE_VM: libc::c_ulong = 0xAE01;
const KVM_CHECK_EXTENSION: libc::c_ulong = 0xAE03;
const KVM_GET_VCPU_MMAP_SIZE: libc::c_ulong = 0xAE04;
const KVM_CREATE_VCPU: libc::c_ulong = 0xAE41;

/// The minimum `KVM_GET_API_VERSION` the engine assumes. KVM_RUN semantics this
/// engine uses are stable well below this; 12 is the Firecracker-era floor that
/// targets the vCPU/MMU ioctls in the tasks ahead.
const MIN_API: libc::c_int = 12;

/// A constructed KVM VM + vCPU with its `kvm_run` regions mmap'd.
///
/// This owns the three kernel fds (kvm/VM/vCPU) and the run buffer. Memory and
/// vmexit wiring are methods layered on in later tasks.
pub struct Vcpu {
    /// The `kvm_run` buffer is written by the kernel on every vmexit. It is
    /// only touched from the thread issuing KVM_RUN, so it is safe to mark
    /// Send. The GuestMemory in a KVM-backed Vm is `!Sync` like the JIT's, and
    /// one Vm runs on one thread, so this is consistent with the rest of
    /// icicle's single-core model.
    run: *mut u8,
    run_len: usize,
    vcpu: RawFd,
    vm: RawFd,
    kvm: RawFd,
}

// SAFETY: the fds and run buffer are owned by this Vcpu; KVM_RUN is issued from
// the same thread that accesses `run`, and icicle runs one Vm per thread. The
// same `Rc`-based non-Send reasoning that excludes `Vm` from other threads
// holds here.
unsafe impl Send for Vcpu {}

impl Drop for Vcpu {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.run.cast(), self.run_len);
            libc::close(self.vcpu);
            libc::close(self.vm);
            libc::close(self.kvm);
        }
    }
}

/// Error from constructing a KVM backend.
#[derive(Debug)]
pub enum KvmError {
    /// The kernel's `KVM_GET_API_VERSION` is older than `MIN_API`.
    ApiTooOld(libc::c_int),
    /// Opening `/dev/kvm` failed (absent, or permission denied).
    Open(std::io::Error),
    /// An ioctl on one of the kvm fds failed.
    Ioctl(&'static str, std::io::Error),
}

impl std::fmt::Display for KvmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvmError::ApiTooOld(v) => write!(f, "KVM API too old: {v} < 12"),
            KvmError::Open(e) => write!(f, "failed to open /dev/kvm: {e}"),
            KvmError::Ioctl(op, e) => write!(f, "KVM ioctl {op} failed: {e}"),
        }
    }
}

impl std::error::Error for KvmError {}

/// The KVM capability ids this engine queries via `KVM_CHECK_EXTENSION`.
const KVM_CAP_USER_MEMORY: libc::c_int = 3;

impl Vcpu {
    /// Open `/dev/kvm`, create the VM and one vCPU, and mmap the run region.
    ///
    /// # Errors
    /// [`KvmError::Open`] if `/dev/kvm` is unavailable, [`KvmError::ApiTooOld`]
    /// if the kernel exposes an older API, or [`KvmError::Ioctl`] on a syscall
    /// failure.
    pub fn new() -> Result<Vcpu, KvmError> {
        let kvm = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm")
            .map_err(KvmError::Open)?
            .into_raw_fd();

        let api_version = unsafe { libc::ioctl(kvm, KVM_GET_API_VERSION, 0) };
        if api_version < 0 {
            unsafe { libc::close(kvm) };
            return Err(KvmError::Ioctl("KVM_GET_API_VERSION", std::io::Error::last_os_error()));
        }
        if api_version < MIN_API {
            unsafe { libc::close(kvm) };
            return Err(KvmError::ApiTooOld(api_version));
        }

        let vm = unsafe { libc::ioctl(kvm, KVM_CREATE_VM, 0) };
        if vm < 0 {
            unsafe { libc::close(kvm) };
            return Err(KvmError::Ioctl("KVM_CREATE_VM", std::io::Error::last_os_error()));
        }

        let vcpu = unsafe { libc::ioctl(vm, KVM_CREATE_VCPU, 0) };
        if vcpu < 0 {
            unsafe {
                libc::close(vm);
                libc::close(kvm);
            }
            return Err(KvmError::Ioctl("KVM_CREATE_VCPU", std::io::Error::last_os_error()));
        }

        // KVM_GET_VCPU_MMAP_SIZE is a KVM *device* ioctl here (not a per-vcpu
        // one): it is dispatched from kvm_dev_ioctl, so it must go on the kvm
        // fd, not the vcpu fd.
        let run_len = unsafe { libc::ioctl(kvm, KVM_GET_VCPU_MMAP_SIZE, 0) };
        if run_len < 0 {
            unsafe {
                libc::close(vcpu);
                libc::close(vm);
                libc::close(kvm);
            }
            return Err(KvmError::Ioctl(
                "KVM_GET_VCPU_MMAP_SIZE",
                std::io::Error::last_os_error(),
            ));
        }

        let run = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                run_len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                vcpu,
                0,
            )
        };
        if run == libc::MAP_FAILED {
            unsafe {
                libc::close(vcpu);
                libc::close(vm);
                libc::close(kvm);
            }
            return Err(KvmError::Ioctl("mmap(kvm_run)", std::io::Error::last_os_error()));
        }

        // Safety: cast from a validated MAP_SHARED mapping that we own.
        let run_ptr = run as *mut u8;
        Ok(Vcpu { run: run_ptr, run_len: run_len as usize, vcpu, vm, kvm })
    }

    /// The VM fd. Required for memslot setup (the next task).
    pub fn vm_fd(&self) -> RawFd {
        self.vm
    }

    /// The vCPU fd. Required for KVM_RUN and register access.
    pub fn vcpu_fd(&self) -> RawFd {
        self.vcpu
    }

    /// The `kvm_run` buffer, mmap'd from the vCPU fd.
    pub fn kvm_run(&self) -> *mut u8 {
        self.run
    }

    /// Whether the host exposes a KVM capability (via `KVM_CHECK_EXTENSION`).
    pub fn has_cap(&self, cap: libc::c_int) -> bool {
        unsafe { libc::ioctl(self.kvm, KVM_CHECK_EXTENSION, cap) > 0 }
    }

    /// Whether the host supports `KVM_SET_USER_MEMORY_REGION`, required to
    /// install guest memory.
    pub fn supports_user_memory(&self) -> bool {
        self.has_cap(KVM_CAP_USER_MEMORY)
    }
}

impl std::fmt::Debug for Vcpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vcpu")
            .field("vm_fd", &self.vm)
            .field("vcpu_fd", &self.vcpu)
            .field("run_len", &self.run_len)
            .finish()
    }
}

/// Run a KVM-backed [`Vm`](crate::Vm) until an exit.
///
/// The run loop (KVM_RUN + vmexit dispatch + memslot setup) is the next task;
/// until then a KVM-selected Vm returns this sentinel without executing.
pub fn run(_vm: &mut crate::Vm) -> crate::VmExit {
    crate::VmExit::Interrupted
}

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_cpu::Config;

    fn kvm_expected() -> bool {
        std::path::Path::new("/dev/kvm").exists() && cfg!(target_arch = "x86_64")
    }

    #[test]
    fn constructs_on_kvm_host() {
        if !kvm_expected() {
            eprintln!("skipping: no /dev/kvm or not x86_64");
            return;
        }
        let vcpu = Vcpu::new().expect("KVM construction should succeed on a kvm host");
        // The API/creation ioctls succeeded; the fds are live and the run
        // buffer is mapped.
        assert!(vcpu.vm_fd() >= 0);
        assert!(vcpu.vcpu_fd() >= 0);
        let (ptr, len) = (vcpu.kvm_run(), vcpu.run_len);
        assert!(!ptr.is_null());
        assert!(len > 0);
        // Every modern host supports KVM_SET_USER_MEMORY_REGION.
        assert!(vcpu.supports_user_memory());
    }

    #[test]
    fn errors_when_kvm_absent() {
        if kvm_expected() {
            eprintln!("skipping: /dev/kvm present, cannot test absence path");
            return;
        }
        assert!(Vcpu::new().is_err());
    }

    #[test]
    fn run_returns_interrupted_sentinel_until_wired() {
        // The run-loop task replaces this; until then a KVM Vm must not claim
        // to have executed.
        let mut vm = crate::build(&Config::from_target_triple("x86_64-none")).unwrap();
        vm.backend = crate::Backend::Kvm;
        assert_eq!(vm.run(), crate::VmExit::Interrupted);
    }
}
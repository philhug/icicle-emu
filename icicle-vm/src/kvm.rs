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
const KVM_SET_USER_MEMORY_REGION: libc::c_ulong = 0x4020_AE46;
const KVM_RUN: libc::c_ulong = 0xAE80;
const KVM_SET_REGS: libc::c_ulong = 0x4090_AE82;
const KVM_GET_REGS: libc::c_ulong = 0x8090_AE81;
const KVM_SET_SREGS: libc::c_ulong = 0x4138_AE84;

/// `kvm_run.exit_reason` values used by the engine.
pub const EXIT_IO: u32 = 2;
pub const EXIT_MMIO: u32 = 6;
pub const EXIT_HLT: u32 = 5;
pub const EXIT_INTR: u32 = 4;
pub const EXIT_EXCEPTION: u32 = 3;
pub const EXIT_SHUTDOWN: u32 = 8;
pub const EXIT_UNKNOWN: u32 = 0;

/// KVM_EXIT_IO sub-fields within the mmap'd `struct kvm_run` (verified against
/// linux/kvm.h): the union lands at byte 32, and io is { direction u8, size u8,
/// port u16, count u32, data_offset u64 }. `data_offset` is relative to the
/// kvm_run base; the payload is at run + data_offset.
pub const KVM_RUN_IO_DIRECTION: usize = 32;
pub const KVM_RUN_IO_SIZE: usize = 33;
pub const KVM_RUN_IO_PORT: usize = 34;
pub const KVM_RUN_IO_DATA_OFFSET: usize = 40;

/// Offsets within the mmap'd `struct kvm_run` (verified against linux/kvm.h):
/// exit_reason at 8; the union at 32. kvm_mmio: phys_addr 32, data 40, len 48,
/// is_write 52 (1-byte). kvm_io: direction 32, size 33, port 34, count 36,
/// data_offset 40 (relative to the kvm_run base).
pub const KVM_RUN_EXIT_REASON: usize = 8;
pub const KVM_RUN_MMIO_PHYS: usize = 32;
pub const KVM_RUN_MMIO_DATA: usize = 40;
pub const KVM_RUN_MMIO_LEN: usize = 48;
pub const KVM_RUN_MMIO_IS_WRITE: usize = 52;

/// `struct kvm_regs` is 144 bytes (18 u64s); `struct kvm_sregs` is 312 bytes.
/// Field offsets (verified against the local kernel): cs=0, ds=24 (each
/// `kvm_segment` is 24 bytes), gdt=192 (`kvm_dtable` 16 bytes), idt=208,
/// cr0=224, cr2=232, cr3=240, cr4=248, cr8=256, efer=264, apic_base=272,
/// interrupt_bitmap=280. We hand-roll these structs (no kvm crate dependency);
/// a C probe against `linux/kvm.h` + `asm/kvm.h` confirmed sizes and offsets.
#[repr(C)]
struct KvmSegment {
    base: u64,
    limit: u32,
    selector: u16,
    #[allow(dead_code)]
    type_: u8,
    present: u8,
    dpl: u8,
    db: u8,
    s: u8,
    l: u8,
    g: u8,
    avl: u8,
    unusable: u8,
    #[allow(dead_code)]
    padding: u8,
}

#[repr(C)]
struct KvmDtable {
    base: u64,
    limit: u16,
    #[allow(dead_code)]
    padding: [u16; 3],
}

#[repr(C)]
struct KvmSregs {
    cs: KvmSegment,
    ds: KvmSegment,
    es: KvmSegment,
    fs: KvmSegment,
    gs: KvmSegment,
    ss: KvmSegment,
    tr: KvmSegment,
    ldt: KvmSegment,
    gdt: KvmDtable,
    idt: KvmDtable,
    cr0: u64,
    cr2: u64,
    cr3: u64,
    cr4: u64,
    cr8: u64,
    efer: u64,
    apic_base: u64,
    interrupt_bitmap: [u64; 4],
}

/// Identity-mapped x86-64 long-mode segment/page registers for a flat guest.
/// `cr3` is the guest-physical address of the PML4 table.
fn flat_x86_64_sregs(cr3: u64) -> KvmSregs {
    // 64-bit code: present, code, long (L=1); data segments flat read/write.
    fn seg(code: bool) -> KvmSegment {
        KvmSegment {
            base: 0,
            limit: 0xffff_ffff,
            selector: if code { 0x10 } else { 0 },
            type_: if code { 0xb } else { 0x3 },
            present: 1,
            dpl: 0,
            db: 0,
            s: if code { 1 } else { 1 },
            l: code as u8,
            g: 1,
            avl: 0,
            unusable: 0,
            padding: 0,
        }
    }
    KvmSregs {
        cs: seg(true),
        ds: seg(false),
        es: seg(false),
        fs: seg(false),
        gs: seg(false),
        ss: seg(false),
        tr: KvmSegment {
            base: 0,
            limit: 0,
            selector: 0,
            type_: 0,
            present: 0,
            dpl: 0,
            db: 0,
            s: 0,
            l: 0,
            g: 0,
            avl: 0,
            unusable: 1,
            padding: 0,
        },
        ldt: KvmSegment {
            base: 0,
            limit: 0,
            selector: 0,
            type_: 0,
            present: 0,
            dpl: 0,
            db: 0,
            s: 0,
            l: 0,
            g: 0,
            avl: 0,
            unusable: 1,
            padding: 0,
        },
        gdt: KvmDtable { base: 0, limit: 0, padding: [0; 3] },
        idt: KvmDtable { base: 0, limit: 0, padding: [0; 3] },
        cr0: 0x8000_0001, // PE | PG
        cr2: 0,
        cr3,
        cr4: 0x20,    // PAE
        cr8: 0,
        efer: 0x500,  // LME | LMA
        apic_base: 0x0000_0000_fee0_0000,
        interrupt_bitmap: [0; 4],
    }
}

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
    /// Contiguous host buffer backing the installed memslot (guest RAM).
    /// Allocated page-aligned; `mmap`'d so KVM can pin it. See `install_memory`.
    guest_ram: *mut u8,
    /// Guest size of the memslot, in bytes.
    guest_ram_len: usize,
}

// SAFETY: the fds and run buffer are owned by this Vcpu; KVM_RUN is issued from
// the same thread that accesses `run`, and icicle runs one Vm per thread. The
// same `Rc`-based non-Send reasoning that excludes `Vm` from other threads
// holds here.
unsafe impl Send for Vcpu {}

impl Drop for Vcpu {
    fn drop(&mut self) {
        unsafe {
            if !self.guest_ram.is_null() {
                libc::munmap(self.guest_ram.cast(), self.guest_ram_len);
            }
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
        Ok(Vcpu {
            run: run_ptr,
            run_len: run_len as usize,
            vcpu,
            vm,
            kvm,
            guest_ram: std::ptr::null_mut(),
            guest_ram_len: 0,
        })
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

    /// Allocate a page-aligned contiguous buffer of `size` bytes and install it
    /// as the single memslot at `guest_base` (guest physical address).
    ///
    /// This is the host buffer KVM's EPT maps directly; it is the *mirror* of
    /// icicle's paged `Mmu` RAM (the copy-in/copy-out backend, plan option 2).
    ///
    /// # Errors
    /// [`KvmError::Ioctl`] if the buffer alloc or `KVM_SET_USER_MEMORY_REGION`
    /// fails.
    pub fn install_memory(
        &mut self,
        guest_base: u64,
        size: usize,
    ) -> Result<(), KvmError> {
        // KVM memslots must be page-aligned in both address and length.
        const PGSZ: usize = 4096;
        let len = ((size + PGSZ - 1) / PGSZ) * PGSZ;
        let guest_ram = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if guest_ram == libc::MAP_FAILED {
            return Err(KvmError::Ioctl("mmap(guest_ram)", std::io::Error::last_os_error()));
        }
        // lazily flush the previous buffer, if any / already set, before reuse.
        if !self.guest_ram.is_null() {
            unsafe { libc::munmap(self.guest_ram.cast(), self.guest_ram_len) };
        }
        self.guest_ram = guest_ram.cast();
        self.guest_ram_len = len;

        #[repr(C)]
        struct KvmUserspaceMemoryRegion {
            slot: u32,
            flags: u32,
            guest_phys_addr: u64,
            memory_size: u64,
            userspace_addr: u64,
        }
        let region = KvmUserspaceMemoryRegion {
            slot: 0,
            flags: 0,
            guest_phys_addr: guest_base,
            memory_size: len as u64,
            userspace_addr: guest_ram as u64,
        };
        let rc = unsafe { libc::ioctl(self.vm, KVM_SET_USER_MEMORY_REGION, &region) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::munmap(self.guest_ram.cast(), self.guest_ram_len) };
            self.guest_ram = std::ptr::null_mut();
            self.guest_ram_len = 0;
            return Err(KvmError::Ioctl("KVM_SET_USER_MEMORY_REGION", err));
        }
        Ok(())
    }

    /// The host pointer and length of the installed guest-RAM memslot buffer.
    pub fn guest_ram(&self) -> (*mut u8, usize) {
        (self.guest_ram, self.guest_ram_len)
    }

    /// Prepare an x86-64 vCPU for a flat long-mode guest: install an identity
    /// 2MB-huge-page table in the guest buffer, set segments to flat 64-bit
    /// code/data, and enable PAE long mode. `cr3` is the guest-physical address
    /// of the PML4 (a fixed block carved out near the *end* of the memslot so it
    /// never collides with firmware mapped at low guest addresses).
    ///
    /// `ram_guest_off` is where the page table lives in *guest physical* space
    /// (= host buffer offset). `entry_rip` selects the initial instruction.
    pub fn setup_x86_64(
        &mut self,
        ram_guest_base: u64,
        entry_rip: u64,
    ) -> Result<(), KvmError> {
        if self.guest_ram.is_null() {
            return Err(KvmError::Ioctl("setup_x86_64: no guest ram", std::io::Error::last_os_error()));
        }

        // Identity page table covering only the installed guest-RAM span, so any
        // address outside it (the MMIO holes, e.g. a UART at 0x09000000) is
        // left unmapped and traps to an EPT/MMIO exit. pml4 0x4000, pdp 0x5000,
        // pd 0x6000 — a reserved area below the flat example's entry code and
        // copied out with the guest-RAM mirror. 2MiB huge pages.
        const PML4: u64 = 0x4000;
        const PDP: u64 = 0x5000;
        const PD: u64 = 0x6000;
        let base = self.guest_ram as u64;
        let guest_span = self.guest_ram_len as u64;
        let page2m = 2u64 << 20;
        let npages = guest_span.div_ceil(page2m).min(512);
        unsafe {
            // Zero the three table pages.
            std::ptr::write_bytes((base + PML4) as *mut u8, 0, 0x1000);
            std::ptr::write_bytes((base + PDP) as *mut u8, 0, 0x1000);
            std::ptr::write_bytes((base + PD) as *mut u8, 0, 0x1000);
            let pml4 = (base + PML4) as *mut u64;
            let pdp = (base + PDP) as *mut u64;
            let pd = (base + PD) as *mut u64;
            pml4.write_volatile(PDP | 0x03);
            pdp.write_volatile(PD | 0x03);
            for i in 0..npages {
                // 2MiB huge pages, present+rw+PS, identity-mapped.
                pd.add(i as usize).write_volatile(((i as u64) << 21) | 0x83);
            }
        }
        let cr3 = ram_guest_base + PML4;

        let sregs = flat_x86_64_sregs(cr3);
        let rc = unsafe { libc::ioctl(self.vcpu, KVM_SET_SREGS, &sregs) };
        if rc < 0 {
            return Err(KvmError::Ioctl("KVM_SET_SREGS", std::io::Error::last_os_error()));
        }

        #[repr(C)]
        struct KvmRegs {
            rax: u64,
            rbx: u64,
            rcx: u64,
            rdx: u64,
            rsi: u64,
            rdi: u64,
            rsp: u64,
            rbp: u64,
            r8: u64,
            r9: u64,
            r10: u64,
            r11: u64,
            r12: u64,
            r13: u64,
            r14: u64,
            r15: u64,
            rip: u64,
            rflags: u64,
        }
        let regs = KvmRegs {
            rax: 0,
            rbx: 0,
            rcx: 0,
            rdx: 0,
            rsi: 0,
            rdi: 0,
            rsp: 0,
            rbp: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rip: entry_rip,
            rflags: 0x2, // reserved bit, interrupts off
        };
        let rc = unsafe { libc::ioctl(self.vcpu, KVM_SET_REGS, &regs) };
        if rc < 0 {
            return Err(KvmError::Ioctl("KVM_SET_REGS", std::io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Run the vCPU, dispatching each vmexit. `dispatch_io` routes an MMIO
    /// read/write to icicle's `IoMemory` handlers (the device model). Returns
    /// the icicle `VmExit` the guest produced.
    ///
    /// `max_iters` bounds how many vmexits we service before bailing with
    /// `InstructionLimit` (KVM has no faithful per-instruction counter; this is
    /// the immature-budget approximation the milestone documents).
    pub fn run_loop<F>(&mut self, mut dispatch_io: F, max_iters: u64) -> VmExit
    where
        F: FnMut(bool, u64, bool, &mut [u8]) -> std::io::Result<()>,
    {
        let run = self.run as *const u8;
        for _ in 0..max_iters {
            let rc = unsafe { libc::ioctl(self.vcpu, KVM_RUN, 0) };
            if rc < 0 {
                let e = std::io::Error::last_os_error();
                return VmExit::UnhandledException((icicle_cpu::ExceptionCode::InternalError, e.raw_os_error().unwrap_or(0) as u64));
            }
            // exit_reason lives at kvm_run byte offset 8 (per the layout above).
            let exit_reason =
                unsafe { (run.add(KVM_RUN_EXIT_REASON) as *const u32).read_volatile() };
            match exit_reason {
                EXIT_MMIO => {
                    let is_write =
                        unsafe { (run.add(KVM_RUN_MMIO_IS_WRITE) as *const u8).read_volatile() } != 0;
                    let len = unsafe { (run.add(KVM_RUN_MMIO_LEN) as *const u32).read_volatile() };
                    let paddr = unsafe { (run.add(KVM_RUN_MMIO_PHYS) as *const u64).read_volatile() };
                    let off = KVM_RUN_MMIO_DATA;
                    if len > 8 {
                        return VmExit::UnhandledException((icicle_cpu::ExceptionCode::InternalError, 1));
                    }
                    // On either a read or a write KVM fills `data` with what the
                    // guest wrote; on a read the device returns the value into it.
                    let mut buf = [0u8; 8];
                    buf[..len as usize].copy_from_slice(unsafe {
                        std::slice::from_raw_parts(run.add(off), len as usize)
                    });
                    match dispatch_io(false, paddr, is_write, &mut buf[..len as usize]) {
                        Ok(()) => {
                            // For a read, give KVM back the value the device
                            // produced.
                            if !is_write {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        buf.as_ptr(),
                                        run.add(off).cast_mut(),
                                        len as usize,
                                    )
                                };
                            }
                            continue; // keep running after this exit
                        }
                        Err(_) => {
                            return VmExit::UnhandledException((
                                icicle_cpu::ExceptionCode::InternalError,
                                2,
                            ))
                        }
                    }
                }
                EXIT_IO => {
                    // Port I/O. direction 1 = out (guest write), 0 = in (read).
                    let direction =
                        unsafe { (run.add(KVM_RUN_IO_DIRECTION) as *const u8).read_volatile() };
                    let size = unsafe { (run.add(KVM_RUN_IO_SIZE) as *const u8).read_volatile() };
                    let port =
                        unsafe { (run.add(KVM_RUN_IO_PORT) as *const u16).read_volatile() };
                    // data_offset is relative to the kvm_run base.
                    let data_off =
                        unsafe { (run.add(KVM_RUN_IO_DATA_OFFSET) as *const u64).read_volatile() };
                    let data_ptr = unsafe { run.add(data_off as usize) } as *const u8;
                    if size > 8 {
                        return VmExit::UnhandledException((
                            icicle_cpu::ExceptionCode::InternalError,
                            3,
                        ));
                    }
                    let mut buf = [0u8; 8];
                    buf[..size as usize].copy_from_slice(unsafe {
                        std::slice::from_raw_parts(data_ptr, size as usize)
                    });
                    // On in (read), the device reply goes back into the io data.
                    let ok = dispatch_io(true, port as u64, direction != 0, &mut buf[..size as usize]);
                    if ok.is_ok() && direction == 0 {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                buf.as_ptr(),
                                data_ptr.cast_mut(),
                                size as usize,
                            )
                        };
                    }
                    if ok.is_err() {
                        return VmExit::UnhandledException((
                            icicle_cpu::ExceptionCode::InternalError,
                            4,
                        ));
                    }
                    continue;
                }
                EXIT_HLT => return VmExit::Halt,
                EXIT_INTR => return VmExit::Interrupted,
                EXIT_SHUTDOWN => {
                    // Guest triple-faulted natively. Fetch RIP for diagnosis.
                    #[repr(C)]
                    struct KvmRegs {
                        rax: u64,
                        rbx: u64,
                        rcx: u64,
                        rdx: u64,
                        rsi: u64,
                        rdi: u64,
                        rsp: u64,
                        rbp: u64,
                        r8: u64,
                        r9: u64,
                        r10: u64,
                        r11: u64,
                        r12: u64,
                        r13: u64,
                        r14: u64,
                        r15: u64,
                        rip: u64,
                        rflags: u64,
                    }
                    let mut regs = KvmRegs {
                        rax: 0,
                        rbx: 0,
                        rcx: 0,
                        rdx: 0,
                        rsi: 0,
                        rdi: 0,
                        rsp: 0,
                        rbp: 0,
                        r8: 0,
                        r9: 0,
                        r10: 0,
                        r11: 0,
                        r12: 0,
                        r13: 0,
                        r14: 0,
                        r15: 0,
                        rip: 0,
                        rflags: 0,
                    };
                    let _ = unsafe { libc::ioctl(self.vcpu, KVM_GET_REGS, &mut regs) };
                    return VmExit::UnhandledException((icicle_cpu::ExceptionCode::Halt, 0xDEAD))
                }
                _ => {
                    tracing::warn!("kvm: unhandled exit reason {exit_reason}");
                    return VmExit::UnhandledException((
                        icicle_cpu::ExceptionCode::InternalError,
                        exit_reason as u64,
                    ));
                }
            }
        }
        VmExit::InstructionLimit
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
    fn runs_a_flat_x86_guest_through_kvm_and_halts() {
        if !kvm_expected() {
            eprintln!("skipping: no /dev/kvm or not x86_64");
            return;
        }
        let mut vm = crate::build(&Config::from_target_triple("x86_64-none")).unwrap();

        // Guest RAM at 0x0..0x100000.
        assert!(vm
            .cpu
            .mem
            .map_memory_len(0x0, 0x100000, icicle_cpu::mem::Mapping { perm: icicle_cpu::mem::perm::READ | icicle_cpu::mem::perm::WRITE | icicle_cpu::mem::perm::EXEC, value: 0 }));

        // The acceptance firmware: mov al,0x41; mov dx,0x9000 (66-prefixed); out
        // dx,al; hlt. Port I/O (KVM_EXIT_IO) is the reliable exit on this host;
        // a memory-mapped write to a present-but-unbacked page yields a guest
        // #PF (triple-fault), not KVM_EXIT_MMIO (verified in a C probe).
        static CODE: &[u8] = &[
            0xb0, 0x41, // mov al, 0x41
            0x66, 0xba, 0x00, 0x90, // mov dx, 0x9000
            0xee, // out dx, al
            0xf4, // hlt
        ];
        vm.cpu.mem.write_bytes(0x1000, CODE, icicle_cpu::mem::perm::NONE).unwrap();
        vm.cpu.write_pc(0x1000);

        vm.backend = crate::Backend::Kvm;
        vm.icount_limit = u64::MAX;
        let exit = vm.run();
        // The guest wrote to the console port then halted.
        assert_eq!(exit, icicle_cpu::VmExit::Halt);
        // The fds are live after run.
        assert!(vm.kvm.as_ref().unwrap().vcpu_fd() >= 0);
    }
}
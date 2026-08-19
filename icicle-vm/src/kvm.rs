//! KVM-backed execution engine (x86-64 first).
//!
//! This is the second execution backend behind [`Vm::run`](crate::Vm::run),
//! selected by [`Backend::Kvm`](crate::Backend::Kvm). Where the JIT lifts to
//! p-code and runs blocks in-process, this engine runs the guest natively on
//! the host through KVM and dispatches vmexits back into the existing
//! `IoMemory`/`VmExit` model.
//!
//! The engine is a work in progress: the run-step seam and backend selection
//! exist (see `lib.rs`), and this module will grow the `/dev/kvm` setup,
//! memslot mirroring, and the vmexit loop.

/// Run a KVM-backed [`Vm`](crate::Vm) until an exit.
///
/// Placeholder until the engine is wired. Unreachable today because no
/// construction path selects [`Backend::Kvm`](crate::Backend::Kvm).
pub fn run(_vm: &mut crate::Vm) -> crate::VmExit {
    crate::VmExit::Interrupted
}
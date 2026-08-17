use std::{cell::RefCell, rc::Rc};

use icicle_cpu::mem::MemResult;

use crate::{
    LinuxMmu, errno,
    fs::{
        self, DEFAULT_INODE_VTABLE, FileKind, FileSystem, Inode, InodeRef, InodeVtable, file,
        host::TempFs,
    },
};

pub const AF_UNSPEC: u64 = 0;

/// Unix domain sockets (UDS).
pub const AF_UNIX: u64 = 1;

/// IPv4 socket.
pub const AF_INET: u64 = 2;

/// IPv6 socket.
pub const AF_INET6: u64 = 10;

/// Netlink socket (used for user space <-> kernel communication).
pub const AF_NETLINK: u64 = 16;

pub const SOCK_STREAM: u64 = 1;
pub const SOCK_DGRAM: u64 = 2;
pub const SOCK_RAW: u64 = 3;
pub const SOCK_RDM: u64 = 4;
pub const SOCK_SEQPACKET: u64 = 5;
pub const SOCK_DCCP: u64 = 6;

pub const SOCK_CLOEXEC: u64 = 0o02000000;
pub const SOCK_NONBLOCK: u64 = 0o00004000;

// @todo: checkme
pub const SOCKET_STORAGE_SIZE: usize = 64;

#[derive(Clone)]
pub struct SocketAddr {
    pub addr: [u8; SOCKET_STORAGE_SIZE],
}

impl SocketAddr {
    pub fn read_user<M: LinuxMmu>(mem: &mut M, addr: u64, len: u64) -> MemResult<Option<Self>> {
        if addr == 0 {
            return Ok(None);
        }

        let mut value = Self::default();
        let len = usize::min(len as usize, value.addr.len());
        mem.read_bytes(addr, &mut value.addr[..len])?;

        Ok(Some(value))
    }
}

impl Default for SocketAddr {
    fn default() -> Self {
        Self { addr: [0; SOCKET_STORAGE_SIZE] }
    }
}

pub struct Message<'a> {
    pub address: Option<&'a mut SocketAddr>,
    pub buf: &'a mut [u8],
}

/// A host-side provider of real network access.
///
/// icicle-linux has no networking of its own and cannot depend on whatever
/// embeds it, so this is the neutral seam: an embedder installs an
/// implementation with [`SocketFs::set_net_backend`], and every network socket
/// the guest creates afterwards routes its `connect`/`send`/`recv`/`close`
/// through it. With no backend installed the guest cannot reach a network at
/// all — [`TcpSocket::connect`] answers `EACCES` — so networking is off unless
/// an embedder deliberately turns it on.
///
/// Handles are opaque to icicle: [`NetBackend::connect`] returns whatever
/// non-negative value identifies the connection on the host side, and the
/// socket inode hands that same value back on every later call. Errors are
/// plain errno values, reported to the guest unchanged.
pub trait NetBackend {
    /// Opens a host socket of `domain`/`kind` (the `AF_*`/`SOCK_*` values the
    /// guest passed to `socket`) connected — or, for datagram sockets, bound —
    /// to `addr`, the raw `sockaddr` bytes the guest passed to `connect`.
    /// Returns an opaque non-negative handle.
    fn connect(&self, domain: u64, kind: u64, addr: &[u8]) -> fs::Result<i64>;

    /// Sends `buf` on the connected handle `h`, returning the number of bytes
    /// accepted.
    fn send(&self, h: i64, buf: &[u8]) -> fs::Result<usize>;

    /// Reads up to `out.len()` bytes from `h`. A return of 0 means the peer
    /// closed the connection.
    fn recv(&self, h: i64, out: &mut [u8]) -> fs::Result<usize>;

    /// Sends `buf` on `h` to `addr`. Datagram sockets only -- see
    /// [`UdpSocket`], which checks a destination on every call rather than
    /// once at connect time.
    fn sendto(&self, h: i64, buf: &[u8], addr: &[u8]) -> fs::Result<usize>;

    /// Reads up to `out.len()` bytes from `h`, writing the sender's address
    /// into `addr`. Returns the byte count and the address length. Datagram
    /// sockets only; see [`NetBackend::sendto`].
    fn recvfrom(&self, h: i64, out: &mut [u8], addr: &mut [u8]) -> fs::Result<(usize, usize)>;

    /// Releases `h`. Called when the last file descriptor referring to the
    /// socket goes away, from a context that cannot report an error, so any
    /// failure is the backend's to handle.
    fn close(&self, h: i64);
}

static UNIX_DGRAM_VTABLE: InodeVtable =
    InodeVtable { recvfrom: UnixDgram::recvfrom, bind: UnixStream::bind, ..DEFAULT_INODE_VTABLE };

/// The maximum number of pending messages that we allow before we start overwritting data.
const MAX_QUEUED_DGRAMS: usize = 16;

#[derive(Clone, Default)]
pub struct UnixDgram {
    recv_index: usize,
    buf: [Vec<u8>; MAX_QUEUED_DGRAMS],
    socket_addr: SocketAddr,
}

impl UnixDgram {
    pub fn recvfrom(inode: &mut Inode, msg: &mut Message) -> fs::Result<usize> {
        let socket = inode.data.downcast_mut::<Self>().unwrap();

        // @fixme: zero length dgrams are allowed.
        if socket.buf[socket.recv_index].is_empty() {
            return Err(errno::EWOULDBLOCK);
        }

        let buf = &mut socket.buf[socket.recv_index];
        let len = usize::min(buf.len(), msg.buf.len());
        msg.buf[..len].copy_from_slice(&buf[..len]);
        buf.clear();

        socket.recv_index += 1;
        if socket.recv_index >= socket.buf.len() {
            socket.recv_index = 0;
        }

        Ok(len)
    }

    pub fn bind(inode: &mut Inode, addr: &SocketAddr) -> fs::Result<()> {
        let socket = inode.data.downcast_mut::<Self>().unwrap();
        socket.socket_addr = addr.clone();
        Ok(())
    }
}

static UNIX_STREAM_VTABLE: InodeVtable = InodeVtable {
    recvfrom: UnixStream::recvfrom,
    sendto: UnixStream::sendto,
    bind: UnixStream::bind,
    ..DEFAULT_INODE_VTABLE
};

#[derive(Clone, Default)]
pub struct UnixStream {
    stream: fs::Stream,
    socket_addr: SocketAddr,
}

impl UnixStream {
    pub fn recvfrom(inode: &mut Inode, msg: &mut Message) -> fs::Result<usize> {
        let socket = inode.data.downcast_mut::<Self>().unwrap();
        socket.stream.read(msg.buf)
    }

    pub fn sendto(inode: &mut Inode, msg: &Message) -> fs::Result<usize> {
        let socket = inode.data.downcast_mut::<Self>().unwrap();
        socket.stream.write(msg.buf)
    }

    pub fn bind(inode: &mut Inode, addr: &SocketAddr) -> fs::Result<()> {
        let socket = inode.data.downcast_mut::<Self>().unwrap();
        socket.socket_addr = addr.clone();
        Ok(())
    }
}

/// An `AF_INET`/`SOCK_STREAM` socket, served entirely by the installed
/// [`NetBackend`].
///
/// The socket owns nothing but the handle the backend gave it: there is no
/// in-emulator byte queue, because every `send`/`recv` is a round trip to the
/// host. `handle` is `None` until `connect` succeeds, which is what makes
/// `send`/`recv` on an unconnected socket `ENOTCONN`; `net` is `None` when the
/// embedder installed no backend at all, which makes `connect` itself
/// `EACCES`.
pub struct TcpSocket {
    net: Option<Rc<dyn NetBackend>>,
    handle: Option<i64>,
    socket_addr: SocketAddr,
}

impl TcpSocket {
    fn new(net: Option<Rc<dyn NetBackend>>) -> Self {
        Self { net, handle: None, socket_addr: SocketAddr::default() }
    }

    fn data(inode: &mut Inode) -> fs::Result<&mut Self> {
        inode.data.downcast_mut::<Self>().ok_or(errno::ENOTSOCK)
    }

    /// The backend and handle of a connected socket.
    ///
    /// Both are copied out rather than borrowed, so that no borrow derived
    /// from the inode's data is still outstanding while the backend runs. (The
    /// caller's `RefMut` on the inode itself is: a host that re-enters the
    /// emulator through the same file descriptor from inside a callback still
    /// panics, exactly as it does for the file-system callbacks.)
    fn connected(inode: &mut Inode) -> fs::Result<(Rc<dyn NetBackend>, i64)> {
        let socket = Self::data(inode)?;
        let handle = socket.handle.ok_or(errno::ENOTCONN)?;
        let net = socket.net.clone().ok_or(errno::ENOTCONN)?;
        Ok((net, handle))
    }

    /// Opens the host-side connection described by `addr` (the first
    /// `addr_len` bytes are the guest's `sockaddr`).
    ///
    /// `EACCES` when no backend is installed: a guest that is not meant to
    /// have network access must be told so, rather than silently succeeding
    /// against nothing.
    pub fn connect(inode: &mut Inode, addr: &SocketAddr, addr_len: usize) -> fs::Result<()> {
        let net = {
            let socket = Self::data(inode)?;
            if socket.handle.is_some() {
                return Err(errno::EISCONN);
            }
            socket.net.clone().ok_or(errno::EACCES)?
        };

        let len = usize::min(addr_len, SOCKET_STORAGE_SIZE);
        let handle = net.connect(AF_INET, SOCK_STREAM, &addr.addr[..len])?;

        let socket = Self::data(inode)?;
        socket.handle = Some(handle);
        socket.socket_addr = addr.clone();
        Ok(())
    }

    pub fn recvfrom_tcp(inode: &mut Inode, msg: &mut Message) -> fs::Result<usize> {
        let (net, handle) = Self::connected(inode)?;
        let len = net.recv(handle, msg.buf)?;
        // A backend that over-reports would otherwise hand the guest bytes
        // from past the end of the buffer it asked for.
        Ok(usize::min(len, msg.buf.len()))
    }

    pub fn sendto_tcp(inode: &mut Inode, msg: &Message) -> fs::Result<usize> {
        let (net, handle) = Self::connected(inode)?;
        let len = net.send(handle, msg.buf)?;
        Ok(usize::min(len, msg.buf.len()))
    }

    pub fn bind_tcp(inode: &mut Inode, addr: &SocketAddr) -> fs::Result<()> {
        let socket = Self::data(inode)?;
        socket.socket_addr = addr.clone();
        Ok(())
    }
}

/// Releases the host-side connection when the last file descriptor referring
/// to this socket goes away.
///
/// The inode is what owns the handle, so its destruction is the one event that
/// covers every way a socket can go: an explicit `close`, a descriptor being
/// reassigned by `dup2`, or the file table being torn down. There is nowhere
/// to report a failure to, which is why [`NetBackend::close`] returns nothing.
impl Drop for TcpSocket {
    fn drop(&mut self) {
        if let (Some(net), Some(handle)) = (self.net.as_ref(), self.handle.take()) {
            net.close(handle);
        }
    }
}

static TCP_SOCKET_VTABLE: InodeVtable = InodeVtable {
    connect: TcpSocket::connect,
    recvfrom: TcpSocket::recvfrom_tcp,
    sendto: TcpSocket::sendto_tcp,
    // Not inherited from `UNIX_STREAM_VTABLE`: that one downcasts the inode's
    // data to `UnixStream`, which a `TcpSocket` no longer is.
    bind: TcpSocket::bind_tcp,
    ..UNIX_STREAM_VTABLE
};

/// An `AF_INET`/`SOCK_DGRAM` socket, served entirely by the installed
/// [`NetBackend`].
///
/// Connectionless, so there is no single moment like `TcpSocket::connect` to
/// hang the host-side handle on -- `UDP_SOCKET_VTABLE` does not override
/// `connect` at all (the default `ENOTSOCK`, unchanged from the AF_UNIX
/// datagram behaviour this replaces). Instead the handle is opened lazily,
/// on whichever of `sendto`/`recvfrom` the guest calls first: see
/// [`UdpSocket::handle`]. The backend's `connect` leg is reused for that
/// open with `kind = SOCK_DGRAM` and an empty address, since there is no
/// destination to fix at open time -- every `sendto` still carries its own,
/// which is what buys the per-datagram destination check the spec requires;
/// the lazily-opened handle is just an ephemeral local socket, not a
/// connection to anywhere.
pub struct UdpSocket {
    net: Option<Rc<dyn NetBackend>>,
    handle: Option<i64>,
    socket_addr: SocketAddr,
}

impl UdpSocket {
    fn new(net: Option<Rc<dyn NetBackend>>) -> Self {
        Self { net, handle: None, socket_addr: SocketAddr::default() }
    }

    fn data(inode: &mut Inode) -> fs::Result<&mut Self> {
        inode.data.downcast_mut::<Self>().ok_or(errno::ENOTSOCK)
    }

    /// The backend and handle for this socket, opening the host-side handle
    /// on first use. `EACCES` when no backend is installed, matching
    /// `TcpSocket::connect` -- a guest that is not meant to have network
    /// access must be told so rather than silently succeeding against
    /// nothing.
    fn handle(inode: &mut Inode) -> fs::Result<(Rc<dyn NetBackend>, i64)> {
        let socket = Self::data(inode)?;
        if let Some(h) = socket.handle {
            let net = socket.net.clone().ok_or(errno::EACCES)?;
            return Ok((net, h));
        }

        let net = socket.net.clone().ok_or(errno::EACCES)?;
        let h = net.connect(AF_INET, SOCK_DGRAM, &[])?;

        let socket = Self::data(inode)?;
        socket.handle = Some(h);
        Ok((net, h))
    }

    pub fn sendto_udp(inode: &mut Inode, msg: &Message) -> fs::Result<usize> {
        // `Message` carries no separate address length (unlike `connect`'s
        // explicit `addr_len`), so the destination the backend sees is
        // always the full fixed-size `SocketAddr` buffer, zero-padded past
        // whatever the guest actually wrote -- harmless, since nothing reads
        // past the family/port/address that matter.
        let dest = msg.address.as_ref().ok_or(errno::EDESTADDRREQ)?;
        let (net, handle) = Self::handle(inode)?;
        let len = net.sendto(handle, msg.buf, &dest.addr)?;
        Ok(usize::min(len, msg.buf.len()))
    }

    pub fn recvfrom_udp(inode: &mut Inode, msg: &mut Message) -> fs::Result<usize> {
        let (net, handle) = Self::handle(inode)?;
        let mut addr_buf = [0u8; SOCKET_STORAGE_SIZE];
        let (len, addr_len) = net.recvfrom(handle, msg.buf, &mut addr_buf)?;

        if let Some(dest) = msg.address.as_mut() {
            let alen = usize::min(addr_len, SOCKET_STORAGE_SIZE);
            dest.addr[..alen].copy_from_slice(&addr_buf[..alen]);
        }

        Ok(usize::min(len, msg.buf.len()))
    }

    pub fn bind_udp(inode: &mut Inode, addr: &SocketAddr) -> fs::Result<()> {
        let socket = Self::data(inode)?;
        socket.socket_addr = addr.clone();
        Ok(())
    }
}

/// Releases the host-side handle when the last file descriptor referring to
/// this socket goes away. See `TcpSocket`'s `Drop` impl -- same rationale,
/// same coverage of `close`/`dup2`/file-table teardown.
impl Drop for UdpSocket {
    fn drop(&mut self) {
        if let (Some(net), Some(handle)) = (self.net.as_ref(), self.handle.take()) {
            net.close(handle);
        }
    }
}

static UDP_SOCKET_VTABLE: InodeVtable = InodeVtable {
    recvfrom: UdpSocket::recvfrom_udp,
    sendto: UdpSocket::sendto_udp,
    bind: UdpSocket::bind_udp,
    ..DEFAULT_INODE_VTABLE
};

static NETLINK_VTABLE: InodeVtable = InodeVtable {
    recvfrom: |_, _| Err(errno::ENOSYS),
    sendto: |_, _| Err(errno::ENOSYS),
    ..UNIX_STREAM_VTABLE
};

pub struct SocketFs {
    fs: Rc<RefCell<TempFs>>,
    /// Where network sockets get their host access from, if an embedder
    /// installed one. Handed to each socket as it is created, so a backend
    /// installed later does not retroactively connect sockets that already
    /// exist.
    net: Option<Rc<dyn NetBackend>>,
}

impl SocketFs {
    pub fn create(dev_id: usize) -> Self {
        Self { fs: TempFs::create(dev_id), net: None }
    }

    /// Installs the host-side network provider, replacing any previous one.
    /// See [`NetBackend`]; without this call the guest has no network access.
    pub fn set_net_backend(&mut self, net: Rc<dyn NetBackend>) {
        self.net = Some(net);
    }

    /// Drops the reference to the host-side network provider.
    ///
    /// For embedders that must guarantee no callback runs after some point
    /// (typically their own teardown): sockets created before this still hold
    /// their own reference and will still close through it, so this only stops
    /// *new* sockets from reaching the backend.
    pub fn clear_net_backend(&mut self) {
        self.net = None;
    }

    pub fn create_socket(&mut self, family: u64, kind: u64, protocol: u64) -> fs::Result<InodeRef> {
        macro_rules! af_not_supported {
            ($name:literal) => {{
                // tracing::warn!(concat!("Address family not supported {}", $name));
                return Err(errno::EAFNOSUPPORT);
            }};
        }

        match family {
            AF_UNSPEC => af_not_supported!("AF_UNSPEC"),
            AF_UNIX => self.create_unix_socket(kind, protocol),
            AF_INET => self.create_ipv4_socket(kind, protocol),
            AF_INET6 => af_not_supported!("AF_INET6"),
            AF_NETLINK => self.create_netlink_socket(kind, protocol),
            _ => af_not_supported!("Unknown"),
        }
    }

    fn create_unix_socket(&mut self, kind: u64, protocol: u64) -> fs::Result<InodeRef> {
        if protocol != AF_UNSPEC && protocol != AF_UNIX {
            return Err(errno::EPROTONOSUPPORT);
        }

        let (data, vtable): (Box<dyn std::any::Any>, &InodeVtable) = match kind {
            SOCK_STREAM => (Box::<UnixStream>::default(), &UNIX_STREAM_VTABLE),
            SOCK_DGRAM => (Box::<UnixDgram>::default(), &UNIX_DGRAM_VTABLE),
            _ => return Err(errno::ESOCKTNOSUPPORT),
        };

        self.create_socket_with(data, vtable)
    }

    fn create_ipv4_socket(&mut self, kind: u64, protocol: u64) -> fs::Result<InodeRef> {
        if protocol != 0 {
            return Err(errno::EPROTONOSUPPORT);
        }

        let (data, vtable): (Box<dyn std::any::Any>, &InodeVtable) = match kind {
            SOCK_STREAM => (Box::new(TcpSocket::new(self.net.clone())), &TCP_SOCKET_VTABLE),
            SOCK_DGRAM => (Box::new(UdpSocket::new(self.net.clone())), &UDP_SOCKET_VTABLE),
            _ => return Err(errno::ESOCKTNOSUPPORT),
        };

        self.create_socket_with(data, vtable)
    }

    // @fixme
    fn create_netlink_socket(&mut self, _kind: u64, _protocol: u64) -> fs::Result<InodeRef> {
        self.create_socket_with(Box::<UnixStream>::default(), &NETLINK_VTABLE)
    }

    fn create_socket_with(
        &mut self,
        data: Box<dyn std::any::Any>,
        vtable: &'static InodeVtable,
    ) -> fs::Result<InodeRef> {
        let inode = self.fs.borrow_mut().alloc_inode()?;
        {
            let mut inode = inode.borrow_mut();
            inode.data = data;
            inode.vtable = vtable;
            inode.kind = FileKind::Socket;
        }
        Ok(inode)
    }

    pub fn alloc_file(&mut self, inode: InodeRef) -> fs::Result<file::ActiveFile> {
        Ok(Rc::new(RefCell::new(file::ActiveFileData::new(vec![], inode))))
    }
}

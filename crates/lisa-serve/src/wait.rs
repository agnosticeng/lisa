//! Request readiness without an idle delay before accepting or dequeuing work.
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SendError, Sender};
use std::time::Duration;

/// External stop flags remain observable without requiring a socket wakeup.
/// This bounds shutdown checks, not the time before incoming work is handled.
pub(super) const STOP_CHECK: Duration = Duration::from_millis(50);

pub(super) struct WakeSender<T> {
    sender: Sender<T>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl<T> Clone for WakeSender<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            wake: self.wake.clone(),
        }
    }
}

impl<T> WakeSender<T> {
    pub(super) fn new(sender: Sender<T>, wake: Option<Arc<dyn Fn() + Send + Sync>>) -> Self {
        Self { sender, wake }
    }

    pub(super) fn send(&self, item: T) -> Result<(), SendError<T>> {
        super::application::trace_http("serve.http.enqueue");
        self.sender.send(item)?;
        // Notify only after the item is visible. A host can therefore drain its
        // request queue immediately, including when it was asleep on UI input.
        if let Some(wake) = &self.wake {
            wake();
        }
        Ok(())
    }
}

pub(super) fn receive_first<T>(receiver: &Receiver<T>, wait: Option<Duration>) -> Option<T> {
    let item = match wait {
        Some(timeout) => receiver.recv_timeout(timeout).ok(),
        None => receiver.try_recv().ok(),
    };
    if item.is_some() {
        super::application::trace_http("serve.http.dequeue");
    }
    item
}

pub(super) fn ready_wave<T>(first: T, receiver: &Receiver<T>, maximum: usize) -> Vec<T> {
    let mut wave = vec![first];
    while wave.len() < maximum.max(1) {
        match receiver.try_recv() {
            Ok(item) => {
                super::application::trace_http("serve.http.dequeue");
                wave.push(item);
            }
            Err(_) => break,
        }
    }
    wave
}

#[cfg(unix)]
fn readable(listener: &TcpListener, timeout: Duration) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        // Round up: a sub-millisecond remainder must not turn into a busy poll.
        let millis = left
            .as_millis()
            .saturating_add(u128::from(!left.subsec_nanos().is_multiple_of(1_000_000)));
        let mut descriptor = libc::pollfd {
            fd: listener.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `descriptor` is one valid pollfd (nfds = 1) on the stack, and the
        // borrowed listener keeps its fd open throughout the bounded wait.
        let result = unsafe { libc::poll(&mut descriptor, 1, millis.min(i32::MAX as u128) as i32) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                if std::time::Instant::now() >= deadline {
                    return Ok(false);
                }
                continue;
            }
            return Err(error);
        }
        if result == 0 {
            return Ok(false);
        }
        if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(io::Error::other("HTTP listener readiness failed"));
        }
        return Ok(descriptor.revents & libc::POLLIN != 0);
    }
}

// The Metal server currently runs on Unix targets. Keep the previous bounded
// compatibility behavior elsewhere until a native socket wait is implemented.
#[cfg(not(unix))]
fn readable(_listener: &TcpListener, timeout: Duration) -> io::Result<bool> {
    std::thread::sleep(timeout);
    Ok(true)
}

pub(super) fn accept(
    listener: Arc<TcpListener>,
    stop: Arc<AtomicBool>,
    handler: impl Fn(TcpStream) + Send + 'static,
) -> io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("lisa-http-accept".into())
        .spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match readable(&listener, STOP_CHECK) {
                    Ok(false) => continue,
                    Ok(true) if stop.load(Ordering::Relaxed) => break,
                    Ok(true) => match listener.accept() {
                        Ok((stream, _)) => handler(stream),
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(error) => {
                            eprintln!("[serve] accept error: {error}");
                            std::thread::sleep(STOP_CHECK);
                        }
                    },
                    Err(error) => {
                        eprintln!("[serve] readiness error: {error}");
                        std::thread::sleep(STOP_CHECK);
                    }
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::mpsc;

    #[cfg(unix)]
    #[test]
    fn socket_readiness_wakes_before_its_idle_timeout() -> io::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let barrier = Arc::new(Barrier::new(2));
        let (ready, observed) = mpsc::channel();
        let worker_barrier = barrier.clone();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            // Deliberately much longer than the test's receive deadline: the
            // socket event, not an idle timeout, must release this wait.
            let result = readable(&listener, Duration::from_secs(5));
            ready.send(result).unwrap();
            listener
        });
        barrier.wait();
        let client = TcpStream::connect(address)?;
        assert!(observed.recv_timeout(Duration::from_secs(1)).unwrap()?);
        let listener = worker.join().unwrap();
        let (accepted, _) = listener.accept()?;
        drop((client, accepted));
        assert!(!readable(&listener, Duration::ZERO)?);
        Ok(())
    }

    #[test]
    fn queued_request_wakes_wait_and_nonblocking_poll_stays_nonblocking() {
        let (sender, receiver) = mpsc::channel();
        assert_eq!(receive_first(&receiver, None), None);
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = barrier.clone();
        let (result, observed) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            result
                .send(receive_first(&receiver, Some(Duration::from_secs(5))))
                .unwrap();
        });
        barrier.wait();
        sender.send(17).unwrap();
        assert_eq!(
            observed.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(17)
        );
        worker.join().unwrap();
    }

    #[test]
    fn ready_waves_preserve_order_and_do_not_wait_to_fill() {
        let (sender, receiver) = mpsc::channel();
        for item in 0..5 {
            sender.send(item).unwrap();
        }
        let first = receive_first(&receiver, None).unwrap();
        assert_eq!(ready_wave(first, &receiver, 2), [0, 1]);
        let first = receive_first(&receiver, None).unwrap();
        assert_eq!(ready_wave(first, &receiver, 0), [2]);
        let first = receive_first(&receiver, None).unwrap();
        assert_eq!(ready_wave(first, &receiver, 8), [3, 4]);
        assert_eq!(receive_first(&receiver, None), None);
        drop(sender);
        assert_eq!(receive_first(&receiver, Some(Duration::from_secs(5))), None);
    }

    #[test]
    fn host_notification_follows_enqueue_and_failed_sends_do_not_wake() {
        let (sender, receiver) = mpsc::channel();
        let (notice, notices) = mpsc::channel();
        let sender = WakeSender::new(sender, Some(Arc::new(move || notice.send(()).unwrap())));
        sender.send(23).unwrap();
        notices.recv().unwrap();
        assert_eq!(receiver.try_recv().unwrap(), 23);
        let cloned = sender.clone();
        cloned.send(29).unwrap();
        notices.recv().unwrap();
        assert_eq!(receiver.try_recv().unwrap(), 29);
        drop(receiver);
        assert!(sender.send(31).is_err());
        assert!(notices.try_recv().is_err());
    }

    #[test]
    fn idle_accept_stop_releases_port_and_restart_accepts_requests() -> io::Result<()> {
        let listener = Arc::new(TcpListener::bind("127.0.0.1:0")?);
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let (accepted, events) = mpsc::channel();
        let worker = accept(listener.clone(), stop.clone(), move |_| {
            accepted.send(()).unwrap();
        })?;
        let client = TcpStream::connect(address)?;
        events.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(client);
        stop.store(true, Ordering::Relaxed);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            worker.join().unwrap();
            done.send(()).unwrap();
        });
        finished.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            events.try_recv().is_err(),
            "shutdown did not create a connection"
        );
        drop(listener);
        let listener = Arc::new(TcpListener::bind(address)?);
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let (accepted, events) = mpsc::channel();
        let worker = accept(listener.clone(), stop.clone(), move |_| {
            accepted.send(()).unwrap();
        })?;
        let client = TcpStream::connect(address)?;
        events.recv_timeout(Duration::from_secs(1)).unwrap();
        stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();
        drop((client, listener));
        Ok(())
    }

    #[test]
    fn stopped_accept_does_not_dispatch_an_already_ready_connection() -> io::Result<()> {
        let listener = Arc::new(TcpListener::bind("127.0.0.1:0")?);
        listener.set_nonblocking(true)?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let stop = Arc::new(AtomicBool::new(true));
        let worker = accept(listener, stop, |_| panic!("request dispatched after stop"))?;
        worker.join().unwrap();
        drop(client);
        Ok(())
    }
}

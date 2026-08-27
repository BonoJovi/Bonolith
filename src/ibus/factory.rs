/// IBus Factory D-Bus interface and service startup.
///
/// Implements org.freedesktop.IBus.Factory which creates engine instances
/// on demand from the IBus daemon.
use log::{info, warn};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zbus::{connection::Builder, interface, object_server::SignalEmitter, Connection};

use super::config::BonolithConfig;
use super::engine_impl::BonolithEngine;

/// Shared "force the focused engine into 日本語ON" window.
///
/// `Some(deadline)` means: any engine that gains focus or receives a key
/// before `deadline` should flip itself enabled. This is how the
/// word-register dialog turns the IME on without injecting synthetic keys
/// (xdotool is X11-only and absent on Wayland) — it calls
/// `org.bonolith.Control.ForceEnable` over the session bus, and the engine
/// instances apply the request through IBus' own focus/key callbacks,
/// which fire identically on X11 and Wayland.
pub type ForceEnable = Arc<Mutex<Option<Instant>>>;

/// Length of the force-on window opened by one `ForceEnable` call.
/// The dialog refreshes it on every field focus-in, so this only needs to
/// outlive the gap between focus events; it also bounds how long a crashed
/// dialog could leave the flag set.
const FORCE_WINDOW: Duration = Duration::from_secs(5);

/// IBus Factory — creates engine instances on request from IBus daemon.
pub struct BonolithFactory {
    config: BonolithConfig,
    force: ForceEnable,
}

impl BonolithFactory {
    pub fn new(force: ForceEnable) -> Self {
        Self {
            config: BonolithConfig::load(),
            force,
        }
    }
}

/// Session-bus control surface used by the word-register/-edit dialog.
///
/// Exposed at `org.bonolith.Control` / `/org/bonolith/Control` so a plain
/// session-bus client (the GTK dialog) can reach it without knowing the
/// IBus private-bus address.
pub struct BonolithControl {
    force: ForceEnable,
}

impl BonolithControl {
    pub fn new(force: ForceEnable) -> Self {
        Self { force }
    }
}

#[interface(name = "org.bonolith.Control")]
impl BonolithControl {
    /// Open a short force-on window: the next engine to focus or receive a
    /// key flips itself 日本語ON. Idempotent and order-independent — safe to
    /// call on every field focus-in.
    async fn force_enable(&self) {
        *self.force.lock().unwrap() = Some(Instant::now() + FORCE_WINDOW);
        info!("Bonolith Control: ForceEnable (+{:?})", FORCE_WINDOW);
    }

    /// Close the force-on window early (dialog is closing).
    async fn force_enable_clear(&self) {
        *self.force.lock().unwrap() = None;
        info!("Bonolith Control: ForceEnableClear");
    }
}

/// Monotonic engine sequence — mirrors the C IBus convention of
/// `/org/freedesktop/IBus/Engine/{N}` so every input context gets its own
/// object path. Sharing a path across contexts made the second CreateEngine
/// a silent no-op (zbus `ObjectServer::at` returns `Ok(false)`), which
/// multiplexed every window onto the first engine and cross-wired state.
static ENGINE_SEQ: AtomicU64 = AtomicU64::new(0);

#[interface(name = "org.freedesktop.IBus.Factory")]
impl BonolithFactory {
    /// Called by IBus daemon to create a new engine instance.
    async fn create_engine(
        &self,
        #[zbus(signal_emitter)] _emitter: SignalEmitter<'_>,
        #[zbus(connection)] connection: &Connection,
        engine_name: &str,
    ) -> zbus::fdo::Result<zbus::zvariant::OwnedObjectPath> {
        let id = ENGINE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = format!("/org/freedesktop/IBus/Engine/{}", id);
        info!("Bonolith Factory: CreateEngine({}) → {}", engine_name, path);

        let engine = BonolithEngine::new(&self.config, self.force.clone(), path.clone());

        let inserted = connection
            .object_server()
            .at(path.as_str(), engine)
            .await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;

        if !inserted {
            warn!(
                "Bonolith Factory: object path {} already registered — engine sequence collision",
                path
            );
            return Err(zbus::fdo::Error::Failed(format!(
                "engine path {} already registered",
                path
            )));
        }

        Ok(path
            .try_into()
            .map_err(|e: zbus::zvariant::Error| zbus::fdo::Error::Failed(e.to_string()))?)
    }
}

/// Resolve the IBus D-Bus address.
///
/// IBus runs its own private bus separate from the session bus.
/// The address is found via:
/// 1. `IBUS_ADDRESS` environment variable (set by IBus when launching engines)
/// 2. Bus file in `~/.config/ibus/bus/`
fn get_ibus_address() -> Option<String> {
    // Check environment variable first
    if let Ok(addr) = std::env::var("IBUS_ADDRESS") {
        if !addr.is_empty() {
            return Some(addr);
        }
    }

    // Fall back to reading the IBus bus file (use the most recently modified one)
    let config_dir = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join(".config")
        });
    let bus_dir = config_dir.join("ibus").join("bus");

    let mut newest: Option<(std::time::SystemTime, String)> = None;
    let entries = std::fs::read_dir(&bus_dir).ok()?;
    for entry in entries.flatten() {
        let mtime = entry.metadata().ok()?.modified().ok()?;
        if let Ok(contents) = std::fs::read_to_string(entry.path()) {
            for line in contents.lines() {
                if let Some(addr) = line.strip_prefix("IBUS_ADDRESS=") {
                    if !addr.is_empty() {
                        if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                            newest = Some((mtime, addr.to_string()));
                        }
                    }
                }
            }
        }
    }

    newest.map(|(_, addr)| addr)
}

/// Start the IBus service: register Factory on the IBus private bus, plus
/// the `org.bonolith.Control` surface on the session bus.
///
/// Returns both connections; the caller must keep them alive for the
/// process lifetime. The control connection is dropped (and its name
/// released) only when the returned tuple is dropped.
pub async fn start_ibus_service() -> zbus::Result<(Connection, Connection)> {
    info!("Bonolith: Starting IBus service...");

    let force: ForceEnable = Arc::new(Mutex::new(None));

    let connection = if let Some(addr) = get_ibus_address() {
        info!("Bonolith: Connecting to IBus bus at {}", addr);
        Builder::address(addr.as_str())?
            .name("org.freedesktop.IBus.Bonolith")?
            .serve_at(
                "/org/freedesktop/IBus/Factory",
                BonolithFactory::new(force.clone()),
            )?
            .build()
            .await?
    } else {
        info!("Bonolith: IBus address not found, falling back to session bus");
        Builder::session()?
            .name("org.freedesktop.IBus.Bonolith")?
            .serve_at(
                "/org/freedesktop/IBus/Factory",
                BonolithFactory::new(force.clone()),
            )?
            .build()
            .await?
    };

    // Control surface on the session bus, reachable by the GTK dialog.
    let control = Builder::session()?
        .name("org.bonolith.Control")?
        .serve_at("/org/bonolith/Control", BonolithControl::new(force))?
        .build()
        .await?;

    info!("Bonolith: IBus service registered successfully");
    info!("Bonolith: org.bonolith.Control registered on session bus");
    info!("Bonolith: Waiting for IBus daemon requests...");

    Ok((connection, control))
}

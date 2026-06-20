//! IBus integration layer
//!
//! Implements the org.freedesktop.IBus.Engine D-Bus interface
//! using the zbus crate for pure-Rust D-Bus communication.
//!
//! Architecture:
//! - BonolithEngine: implements org.freedesktop.IBus.Engine (key events, preedit, commit)
//! - BonolithFactory: implements org.freedesktop.IBus.Factory (engine creation)
//! - Component XML: registration file for IBus daemon

mod config;
mod engine_impl;
mod factory;
pub mod keymap;

pub use factory::start_ibus_service;

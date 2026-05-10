mod ibus;

use log::info;

use jaim::core::dictionary::Dictionary;
use std::path::PathBuf;

fn init_logging() {
    // TODO: File logging disabled due to I/O latency. Re-enable after
    // moving file writes to a background thread.
    // use std::fs;
    // let log_dir = std::env::var("XDG_CACHE_HOME")
    //     .map(PathBuf::from)
    //     .unwrap_or_else(|_| {
    //         let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    //         PathBuf::from(home).join(".cache")
    //     })
    //     .join("jaim");
    // let _ = fs::create_dir_all(&log_dir);
    // let log_path = log_dir.join("jaim.log");
    // let file = fs::OpenOptions::new()
    //     .create(true)
    //     .append(true)
    //     .open(&log_path)
    //     .ok()
    //     .map(|f| std::sync::Mutex::new(std::io::LineWriter::new(f)));

    // Stderr: controlled by RUST_LOG (default: info).
    env_logger::Builder::from_default_env()
        .format(move |buf, record| {
            use std::io::Write as _;
            writeln!(buf, "[{} {}] {}", record.level(), record.target(), record.args())
        })
        .init();
}

fn print_usage() {
    eprintln!("Usage: jaim [COMMAND]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  (none)              Start the IBus engine");
    eprintln!("  export <file>       Export dictionary to a JSON file");
    eprintln!("  import <file>       Import dictionary from a JSON file");
    eprintln!("  llm <on|off|status> Toggle the local LLM server (jaim-llm-server)");
    eprintln!("  help                Show this help message");
}

fn llm_systemctl(args: &[&str]) -> std::io::Result<std::process::ExitStatus> {
    std::process::Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
}

fn llm_on() {
    match llm_systemctl(&["enable", "--now", "jaim-llm-server.service"]) {
        Ok(s) if s.success() => {
            println!(
                "LLM enabled. jaim-llm-server.service started and will start on login.\n\
                 If you ran `jaim llm off` earlier in this session (or the\n\
                 server crashed), the running IM has stopped sending scoring\n\
                 requests. Restart it with `ibus-daemon -drx` so the engine\n\
                 reconnects."
            );
        }
        Ok(s) => {
            eprintln!("Error: systemctl exited with status {}", s);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error: could not run systemctl: {}", e);
            std::process::exit(1);
        }
    }
}

fn llm_off() {
    let _ = llm_systemctl(&["stop", "jaim-llm-server.service"]);
    match llm_systemctl(&["disable", "jaim-llm-server.service"]) {
        Ok(_) => {
            println!(
                "LLM disabled. jaim-llm-server.service stopped.\n\
                 The running IM detects the missing server on the next\n\
                 keystroke and falls back to the dictionary-only ranker —\n\
                 no IM restart needed."
            );
        }
        Err(e) => {
            eprintln!("Error: could not run systemctl: {}", e);
            std::process::exit(1);
        }
    }
}

fn llm_status() {
    use std::process::Command;
    let read = |action: &str| -> String {
        Command::new("systemctl")
            .args(["--user", action, "jaim-llm-server.service"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    };
    println!("LLM service status:");
    println!("  active:  {}", read("is-active"));
    println!("  enabled: {}", read("is-enabled"));
}

#[tokio::main]
async fn main() {
    init_logging();

    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("export") => {
            let path = match args.get(2) {
                Some(p) => PathBuf::from(p),
                None => {
                    eprintln!("Error: export requires a file path");
                    eprintln!("Usage: jaim export <file>");
                    std::process::exit(1);
                }
            };
            let (dict, loaded) = match Dictionary::with_default_store() {
                Ok(d) => {
                    let n = d.user_entries().len();
                    (d, n)
                }
                Err(e) => {
                    eprintln!(
                        "Warning: could not open user dictionary store; \
                         exporting builtin entries only.\n  {}",
                        e
                    );
                    (Dictionary::new(), 0)
                }
            };
            match dict.export(&path) {
                Ok(()) => {
                    println!(
                        "Exported dictionary to {} (builtin + {} user entries)",
                        path.display(),
                        loaded
                    );
                }
                Err(e) => {
                    eprintln!("Error: failed to export dictionary: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some("import") => {
            let path = match args.get(2) {
                Some(p) => PathBuf::from(p),
                None => {
                    eprintln!("Error: import requires a file path");
                    eprintln!("Usage: jaim import <file>");
                    std::process::exit(1);
                }
            };
            let mut dict = match Dictionary::with_default_store() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!(
                        "Error: could not open user dictionary store.\n  {}\n\
                         Cannot import without a working store.",
                        e
                    );
                    std::process::exit(1);
                }
            };
            match dict.import(&path) {
                Ok(added) => {
                    if let Err(e) = dict.sync_user_entries_to_store() {
                        eprintln!("Error: failed to persist user dictionary: {}", e);
                        std::process::exit(1);
                    }
                    println!(
                        "Imported {} new entries from {}",
                        added,
                        path.display()
                    );
                }
                Err(e) => {
                    eprintln!("Error: failed to import dictionary: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some("llm") => {
            match args.get(2).map(|s| s.as_str()) {
                Some("on") => llm_on(),
                Some("off") => llm_off(),
                Some("status") | None => llm_status(),
                Some(other) => {
                    eprintln!("Error: unknown llm subcommand '{}'", other);
                    eprintln!("Usage: jaim llm <on|off|status>");
                    std::process::exit(1);
                }
            }
        }
        Some("help") | Some("--help") | Some("-h") => {
            print_usage();
        }
        Some(cmd) => {
            eprintln!("Error: unknown command '{}'", cmd);
            print_usage();
            std::process::exit(1);
        }
        None => {
            info!("JaIM - Japanese AI-powered Input Method");
            info!("Starting JaIM engine...");

            match ibus::start_ibus_service().await {
                Ok(connection) => {
                    info!("JaIM: IBus service started successfully");
                    loop {
                        connection.monitor_activity().await;
                    }
                }
                Err(e) => {
                    eprintln!("JaIM: Failed to start IBus service: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

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
    eprintln!("  help                Show this help message");
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

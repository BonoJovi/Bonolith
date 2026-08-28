# Bonolith Project Context

## Project

- **Type**: Japanese AI Input Method for Linux (IBus + Fcitx5)
- **Purpose**: LLM-based intelligent Japanese input method
- **Version**: v3.1.10
- **Branch**: `dev` for development, `main` for releases

## Essential Rules

1. **All code changes on `dev` branch** — merge to `main` for releases only
2. **Version files**: Update all 6 files simultaneously — `Cargo.toml`, `README.md`, `data/bonolith.xml`, `fcitx5/bonolith-addon.conf`, `fcitx5/CMakeLists.txt`, `CLAUDE.md` (the `**Version**` line below). `cargo build --release` regenerates `Cargo.lock` — commit it too.
3. **No `unwrap()` in production Rust** — always use `Result<T, E>`
4. **Commit messages**: English, conventional format `type(scope): description`
5. **Release**: Use `/release`

## Install & Restart

Install is done via `scripts/install.sh`, which handles both IBus and
Fcitx5, restarts daemons, and internally escalates for `/usr/bin` writes.
**Do not** prefix `sudo` and **do not** invoke `cp`/`pkill`/`ibus-daemon`
directly — the script is the only supported path.

```bash
# Rust build (always required after src/ changes)
cargo build --release

# Fcitx5 addon rebuild (only when fcitx5/*.cpp or CMakeLists changed)
(cd fcitx5/build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make)

# Install both frontends + restart daemons
./scripts/install.sh
```

## Commands

```bash
cargo build --release  # Build
cargo test             # Tests
```

## On-Demand Context

Load with `@` when needed:
- Developer Profile: `@.ai-context/shared/developer/YOSHIHIRO_NAKAHARA_PROFILE.md`
- Methodology: `@.ai-context/shared/methodology/AI_COLLABORATION.md`
- Insights: `@.ai-context/shared/insights/INSIGHTS_OVERVIEW.md`

# Bonolith Project Context

## Project

- **Type**: Japanese AI Input Method for Linux (IBus + Fcitx5)
- **Purpose**: LLM-based intelligent Japanese input method
- **Version**: v3.0.1
- **Branch**: `dev` for development, `main` for releases

## Essential Rules

1. **All code changes on `dev` branch** — merge to `main` for releases only
2. **Version files**: Update all 5 files simultaneously — `Cargo.toml`, `README.md`, `data/bonolith.xml`, `fcitx5/bonolith-addon.conf`, `fcitx5/CMakeLists.txt`
3. **No `unwrap()` in production Rust** — always use `Result<T, E>`
4. **Commit messages**: English, conventional format `type(scope): description`
5. **Release**: Use `/release`

## Install & Restart

```bash
# Build
cargo build --release

# IBus — install and restart
sudo pkill -f ibus-daemon && sleep 1 && sudo rm -f /usr/bin/ibus-engine-bonolith && sudo cp target/release/bonolith /usr/bin/ibus-engine-bonolith && sleep 2 && ibus-daemon -drx

# Fcitx5 — build and install
cd fcitx5/build && make clean && cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make && sudo make install && fcitx5 -r -d
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

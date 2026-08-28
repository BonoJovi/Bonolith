# Release Workflow

Release checklist for Bonolith. Run this before creating a release tag.

## Steps

### 1. Version Update

Update version in ALL 6 files simultaneously:
- `Cargo.toml` → `version = "X.Y.Z"`
- `README.md` → Version badge
- `data/bonolith.xml` → `<version>X.Y.Z</version>`
- `fcitx5/bonolith-addon.conf` → `Version=X.Y.Z`
- `fcitx5/CMakeLists.txt` → `project(fcitx5-bonolith VERSION X.Y.Z)`
- `CLAUDE.md` → `- **Version**: vX.Y.Z`

Verify all updated:
```bash
grep -n "X.Y.Z" Cargo.toml README.md data/bonolith.xml fcitx5/bonolith-addon.conf fcitx5/CMakeLists.txt CLAUDE.md
```

### 2. Build & Test

```bash
cargo build --release
cargo test
```

`cargo build` regenerates `Cargo.lock` with the new version — it must be
committed alongside the other bumps in step 3, otherwise `git status`
after the release commit shows a stale lockfile and a follow-up commit
is needed (v3.1.2 shipped without the lock bump and required a
force-push to recover). All tests must pass before proceeding.

### 3. Commit & Merge

```bash
git add Cargo.toml Cargo.lock README.md data/bonolith.xml fcitx5/bonolith-addon.conf fcitx5/CMakeLists.txt CLAUDE.md
git commit -m "release: vX.Y.Z"
git pull origin dev
git push origin dev
git checkout main && git pull origin main && git merge dev
git push origin main
```

### 4. Tag & Push

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 5. Return to dev

```bash
git checkout dev && git merge main
git push origin dev
```

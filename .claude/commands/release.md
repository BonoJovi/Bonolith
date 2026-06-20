# Release Workflow

Release checklist for Bonolith. Run this before creating a release tag.

## Steps

### 1. Version Update

Update version in ALL 5 files simultaneously:
- `Cargo.toml` → `version = "X.Y.Z"`
- `README.md` → Version badge
- `data/bonolith.xml` → `<version>X.Y.Z</version>`
- `fcitx5/bonolith-addon.conf` → `Version=X.Y.Z`
- `fcitx5/CMakeLists.txt` → `project(fcitx5-bonolith VERSION X.Y.Z)`

Verify all updated:
```bash
grep -n "X.Y.Z" Cargo.toml README.md data/bonolith.xml fcitx5/bonolith-addon.conf fcitx5/CMakeLists.txt
```

### 2. Build & Test

```bash
cargo build --release
cargo test
```

All tests must pass before proceeding.

### 3. Commit & Merge

```bash
git add Cargo.toml README.md data/bonolith.xml fcitx5/bonolith-addon.conf fcitx5/CMakeLists.txt
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

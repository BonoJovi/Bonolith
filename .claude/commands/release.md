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

### 3. Commit release bump on dev

```bash
git add Cargo.toml Cargo.lock README.md data/bonolith.xml fcitx5/bonolith-addon.conf fcitx5/CMakeLists.txt CLAUDE.md
git commit -m "release: vX.Y.Z"
git pull origin dev
git push origin dev
```

### 4. Open PR dev → main (Devin review gate)

**Never merge dev → main directly.** Devin's auto-review is wired to
main-bound PRs; a fast-forward merge from the terminal skips the
review entirely, and every fix in the release ships unreviewed
(v3.1.10 hit this — 22 bug fixes were reviewed only after release).

```bash
gh pr create -B main -H dev \
  --title "release: vX.Y.Z" \
  --body "$(git log --oneline main..dev)"
```

Wait for Devin to complete its review before merging. Address
findings on dev with follow-up commits (the PR updates in place),
re-request review if needed.

### 5. Merge PR and tag

Once Devin's review is clean:

```bash
gh pr merge --merge   # or --squash if the release is one logical unit
git checkout main && git pull origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 6. Return to dev

```bash
git checkout dev && git merge main
git push origin dev
```

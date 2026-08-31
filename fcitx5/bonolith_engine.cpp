/// Fcitx5 addon for Bonolith — thin C++ wrapper over the Rust engine.

#include "bonolith_engine.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fcitx-utils/utf8.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputpanel.h>

extern char **environ;

namespace bonolith {

// ── BonolithState (per-InputContext) ─────────────────────────────────────────

BonolithState::BonolithState(BonolithEngine *engine, fcitx::InputContext *ic)
    : engine_(engine), ic_(ic), ctx_(bonolith_context_new()) {}

BonolithState::~BonolithState() {
    if (ctx_) {
        bonolith_context_free(ctx_);
    }
}

void BonolithState::keyEvent(fcitx::KeyEvent &event) {
    if (!ctx_)
        return;

    uint32_t sym = event.rawKey().sym();
    uint32_t state = event.rawKey().states();
    if (event.isRelease())
        state |= (1u << 30); // RELEASE_MASK

    // Snapshot "was showing anything?" so we can skip updateUI when a
    // passthrough key touched nothing on our side — the prior
    // unconditional updateUI() reset the panel every keystroke and
    // could stomp state owned by another addon showing in parallel
    // (Devin PR #3 review #10).
    bool wasShowing = bonolith_is_converting(ctx_)
                      || bonolith_has_preedit(ctx_);
    bool consumed = bonolith_handle_key(ctx_, sym, state);
    if (consumed) {
        event.filterAndAccept();
    }
    bool nowShowing = bonolith_is_converting(ctx_)
                      || bonolith_has_preedit(ctx_);
    if (consumed || wasShowing || nowShowing) {
        updateUI();
    }

    // If this key (conversion start / boundary resize) kicked off a background
    // LLM rerank, poll for its result and refresh the panel when it lands.
    if (bonolith_rerank_pending(ctx_)) {
        scheduleRerankRefresh();
    }
}

void BonolithState::reset() {
    if (ctx_)
        bonolith_reset(ctx_);
    ic_->inputPanel().reset();
    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void BonolithState::commitInput() {
    // Focus-loss handoff. Do NOT send our own commitString here — the
    // Wayland compositor (and GTK4's IM module) auto-finalize the
    // already-shown client preedit into the focused-away client BEFORE
    // this handler runs, so adding our commitString produced duplicated
    // text like "あいあい" (observed on Ghostty and Gnome Terminal at the
    // instant of Alt+Tab). The compositor's finalize is enough; we only
    // need to clear our engine state so a fresh composition starts
    // when the user returns.
    //
    // Trade-off: on hypothetical clients that neither finalize nor
    // notify us before we get here, in-progress preedit is lost. That
    // scenario is unlikely on modern GTK / Qt / Wayland stacks, and the
    // previous "commit-on-deactivate" behaviour was itself lossy on
    // IBus/Ghostty (compositor dropped the preedit AND our commit
    // arrived post-focus, so text vanished). Trading a rare loss for
    // reliable no-duplication is the right call.
    if (!ctx_) return;
    bonolith_reset(ctx_);
    ic_->inputPanel().reset();
    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void BonolithState::updateUI() {
    auto &panel = ic_->inputPanel();
    panel.reset();

    // Single FFI call to get all UI state
    BonolithUiState ui{};
    bonolith_get_ui_state(ctx_, &ui);

    // 1) Check for committed text
    if (ui.committed && ui.committed[0]) {
        ic_->commitString(ui.committed);
    }

    // 2) Update preedit
    if (ui.converting && ui.preedit && ui.preedit[0]) {
        // Conversion mode: show composed text with segment highlighting
        fcitx::Text preedit;
        std::string full(ui.preedit);

        auto charToBytes = [&full](int charPos) -> size_t {
            if (charPos <= 0) return 0;
            return fcitx::utf8::ncharByteLength(
                full.begin(), static_cast<size_t>(charPos));
        };

        for (int i = 0; i < ui.segment_count; i++) {
            int startCh = ui.segments[i].start_chars;
            int lenCh = ui.segments[i].char_len;
            size_t startByte = charToBytes(startCh);
            size_t endByte = charToBytes(startCh + lenCh);
            std::string segText = full.substr(startByte, endByte - startByte);

            auto flag = (i == ui.focus_index)
                ? fcitx::TextFormatFlag::HighLight
                : fcitx::TextFormatFlag::Underline;
            preedit.append(segText, flag);
        }
        preedit.setCursor(full.size());
        panel.setClientPreedit(preedit);

        // Build candidate list for focused segment
        if (ui.candidate_count > 0) {
            auto candList = std::make_unique<fcitx::CommonCandidateList>();
            candList->setPageSize(10);
            for (int j = 0; j < ui.candidate_count; j++) {
                if (ui.candidates[j]) {
                    candList->append<fcitx::DisplayOnlyCandidateWord>(
                            fcitx::Text(ui.candidates[j]));
                }
            }
            if (ui.selected_index >= 0 && ui.selected_index < ui.candidate_count) {
                candList->setGlobalCursorIndex(ui.selected_index);
                int page = ui.selected_index / candList->pageSize();
                candList->setPage(page);
            }
            panel.setCandidateList(std::move(candList));
        }
    } else if (ui.has_preedit && ui.preedit && ui.preedit[0]) {
        // Preedit mode: show raw kana
        fcitx::Text preedit;
        preedit.append(ui.preedit, fcitx::TextFormatFlag::Underline);
        preedit.setCursor(std::string(ui.preedit).size());
        panel.setClientPreedit(preedit);
    }

    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void BonolithState::scheduleRerankRefresh() {
    // Poll cadence/budget for the background rerank. The pass is a llama-server
    // round-trip per segment, so allow ~2s before giving up.
    constexpr uint64_t kPollUs = 60000; // 60ms
    constexpr int kMaxTicks = 34;       // ~2.0s total

    rerankTicks_ = 0;
    auto *instance = engine_->instance();
    rerankTimer_ = instance->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, fcitx::now(CLOCK_MONOTONIC) + kPollUs, 0,
        [this](fcitx::EventSourceTime *time, uint64_t) {
            // Stop (don't re-arm) once the user has left conversion mode.
            if (!ctx_ || !bonolith_is_converting(ctx_)) {
                return true;
            }
            // Result landed AND candidates reordered: apply, repaint, stop.
            if (bonolith_poll_apply_rerank(ctx_)) {
                updateUI();
                return true;
            }
            // Applied-no-change or still pending — distinguish by
            // asking the engine whether the worker is still in flight
            // (Devin PR #3 review #11). Prior code always re-armed
            // for the full 2s budget on a false return, burning CPU
            // for the applied-no-change case where the worker had
            // already finished and there was nothing to wait for.
            if (!bonolith_rerank_pending(ctx_)) {
                return true;
            }
            // Still in flight: re-arm until the budget is spent.
            if (++rerankTicks_ < kMaxTicks) {
                time->setNextInterval(kPollUs);
                time->setOneShot();
            }
            return true;
        });
}

// ── BonolithEngine (addon) ──────────────────────────────────────────────────

/// Remove inherited env vars that point at snap-namespaced resources.
/// If the Fcitx5 daemon (or its parent) was launched from a snap
/// terminal like ghostty, GDK_PIXBUF_MODULE_FILE will point into the
/// snap's gdk-pixbuf cache; any GTK subprocess we spawn (zenity, the
/// Python register dialog, etc.) would then dlopen the snap's loaders
/// — built against a different libc — and crash. Clearing it makes
/// subprocesses fall back to the distribution loaders.
static void sanitizeInheritedEnv() {
    for (const char *var : {"GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR"}) {
        const char *val = ::getenv(var);
        if (val && (std::string(val).find("/snap/") != std::string::npos ||
                    std::string(val).find("/.snap") != std::string::npos)) {
            ::unsetenv(var);
        }
    }
}

BonolithEngine::BonolithEngine(fcitx::Instance *instance)
    : instance_(instance),
      factory_([this](fcitx::InputContext &ic) {
          return new BonolithState(this, &ic);
      }) {
    sanitizeInheritedEnv();
    instance_->inputContextManager().registerProperty("bonolithState", &factory_);

    // Set up menu actions
    actionRegister_.setShortText("単語登録");
    actionRegister_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { BonolithEngine::runWordRegister(); }).detach();
        });

    actionManage_.setShortText("辞書管理");
    actionManage_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { BonolithEngine::runManageDict(); }).detach();
        });

    actionExport_.setShortText("辞書エクスポート");
    actionExport_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { BonolithEngine::runExportDict(); }).detach();
        });

    actionImport_.setShortText("辞書インポート");
    actionImport_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { BonolithEngine::runImportDict(); }).detach();
        });

    actionClearLearning_.setShortText("学習履歴をクリア");
    actionClearLearning_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { BonolithEngine::runClearLearning(); }).detach();
        });

    instance_->userInterfaceManager().registerAction("bonolith-register",
                                                     &actionRegister_);
    instance_->userInterfaceManager().registerAction("bonolith-manage",
                                                     &actionManage_);
    instance_->userInterfaceManager().registerAction("bonolith-export",
                                                     &actionExport_);
    instance_->userInterfaceManager().registerAction("bonolith-import",
                                                     &actionImport_);
    instance_->userInterfaceManager().registerAction("bonolith-clear-learning",
                                                     &actionClearLearning_);

    menu_.addAction(&actionRegister_);
    menu_.addAction(&actionManage_);
    menu_.addAction(&actionExport_);
    menu_.addAction(&actionImport_);
    menu_.addAction(&actionClearLearning_);

    menuAction_.setShortText("Bonolith");
    menuAction_.setMenu(&menu_);
    instance_->userInterfaceManager().registerAction("bonolith-menu", &menuAction_);
}

std::vector<fcitx::InputMethodEntry> BonolithEngine::listInputMethods() {
    std::vector<fcitx::InputMethodEntry> result;
    result.emplace_back("bonolith", "Bonolith - Japanese AI Input", "ja",
                        "bonolith");
    return result;
}

void BonolithEngine::keyEvent(const fcitx::InputMethodEntry & /*entry*/,
                          fcitx::KeyEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    state->keyEvent(event);
}

void BonolithEngine::activate(const fcitx::InputMethodEntry & /*entry*/,
                          fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    // Use AfterInputMethod so the menu persists even when IME is deactivated
    // (InputMethod group is auto-cleared on deactivate)
    ic->statusArea().addAction(fcitx::StatusGroup::AfterInputMethod, &menuAction_);
}

void BonolithEngine::deactivate(const fcitx::InputMethodEntry & /*entry*/,
                            fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    // Commit (not discard) in-progress text on focus loss, like Mozc/Google IME.
    state->commitInput();
}

void BonolithEngine::reset(const fcitx::InputMethodEntry & /*entry*/,
                       fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    state->reset();
}

// ── Dictionary management (zenity dialogs) ─────────────────────────────

/// Helper: spawn `path` with the given argv (path is argv[0] verbatim, plus
/// each `extra_args` element), read its stdout, and return the stdout as a
/// string with any trailing newline trimmed. Returns empty string on
/// spawn failure or non-zero exit. `env_overrides` prepends `KEY=VALUE`
/// pairs onto the child's environment (later duplicates on the same key
/// are ignored per POSIX).
///
/// Uses posix_spawn + pipe instead of popen("sh -c …") so there is no
/// shell in the loop: entries carrying any characters — apostrophes,
/// backslashes, dollar signs — are delivered to the child as literal
/// bytes in argv without expansion or quoting, and the concatenated
/// command line's MAX_ARG_STRLEN limit no longer applies (each argv
/// element is bounded independently). Fixes fcitx5 side of bug [4]
/// (Fable-5 review 2026-08-31) and Devin PR #7 [R2-6]: the previous
/// popen path burst the 128 KiB shell-arg cap once quoted, so entries
/// with many apostrophes silently killed the dialog.
static std::string runProcessCaptureStdout(
    const std::string &path,
    const std::vector<std::string> &extra_args,
    const std::vector<std::string> &env_overrides = {})
{
    // O_CLOEXEC on both pipe fds is essential: fcitx5's dictionary
    // menu actions run on detached threads, so two dialogs can be
    // spawning concurrently. Without close-on-exec, the second
    // spawn's child would inherit the first pipe's write end, and
    // the first `read(pipefd[0])` never sees EOF even after the
    // first zenity exits — the first thread stalls until the second
    // dialog is dismissed. posix_spawn's file_actions closes the fd
    // in the direct child, but the O_CLOEXEC flag is what keeps
    // unrelated concurrent spawns from picking it up (Devin PR #7
    // [R3-9]). posix_spawn_file_actions_adddup2 clears CLOEXEC on
    // the duplicated stdout fd, so the child's stdout still works.
    int pipefd[2];
    if (pipe2(pipefd, O_CLOEXEC) != 0) return "";

    // Build argv: [path, extra_args..., NULL]. Each entry is passed as a
    // literal byte string; the child sees exactly what we put here, with
    // no quoting, splitting, or word expansion.
    std::vector<char *> argv;
    argv.reserve(extra_args.size() + 2);
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const auto &a : extra_args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);

    // Build envp: env_overrides first (so posix_spawn's search stops at
    // the override), then the parent env passthrough.
    std::vector<char *> envp;
    if (!env_overrides.empty()) {
        envp.reserve(env_overrides.size() + 1);
        for (const auto &e : env_overrides) {
            envp.push_back(const_cast<char *>(e.c_str()));
        }
        for (char **p = environ; *p; ++p) envp.push_back(*p);
        envp.push_back(nullptr);
    }
    char **child_env = env_overrides.empty() ? environ : envp.data();

    pid_t pid = 0;
    int rc = posix_spawnp(&pid, path.c_str(), &actions, nullptr,
                          argv.data(), child_env);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);

    if (rc != 0) {
        close(pipefd[0]);
        return "";
    }

    std::string result;
    char buf[512];
    for (;;) {
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        result.append(buf, n);
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    // Non-zero exit (including "user cancelled" from zenity) collapses
    // to an empty result — callers use empty-string as their cancel /
    // failure signal.
    bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok) return "";

    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

/// Helper: run zenity with `args`. See runProcessCaptureStdout for the
/// shell-free spawn contract.
static std::string runZenity(const std::vector<std::string> &args) {
    return runProcessCaptureStdout("zenity", args);
}

/// Helper: run a command and return true on exit-0. Used for the
/// delete-confirmation path where we want the boolean, not the stdout.
static bool runProcessCheckExit(
    const std::string &path,
    const std::vector<std::string> &extra_args)
{
    std::vector<char *> argv;
    argv.reserve(extra_args.size() + 2);
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const auto &a : extra_args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = 0;
    int rc = posix_spawnp(&pid, path.c_str(), nullptr, nullptr,
                          argv.data(), environ);
    if (rc != 0) return false;

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

void BonolithEngine::runWordRegister() {
    // Custom GTK dialog that re-activates Fcitx5 on every entry focus-in,
    // so 単語 stays 日本語ON even after Tab. Output: "<reading>|<surface>".
    // GDK_BACKEND=x11 forces XWayland on Wayland sessions so the D-Bus
    // activate can reach fcitx5-remote.
    std::string result = runProcessCaptureStdout(
        "/usr/bin/python3",
        {"/usr/share/bonolith/scripts/bonolith_word_register.py", "fcitx5"},
        {"GDK_BACKEND=x11"});
    if (result.empty()) return;

    auto sep = result.find('|');
    if (sep == std::string::npos) return;
    std::string reading = result.substr(0, sep);
    std::string surface = result.substr(sep + 1);
    if (reading.empty() || surface.empty()) {
        runZenity({"--error", "--title=Bonolith",
                   "--text=よみと単語の両方を入力してください"});
        return;
    }
    // Reject '|' in reading: the IPC format is `reading|surface` on the
    // first pipe, so a pipe in reading silently truncates the row
    // (Devin PR #7 [D6]). Reading is hiragana; refuse rather than guess.
    if (reading.find('|') != std::string::npos) {
        runZenity({"--error", "--title=Bonolith",
                   "--text=よみに '|' を含めることはできません"});
        return;
    }

    if (bonolith_dict_add_entry(reading.c_str(), surface.c_str())) {
        runZenity({"--info", "--title=Bonolith",
                   "--text=登録しました: " + reading + " → " + surface});
    } else {
        runZenity({"--error", "--title=Bonolith", "--text=登録に失敗しました"});
    }
}

void BonolithEngine::runManageDict() {
    BonolithDictEntries dict = bonolith_dict_get_user_entries();
    if (dict.count <= 0) {
        runZenity({"--info", "--title=Bonolith",
                   "--text=ユーザー辞書にエントリがありません"});
        bonolith_dict_free_entries(dict);
        return;
    }

    // Runtime cap on how many rows to render in one zenity list.
    // With runZenity now going through posix_spawn + pipe (no shell),
    // each argv element is bounded independently by ARG_MAX and the
    // combined-string MAX_ARG_STRLEN cap does not apply — a large user
    // dictionary no longer silently kills the dialog (bug [4] in
    // Fable-5 review 2026-08-31, and Devin PR #7 [R2-6] closed the
    // shell-quoting escape hatch [D5] tried to bound with a byte
    // budget). MAX_DISPLAY is kept as a UX guard so the list stays
    // legible; export/import cover the rest.
    constexpr int MAX_DISPLAY = 500;
    int display_count = dict.count;
    if (dict.count > MAX_DISPLAY) {
        runZenity({"--warning", "--title=Bonolith",
                   "--text=ユーザー辞書のエントリ数が" +
                   std::to_string(MAX_DISPLAY) + "件を超えています ("
                   + std::to_string(dict.count) + " 件)。\n先頭 "
                   + std::to_string(MAX_DISPLAY) +
                   " 件のみ表示します。\nエクスポートして内容を確認してください。"});
        display_count = MAX_DISPLAY;
    }

    // Step 1: Show list
    std::vector<std::string> args = {
        "--list",
        "--title=Bonolith: 辞書管理",
        "--text=エントリを選択してOKを押してください",
        "--column=#",
        "--column=よみ",
        "--column=単語",
        "--print-column=1",
        "--width=500",
        "--height=400",
    };
    for (int i = 0; i < display_count; i++) {
        args.push_back(std::to_string(i));
        args.push_back(dict.entries[i].reading);
        args.push_back(dict.entries[i].surface);
    }

    auto selected = runZenity(args);
    if (selected.empty()) {
        bonolith_dict_free_entries(dict);
        return;
    }

    // zenity emits "N|N" when the user double-clicks a row (see the
    // IBus side's parse for the same case, bug [7]); take the first
    // pipe-delimited field so both single-click ("3") and double-click
    // ("3|3") gestures parse to the same index.
    auto pipe = selected.find('|');
    std::string idx_str =
        (pipe == std::string::npos) ? selected : selected.substr(0, pipe);
    int idx = std::atoi(idx_str.c_str());
    if (idx < 0 || idx >= display_count) {
        bonolith_dict_free_entries(dict);
        return;
    }

    std::string selReading = dict.entries[idx].reading;
    std::string selSurface = dict.entries[idx].surface;
    bonolith_dict_free_entries(dict);

    // Step 2: Choose action
    auto action = runZenity({
        "--list", "--radiolist",
        "--title=Bonolith: 操作を選択",
        "--text=選択中: " + selReading + " → " + selSurface,
        "--column=", "--column=操作",
        "TRUE", "編集",
        "FALSE", "削除",
    });

    if (action == "削除") {
        // zenity --question returns exit code 0 for OK, non-0 for
        // cancel. Go through runProcessCheckExit — same shell-free
        // spawn as the rest, so an apostrophe in the confirmation
        // text no longer breaks the dialog and there is nothing to
        // shell-inject through a hostile registered word.
        std::string textArg = "--text=「" + selReading + "」→「" + selSurface + "」を削除しますか？";
        if (runProcessCheckExit("zenity",
                                {"--question",
                                 "--title=Bonolith: 削除の確認",
                                 textArg})) {
            // Apply by (reading, surface) identity captured above rather
            // than the display-time index: another dialog / process may
            // have added or reordered entries between showing the list
            // and confirming the delete, and an index-based apply would
            // clobber those changes via the store-level replace (bug [17]).
            if (bonolith_dict_delete_entry_by_identity(
                    selReading.c_str(), selSurface.c_str())) {
                runZenity({"--info", "--title=Bonolith", "--text=削除しました"});
            }
        }
    } else if (action == "編集") {
        // Reuse the GTK register dialog in edit mode (prefilled).
        // GDK_BACKEND=x11 forces XWayland on Wayland so the D-Bus
        // activate reaches fcitx5-remote.
        std::string result = runProcessCaptureStdout(
            "/usr/bin/python3",
            {"/usr/share/bonolith/scripts/bonolith_word_register.py",
             "fcitx5", "--mode", "edit",
             "--reading", selReading,
             "--surface", selSurface},
            {"GDK_BACKEND=x11"});
        if (result.empty()) return;

        auto sep = result.find('|');
        if (sep == std::string::npos) return;
        std::string newReading = result.substr(0, sep);
        std::string newSurface = result.substr(sep + 1);
        if (newReading.empty() || newSurface.empty()) return;
        // Reject '|' in reading — see the register-path comment above
        // for rationale (Devin PR #7 [D6]).
        if (newReading.find('|') != std::string::npos) {
            runZenity({"--error", "--title=Bonolith",
                       "--text=よみに '|' を含めることはできません"});
            return;
        }
        if (newReading == selReading && newSurface == selSurface) return;

        // Apply by (old_reading, old_surface) identity — same rationale
        // as the delete path above (bug [17]).
        if (bonolith_dict_update_entry_by_identity(
                selReading.c_str(), selSurface.c_str(),
                newReading.c_str(), newSurface.c_str())) {
            runZenity({"--info", "--title=Bonolith", "--text=辞書を更新しました"});
        } else {
            runZenity({"--error", "--title=Bonolith", "--text=更新に失敗しました"});
        }
    }
}

void BonolithEngine::runExportDict() {
    auto path = runZenity({
        "--file-selection", "--save",
        "--title=Bonolith: 辞書エクスポート",
        "--filename=bonolith_dict_export.json",
    });
    if (path.empty()) return;

    if (bonolith_dict_export(path.c_str())) {
        runZenity({"--info", "--title=Bonolith",
                   "--text=エクスポートしました: " + path});
    } else {
        runZenity({"--error", "--title=Bonolith", "--text=エクスポートに失敗しました"});
    }
}

void BonolithEngine::runImportDict() {
    auto path = runZenity({
        "--file-selection",
        "--title=Bonolith: 辞書インポート",
        "--file-filter=JSON files (*.json) | *.json",
    });
    if (path.empty()) return;

    int count = bonolith_dict_import(path.c_str());
    if (count >= 0) {
        runZenity({"--info", "--title=Bonolith",
                   "--text=" + std::to_string(count) + " 件インポートしました"});
    } else {
        runZenity({"--error", "--title=Bonolith", "--text=インポートに失敗しました"});
    }
}

void BonolithEngine::runClearLearning() {
    // zenity --question returns exit code 0 for OK, non-0 for cancel.
    // runProcessCheckExit is the shell-free equivalent of the old
    // system() call (Devin PR #7 [R2-6]).
    if (!runProcessCheckExit("zenity",
                             {"--question",
                              "--title=Bonolith 学習履歴クリア",
                              "--text=変換の学習履歴をすべて消去します。\n"
                              "この操作は元に戻せません。よろしいですか？",
                              "--ok-label=クリア",
                              "--cancel-label=キャンセル"})) {
        return;
    }

    int n = bonolith_clear_learning();
    if (n >= 0) {
        runZenity({"--info", "--title=Bonolith",
                   "--text=学習履歴を消去しました（" + std::to_string(n) +
                       " 件）。\n次回起動時から反映されます。"});
    } else {
        runZenity({"--error", "--title=Bonolith",
                   "--text=学習履歴のクリアに失敗しました"});
    }
}

} // namespace bonolith

FCITX_ADDON_FACTORY(bonolith::BonolithEngineFactory);

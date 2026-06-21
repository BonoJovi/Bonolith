/// Fcitx5 addon for Bonolith — thin C++ wrapper over the Rust engine.

#include "bonolith_engine.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <thread>

#include <fcitx-utils/utf8.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputpanel.h>

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

    if (bonolith_handle_key(ctx_, sym, state)) {
        event.filterAndAccept();
    }
    // Always update UI after key events to keep preedit display in sync
    updateUI();

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
    // Commit any in-progress composition instead of discarding it, then refresh
    // the UI so the committed text is flushed and the panel cleared. Used on
    // focus loss so clicking away (e.g. onto a modal) preserves typed text,
    // matching standard JP IMEs (Mozc etc.).
    if (ctx_)
        bonolith_commit_input(ctx_);
    updateUI();
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
            // Result landed: apply, repaint, and stop.
            if (bonolith_poll_apply_rerank(ctx_)) {
                updateUI();
                return true;
            }
            // Still pending: re-arm until the budget is spent.
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

/// Helper: shell-quote a value so it survives `popen`/shell expansion intact.
static std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

/// Helper: run a command and capture stdout. Returns empty string on failure.
static std::string runZenity(const std::vector<std::string> &args) {
    std::string cmd = "zenity";
    for (auto &a : args) {
        cmd += " '";
        // Simple quoting — replace ' with '\'' inside the argument
        std::string escaped = a;
        size_t pos = 0;
        while ((pos = escaped.find('\'', pos)) != std::string::npos) {
            escaped.replace(pos, 1, "'\\''");
            pos += 4;
        }
        cmd += escaped;
        cmd += "'";
    }
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return "";
    std::string result;
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        result += buf;
    }
    int status = pclose(fp);
    if (status != 0) return "";
    // Trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

void BonolithEngine::runWordRegister() {
    // Custom GTK dialog that re-activates Fcitx5 on every entry focus-in,
    // so 単語 stays 日本語ON even after Tab. Output: "<reading>|<surface>".
    FILE *fp = popen("GDK_BACKEND=x11 /usr/bin/python3 "
                     "/usr/share/bonolith/scripts/bonolith_word_register.py "
                     "fcitx5",
                     "r");
    if (!fp) return;
    std::string result;
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        result += buf;
    }
    int status = pclose(fp);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    if (status != 0 || result.empty()) return;

    auto sep = result.find('|');
    if (sep == std::string::npos) return;
    std::string reading = result.substr(0, sep);
    std::string surface = result.substr(sep + 1);
    if (reading.empty() || surface.empty()) {
        runZenity({"--error", "--title=Bonolith",
                   "--text=よみと単語の両方を入力してください"});
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
    for (int i = 0; i < dict.count; i++) {
        args.push_back(std::to_string(i));
        args.push_back(dict.entries[i].reading);
        args.push_back(dict.entries[i].surface);
    }

    auto selected = runZenity(args);
    if (selected.empty()) {
        bonolith_dict_free_entries(dict);
        return;
    }

    int idx = std::atoi(selected.c_str());
    if (idx < 0 || idx >= dict.count) {
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
        // zenity --question returns exit code 0 for OK, non-0 for cancel.
        // Use system() since runZenity() treats non-zero exit as failure (empty string).
        std::string cmd = "zenity --question '--title=Bonolith: 削除の確認' "
                          "'--text=「" + selReading + "」→「" + selSurface + "」を削除しますか？'";
        if (system(cmd.c_str()) == 0) {
            if (bonolith_dict_delete_entry(idx)) {
                runZenity({"--info", "--title=Bonolith", "--text=削除しました"});
            }
        }
    } else if (action == "編集") {
        // Reuse the GTK register dialog in edit mode (prefilled). GDK_BACKEND=x11
        // forces XWayland on Wayland sessions so xdotool key delivery works.
        std::string cmd = "GDK_BACKEND=x11 /usr/bin/python3 "
                          "/usr/share/bonolith/scripts/bonolith_word_register.py "
                          "fcitx5 --mode edit "
                          "--reading " + shellQuote(selReading) + " "
                          "--surface " + shellQuote(selSurface);
        FILE *fp = popen(cmd.c_str(), "r");
        if (!fp) return;
        std::string result;
        char buf[256];
        while (fgets(buf, sizeof(buf), fp)) {
            result += buf;
        }
        int status = pclose(fp);
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
            result.pop_back();
        if (status != 0 || result.empty()) return;

        auto sep = result.find('|');
        if (sep == std::string::npos) return;
        std::string newReading = result.substr(0, sep);
        std::string newSurface = result.substr(sep + 1);
        if (newReading.empty() || newSurface.empty()) return;
        if (newReading == selReading && newSurface == selSurface) return;

        if (bonolith_dict_update_entry(idx, newReading.c_str(), newSurface.c_str())) {
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
    // runZenity() treats non-zero exit as failure (empty stdout), so use system() here.
    std::string cmd = "zenity --question "
                      "'--title=Bonolith 学習履歴クリア' "
                      "'--text=変換の学習履歴をすべて消去します。\n"
                      "この操作は元に戻せません。よろしいですか？' "
                      "'--ok-label=クリア' "
                      "'--cancel-label=キャンセル'";
    if (system(cmd.c_str()) != 0) {
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

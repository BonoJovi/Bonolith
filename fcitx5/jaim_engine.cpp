/// Fcitx5 addon for JaIM — thin C++ wrapper over the Rust engine.

#include "jaim_engine.h"

#include <cstdlib>
#include <thread>

#include <fcitx-utils/utf8.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputpanel.h>

namespace jaim {

// ── JaimState (per-InputContext) ─────────────────────────────────────────

JaimState::JaimState(JaimEngine *engine, fcitx::InputContext *ic)
    : engine_(engine), ic_(ic), ctx_(jaim_context_new()) {}

JaimState::~JaimState() {
    if (ctx_) {
        jaim_context_free(ctx_);
    }
}

void JaimState::keyEvent(fcitx::KeyEvent &event) {
    if (!ctx_)
        return;

    uint32_t sym = event.rawKey().sym();
    uint32_t state = event.rawKey().states();
    if (event.isRelease())
        state |= (1u << 30); // RELEASE_MASK

    if (jaim_handle_key(ctx_, sym, state)) {
        event.filterAndAccept();
    }
    // Always update UI after key events to keep preedit display in sync
    updateUI();
}

void JaimState::reset() {
    if (ctx_)
        jaim_reset(ctx_);
    ic_->inputPanel().reset();
    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void JaimState::updateUI() {
    auto &panel = ic_->inputPanel();
    panel.reset();

    // Single FFI call to get all UI state
    JaimUiState ui{};
    jaim_get_ui_state(ctx_, &ui);

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

// ── JaimEngine (addon) ──────────────────────────────────────────────────

JaimEngine::JaimEngine(fcitx::Instance *instance)
    : instance_(instance),
      factory_([this](fcitx::InputContext &ic) {
          return new JaimState(this, &ic);
      }) {
    instance_->inputContextManager().registerProperty("jaimState", &factory_);

    // Set up menu actions
    actionRegister_.setShortText("単語登録");
    actionRegister_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { JaimEngine::runWordRegister(); }).detach();
        });

    actionManage_.setShortText("辞書管理");
    actionManage_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { JaimEngine::runManageDict(); }).detach();
        });

    actionExport_.setShortText("辞書エクスポート");
    actionExport_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { JaimEngine::runExportDict(); }).detach();
        });

    actionImport_.setShortText("辞書インポート");
    actionImport_.connect<fcitx::SimpleAction::Activated>(
        [](fcitx::InputContext * /*ic*/) {
            std::thread([]() { JaimEngine::runImportDict(); }).detach();
        });

    instance_->userInterfaceManager().registerAction("jaim-register",
                                                     &actionRegister_);
    instance_->userInterfaceManager().registerAction("jaim-manage",
                                                     &actionManage_);
    instance_->userInterfaceManager().registerAction("jaim-export",
                                                     &actionExport_);
    instance_->userInterfaceManager().registerAction("jaim-import",
                                                     &actionImport_);

    menu_.addAction(&actionRegister_);
    menu_.addAction(&actionManage_);
    menu_.addAction(&actionExport_);
    menu_.addAction(&actionImport_);

    menuAction_.setShortText("JaIM");
    menuAction_.setMenu(&menu_);
    instance_->userInterfaceManager().registerAction("jaim-menu", &menuAction_);
}

std::vector<fcitx::InputMethodEntry> JaimEngine::listInputMethods() {
    std::vector<fcitx::InputMethodEntry> result;
    result.emplace_back("jaim", "JaIM - Japanese AI Input", "ja",
                        "jaim");
    return result;
}

void JaimEngine::keyEvent(const fcitx::InputMethodEntry & /*entry*/,
                          fcitx::KeyEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    state->keyEvent(event);
}

void JaimEngine::activate(const fcitx::InputMethodEntry & /*entry*/,
                          fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    ic->statusArea().addAction(fcitx::StatusGroup::InputMethod, &menuAction_);
}

void JaimEngine::deactivate(const fcitx::InputMethodEntry & /*entry*/,
                            fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    state->reset();
}

void JaimEngine::reset(const fcitx::InputMethodEntry & /*entry*/,
                       fcitx::InputContextEvent &event) {
    auto *ic = event.inputContext();
    auto *state = ic->propertyFor(&factory_);
    state->reset();
}

// ── Dictionary management (zenity dialogs) ─────────────────────────────

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

void JaimEngine::runWordRegister() {
    auto result = runZenity({
        "--forms",
        "--title=JaIM: 単語登録",
        "--text=ユーザー辞書に新しい単語を登録します",
        "--add-entry=よみ (ひらがな)",
        "--add-entry=単語 (漢字・カタカナなど)",
        "--separator=|",
    });
    if (result.empty()) return;

    auto sep = result.find('|');
    if (sep == std::string::npos) return;
    std::string reading = result.substr(0, sep);
    std::string surface = result.substr(sep + 1);
    if (reading.empty() || surface.empty()) {
        runZenity({"--error", "--title=JaIM",
                   "--text=よみと単語の両方を入力してください"});
        return;
    }

    if (jaim_dict_add_entry(reading.c_str(), surface.c_str())) {
        runZenity({"--info", "--title=JaIM",
                   "--text=登録しました: " + reading + " → " + surface});
    } else {
        runZenity({"--error", "--title=JaIM", "--text=登録に失敗しました"});
    }
}

void JaimEngine::runManageDict() {
    JaimDictEntries dict = jaim_dict_get_user_entries();
    if (dict.count <= 0) {
        runZenity({"--info", "--title=JaIM",
                   "--text=ユーザー辞書にエントリがありません"});
        jaim_dict_free_entries(dict);
        return;
    }

    // Step 1: Show list
    std::vector<std::string> args = {
        "--list",
        "--title=JaIM: 辞書管理",
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
        jaim_dict_free_entries(dict);
        return;
    }

    int idx = std::atoi(selected.c_str());
    if (idx < 0 || idx >= dict.count) {
        jaim_dict_free_entries(dict);
        return;
    }

    std::string selReading = dict.entries[idx].reading;
    std::string selSurface = dict.entries[idx].surface;
    jaim_dict_free_entries(dict);

    // Step 2: Choose action
    auto action = runZenity({
        "--list", "--radiolist",
        "--title=JaIM: 操作を選択",
        "--text=選択中: " + selReading + " → " + selSurface,
        "--column=", "--column=操作",
        "TRUE", "編集",
        "FALSE", "削除",
    });

    if (action == "削除") {
        // zenity --question returns exit code 0 for OK, non-0 for cancel.
        // Use system() since runZenity() treats non-zero exit as failure (empty string).
        std::string cmd = "zenity --question '--title=JaIM: 削除の確認' "
                          "'--text=「" + selReading + "」→「" + selSurface + "」を削除しますか？'";
        if (system(cmd.c_str()) == 0) {
            if (jaim_dict_delete_entry(idx)) {
                runZenity({"--info", "--title=JaIM", "--text=削除しました"});
            }
        }
    } else if (action == "編集") {
        auto result = runZenity({
            "--forms",
            "--title=JaIM: 単語の編集",
            "--text=現在のよみ: " + selReading + "\n現在の単語: " + selSurface +
                "\n\n変更する項目のみ入力してください (空欄は変更なし)",
            "--add-entry=よみ (ひらがな)",
            "--add-entry=単語 (漢字・カタカナなど)",
            "--separator=|",
        });
        if (result.empty()) return;
        auto sep = result.find('|');
        if (sep == std::string::npos) return;
        std::string newReading = result.substr(0, sep);
        std::string newSurface = result.substr(sep + 1);
        if (newReading.empty() && newSurface.empty()) return;

        if (jaim_dict_update_entry(idx, newReading.c_str(), newSurface.c_str())) {
            runZenity({"--info", "--title=JaIM", "--text=辞書を更新しました"});
        } else {
            runZenity({"--error", "--title=JaIM", "--text=更新に失敗しました"});
        }
    }
}

void JaimEngine::runExportDict() {
    auto path = runZenity({
        "--file-selection", "--save",
        "--title=JaIM: 辞書エクスポート",
        "--filename=jaim_dict_export.json",
    });
    if (path.empty()) return;

    if (jaim_dict_export(path.c_str())) {
        runZenity({"--info", "--title=JaIM",
                   "--text=エクスポートしました: " + path});
    } else {
        runZenity({"--error", "--title=JaIM", "--text=エクスポートに失敗しました"});
    }
}

void JaimEngine::runImportDict() {
    auto path = runZenity({
        "--file-selection",
        "--title=JaIM: 辞書インポート",
        "--file-filter=JSON files (*.json) | *.json",
    });
    if (path.empty()) return;

    int count = jaim_dict_import(path.c_str());
    if (count >= 0) {
        runZenity({"--info", "--title=JaIM",
                   "--text=" + std::to_string(count) + " 件インポートしました"});
    } else {
        runZenity({"--error", "--title=JaIM", "--text=インポートに失敗しました"});
    }
}

} // namespace jaim

FCITX_ADDON_FACTORY(jaim::JaimEngineFactory);

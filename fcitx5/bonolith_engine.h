/// Fcitx5 addon for Bonolith — thin C++ wrapper over the Rust engine.

#ifndef BONOLITH_ENGINE_H
#define BONOLITH_ENGINE_H

#include <fcitx/action.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputcontextproperty.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>
#include <fcitx/menu.h>
#include <fcitx/statusarea.h>
#include <fcitx/userinterfacemanager.h>

#include "bonolith_ffi.h"

namespace bonolith {

class BonolithEngine;

/// Per-InputContext state wrapping a BonolithContext (Rust engine).
class BonolithState : public fcitx::InputContextProperty {
public:
    BonolithState(BonolithEngine *engine, fcitx::InputContext *ic);
    ~BonolithState();

    void keyEvent(fcitx::KeyEvent &event);
    void reset();
    void commitInput();

private:
    void updateUI();

    BonolithEngine *engine_;
    fcitx::InputContext *ic_;
    BonolithContext *ctx_;
};

/// Fcitx5 input method engine addon.
class BonolithEngine : public fcitx::InputMethodEngineV2 {
public:
    BonolithEngine(fcitx::Instance *instance);

    void keyEvent(const fcitx::InputMethodEntry &entry,
                  fcitx::KeyEvent &event) override;
    void activate(const fcitx::InputMethodEntry &entry,
                  fcitx::InputContextEvent &event) override;
    void deactivate(const fcitx::InputMethodEntry &entry,
                    fcitx::InputContextEvent &event) override;
    void reset(const fcitx::InputMethodEntry &entry,
               fcitx::InputContextEvent &event) override;

    std::vector<fcitx::InputMethodEntry> listInputMethods() override;

    auto &factory() { return factory_; }

private:
    // Dictionary management via zenity dialogs (run in background threads)
    static void runWordRegister();
    static void runManageDict();
    static void runExportDict();
    static void runImportDict();
    static void runClearLearning();

    fcitx::Instance *instance_;
    fcitx::FactoryFor<BonolithState> factory_;

    // Menu actions
    fcitx::SimpleAction actionRegister_;
    fcitx::SimpleAction actionManage_;
    fcitx::SimpleAction actionExport_;
    fcitx::SimpleAction actionImport_;
    fcitx::SimpleAction actionClearLearning_;
    fcitx::Menu menu_;
    fcitx::SimpleAction menuAction_;
};

class BonolithEngineFactory : public fcitx::AddonFactory {
    fcitx::AddonInstance *
    create(fcitx::AddonManager *manager) override {
        return new BonolithEngine(manager->instance());
    }
};

} // namespace bonolith

#endif // BONOLITH_ENGINE_H

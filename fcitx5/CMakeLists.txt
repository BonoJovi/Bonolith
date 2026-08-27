cmake_minimum_required(VERSION 3.16)
project(fcitx5-bonolith VERSION 3.1.1)

set(CMAKE_CXX_STANDARD 17)

find_package(Fcitx5Core REQUIRED)
find_package(Fcitx5Utils REQUIRED)

include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake" OPTIONAL)

# Rust shared library (libbonolith.so) — built separately by cargo
set(BONOLITH_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/release"
    CACHE PATH "Directory containing libbonolith.so")

add_library(fcitx5-bonolith MODULE bonolith_engine.cpp)
# Drop the lib prefix and any RPATH that would otherwise embed the
# developer's build-tree absolute path. After install, the runtime
# loader resolves libbonolith.so via the standard search path
# (/usr/lib/x86_64-linux-gnu, which is where we install it).
# BUILD_WITH_INSTALL_RPATH=TRUE makes the build artifact already use
# the (empty) INSTALL_RPATH, so install.sh's plain `install -D` copy
# from build/ to /usr/lib delivers a clean binary without needing
# `make install` to strip the convenience build rpath afterwards.
set_target_properties(fcitx5-bonolith PROPERTIES
    PREFIX ""
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH ""
    INSTALL_RPATH_USE_LINK_PATH FALSE
)

target_include_directories(fcitx5-bonolith PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libbonolith.so by name (-lbonolith) using a link directory hint, rather
# than passing the absolute build-tree path, so the resulting NEEDED
# entry is just "libbonolith.so" with no path baked in.
target_link_directories(fcitx5-bonolith PRIVATE "${BONOLITH_LIB_DIR}")
target_link_libraries(fcitx5-bonolith
    Fcitx5::Core
    Fcitx5::Config
    Fcitx5::Utils
    bonolith
)

# Install the addon shared library
install(TARGETS fcitx5-bonolith
    LIBRARY DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5"
)

# Install the addon descriptor
install(FILES bonolith-addon.conf
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
    RENAME bonolith.conf
)

# Install the input method descriptor
install(FILES bonolith-im.conf
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod"
    RENAME bonolith.conf
)

# Install the Rust shared library
install(FILES "${BONOLITH_LIB_DIR}/libbonolith.so"
    DESTINATION "${FCITX_INSTALL_LIBDIR}"
)

# Clean up stale files from previous /usr/local installs to prevent conflicts
install(CODE "
    set(_stale_files
        \"/usr/local/share/fcitx5/addon/bonolith.conf\"
        \"/usr/local/share/fcitx5/inputmethod/bonolith.conf\"
        \"/usr/local/lib/x86_64-linux-gnu/fcitx5/fcitx5-bonolith.so\"
        \"/usr/local/lib/x86_64-linux-gnu/libbonolith.so\"
    )
    foreach(_f \${_stale_files})
        if(EXISTS \"\${_f}\")
            message(STATUS \"Removing stale file: \${_f}\")
            file(REMOVE \"\${_f}\")
        endif()
    endforeach()
")

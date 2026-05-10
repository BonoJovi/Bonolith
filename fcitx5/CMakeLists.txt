cmake_minimum_required(VERSION 3.16)
project(fcitx5-jaim VERSION 1.2.3)

set(CMAKE_CXX_STANDARD 17)

find_package(Fcitx5Core REQUIRED)
find_package(Fcitx5Utils REQUIRED)

include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake" OPTIONAL)

# Rust shared library (libjaim.so) — built separately by cargo
set(JAIM_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/release"
    CACHE PATH "Directory containing libjaim.so")

add_library(fcitx5-jaim MODULE jaim_engine.cpp)
set_target_properties(fcitx5-jaim PROPERTIES PREFIX "")

target_include_directories(fcitx5-jaim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fcitx5-jaim
    Fcitx5::Core
    Fcitx5::Config
    Fcitx5::Utils
    "${JAIM_LIB_DIR}/libjaim.so"
)

# Install the addon shared library
install(TARGETS fcitx5-jaim
    LIBRARY DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5"
)

# Install the addon descriptor
install(FILES jaim-addon.conf
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
    RENAME jaim.conf
)

# Install the input method descriptor
install(FILES jaim-im.conf
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod"
    RENAME jaim.conf
)

# Install the Rust shared library
install(FILES "${JAIM_LIB_DIR}/libjaim.so"
    DESTINATION "${FCITX_INSTALL_LIBDIR}"
)

# Clean up stale files from previous /usr/local installs to prevent conflicts
install(CODE "
    set(_stale_files
        \"/usr/local/share/fcitx5/addon/jaim.conf\"
        \"/usr/local/share/fcitx5/inputmethod/jaim.conf\"
        \"/usr/local/lib/x86_64-linux-gnu/fcitx5/fcitx5-jaim.so\"
        \"/usr/local/lib/x86_64-linux-gnu/libjaim.so\"
    )
    foreach(_f \${_stale_files})
        if(EXISTS \"\${_f}\")
            message(STATUS \"Removing stale file: \${_f}\")
            file(REMOVE \"\${_f}\")
        endif()
    endforeach()
")

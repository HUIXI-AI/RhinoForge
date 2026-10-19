set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../VERSION"
    "${CMAKE_CURRENT_LIST_DIR}/runtime-dependencies.json")
file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" RHINOFORGE_VERSION)
string(STRIP "${RHINOFORGE_VERSION}" RHINOFORGE_VERSION)
if(NOT RHINOFORGE_VERSION MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)(rc[0-9]+|\\.dev[0-9]+)?$")
    message(FATAL_ERROR "VERSION must contain a release, rc or dev version")
endif()
set(RHINOFORGE_CMAKE_VERSION "${CMAKE_MATCH_1}")
set(RHINOFORGE_VERSION_STAGE "${CMAKE_MATCH_2}")

file(READ "${CMAKE_CURRENT_LIST_DIR}/runtime-dependencies.json" RHINOFORGE_RUNTIME_DEPENDENCIES)
foreach(field IN ITEMS cmake_version release_version abi_version)
    string(JSON value GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel "${field}")
    string(TOUPPER "${field}" upper_field)
    set("RHINOFORGE_LAUNCH_${upper_field}" "${value}")
endforeach()
string(JSON RHINOFORGE_RUNTIME_BUNDLE_VERSION GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" runtime_bundle version)
string(JSON launch_version GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel version)
string(REGEX REPLACE "(rc[0-9]+|\\.dev[0-9]+)$" "" launch_release_base "${RHINOFORGE_LAUNCH_RELEASE_VERSION}")
if(NOT launch_version STREQUAL RHINOFORGE_LAUNCH_CMAKE_VERSION OR
   (RHINOFORGE_LAUNCH_RELEASE_VERSION AND
    NOT launch_release_base STREQUAL RHINOFORGE_LAUNCH_CMAKE_VERSION))
    message(FATAL_ERROR "Launch version fields disagree in runtime-dependencies.json")
endif()

# The delivered bundle predates exported release/ABI metadata; pin its actual files.
function(rhinoforge_check_launch_package)
    get_filename_component(prefix "${rhino_launch_DIR}/../../.." REALPATH)
    string(JSON count LENGTH "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel files)
    if(count LESS 1)
        message(FATAL_ERROR "Launch package file identities are required")
    endif()
    math(EXPR last "${count} - 1")
    foreach(index RANGE ${last})
        string(JSON path MEMBER "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel files ${index})
        string(JSON expected GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel files "${path}")
        if(NOT EXISTS "${prefix}/${path}")
            message(FATAL_ERROR "Launch package file missing: ${path}; use runtime ${RHINOFORGE_RUNTIME_BUNDLE_VERSION}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${prefix}/${path}")
        file(SHA256 "${prefix}/${path}" actual)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "Launch package checksum mismatch: ${path}; use runtime ${RHINOFORGE_RUNTIME_BUNDLE_VERSION}")
        endif()
    endforeach()
endfunction()

# Provider identities invalidate external cost certificates after a runtime change.
function(rhinoforge_generate_runtime_identity output_dir)
    string(JSON launch_commit GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel commit)
    string(JSON main_sha GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" artifacts rhino_ops_main sha256)
    string(JSON expansion_sha GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" artifacts rhino_ops_expansion sha256)
    file(MAKE_DIRECTORY "${output_dir}")
    file(CONFIGURE OUTPUT "${output_dir}/rhino_runtime_identity.h" CONTENT
"#pragma once
inline constexpr char kKvInsertCostRuntimeRelease[] = \"@RHINOFORGE_VERSION@\";
inline constexpr char kKvInsertCostLaunchCommit[] = \"@launch_commit@\";
inline constexpr char kKvInsertCostOperatorSha256[] = \"@main_sha@\";
inline constexpr char kKvInsertCostExpansionSha256[] = \"@expansion_sha@\";
" @ONLY)
endfunction()

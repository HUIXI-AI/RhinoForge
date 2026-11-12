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
string(JSON op_abi_epoch_type TYPE "${RHINOFORGE_RUNTIME_DEPENDENCIES}" op_abi_epoch)
string(JSON RHINOFORGE_OP_ABI_EPOCH GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" op_abi_epoch)
if(NOT op_abi_epoch_type STREQUAL "NUMBER" OR NOT RHINOFORGE_OP_ABI_EPOCH MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "Operator ABI epoch must be a positive integer")
endif()
foreach(field IN ITEMS version cmake_version release_version abi_version)
    string(JSON value GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel "${field}")
    string(TOUPPER "${field}" upper_field)
    set("RHINOFORGE_LAUNCH_${upper_field}" "${value}")
endforeach()
if(NOT RHINOFORGE_LAUNCH_RELEASE_VERSION MATCHES
   "^((0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*))(rc[1-9][0-9]*|\\.dev(0|[1-9][0-9]*))?$")
    message(FATAL_ERROR "Launch release version must be a canonical release, rc or dev version")
endif()
if(NOT RHINOFORGE_LAUNCH_VERSION STREQUAL CMAKE_MATCH_1 OR
   NOT RHINOFORGE_LAUNCH_CMAKE_VERSION STREQUAL CMAKE_MATCH_1)
    message(FATAL_ERROR "Launch version and CMake version must match its release version's numeric version")
endif()

# Provider identities invalidate external cost certificates after a runtime change.
function(rhinoforge_generate_runtime_identity output_dir)
    string(JSON op_abi_epoch GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" op_abi_epoch)
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
inline constexpr int kRhinoOpAbiEpoch = @op_abi_epoch@;
" @ONLY)
endfunction()

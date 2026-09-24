cmake_minimum_required(VERSION 3.18)
project(RhinoForge VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# The wheel/runtime loads Launch and PyTorch through the deployment
# environment.  Never bake a build-machine RUNPATH into the publishable
# extension (it can disclose internal paths and can silently select the wrong
# runtime set).  Validation builds may still provide LD_LIBRARY_PATH.
set(CMAKE_SKIP_RPATH ON)

# Strip absolute build-tree paths from __FILE__/__builtin_FILE (assert/TORCH_CHECK
# messages) so the shipped .so does not leak the build machine's source layout.
add_compile_options(
    "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=.")

# === Python3 ===
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# === PyTorch: auto-detect from current Python environment ===
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX} ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
find_package(rhino_launch 1.0.0 EXACT CONFIG REQUIRED)

# Check exact capabilities because package version/SONAME alone do not prove
# that headers and the shared object came from the required Release build.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES rhino_launch::rhino_launch)
unset(RHINOFORGE_HAS_LAUNCH_RELEASE_RUNTIME_V2 CACHE)
check_cxx_source_compiles(
    "#include <rhino_launch_features.h>
     #include <rhino_launch_def.h>
     #include <rhino_launch_kernel.h>
     #include <rhino_launch_queue.h>
     #ifndef RHINO_LAUNCH_HAS_DEV_PROGRAM_API
     #error Rhino Launch feature identity is missing
     #endif
     #if RHINO_LAUNCH_HAS_DEV_PROGRAM_API != 0
     #error RhinoForge requires the sanitized Release API
     #endif
     uint32_t probe_register_state_token(const rhino_lkn::Kernel_t& kernel) {
       uint64_t register_token = 0;
       return kernel.query_register_state_token(&register_token);
     }
     int main(int argc, char**) {
       if (argc == 42) {
         if (rhino_lkn::rhino_launch_runtime_build_flavor() !=
                 rhino_lkn::kRuntimeBuildRelease ||
             (rhino_lkn::rhino_launch_runtime_capabilities() &
              rhino_lkn::kRuntimeCapabilitySanitizedHwPerfChrome) == 0 ||
             (rhino_lkn::rhino_launch_runtime_capabilities() &
              rhino_lkn::kRuntimeCapabilityDevProgramApi) != 0)
           return 2;
         rhino_lkn::GraphArenaRequirement requirement{1, 1, 1, 0};
         rhino_lkn::GraphArenaPoolStats stats{};
         (void)rhino_lkn::ConfigureGraphArenaPool(&requirement, 1);
         (void)rhino_lkn::QueryGraphArenaPoolStats(&stats);
         rhino_lkn::Queue_t queue(1);
         (void)queue.discard_completed_batch();
         queue.set_enable_hw_perf(false);
         (void)queue.is_hw_perf_enabled();
         (void)queue.dump_hw_perf_chrome(\"unused.json\");
         auto* volatile token_probe = &probe_register_state_token;
         (void)token_probe;
       }
       return 0;
     }"
    RHINOFORGE_HAS_LAUNCH_RELEASE_RUNTIME_V2)
cmake_pop_check_state()
if(NOT RHINOFORGE_HAS_LAUNCH_RELEASE_RUNTIME_V2)
    message(FATAL_ERROR
        "RhinoForge requires a sanitized Release Rhino Launch package with "
        "graph-arena-pool-v2, register-state-token, and Chrome hwperf support")
endif()

# Add torch lib directory so linker can find libtorch_python.so
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories(${TORCH_LIB_DIR})

# Torch's own headers embed their absolute path in __FILE__ via TORCH_CHECK /
# TORCH_INTERNAL_ASSERT in inlined templates, which our .so then carries as
# string constants (leaks the build env's conda path). Rewrite the torch install
# root to a neutral prefix — same mechanism as the CMAKE_SOURCE_DIR map above.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch,os;print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_INSTALL_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os,torch;print(os.path.realpath(os.path.dirname(torch.__file__)))"
    OUTPUT_VARIABLE TORCH_INSTALL_ROOT_REAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_options(
    "-ffile-prefix-map=${TORCH_INSTALL_ROOT}=torch"
    "-fmacro-prefix-map=${TORCH_INSTALL_ROOT}=torch")
if(NOT TORCH_INSTALL_ROOT_REAL STREQUAL TORCH_INSTALL_ROOT)
    add_compile_options(
        "-ffile-prefix-map=${TORCH_INSTALL_ROOT_REAL}=torch"
        "-fmacro-prefix-map=${TORCH_INSTALL_ROOT_REAL}=torch")
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

add_subdirectory(src)
set_target_properties(rpu_backend PROPERTIES PREFIX "")

target_link_libraries(rpu_backend
    PRIVATE
    ${TORCH_LIBRARIES}
    torch_python
    Python3::Module
)

set_target_properties(rpu_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/rpu_backend
)

install(TARGETS rpu_backend LIBRARY DESTINATION rpu_backend)

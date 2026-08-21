cmake_minimum_required(VERSION 3.18)
project(RhinoForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# Add torch lib directory so linker can find libtorch_python.so
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path + '/../../lib')"
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
add_compile_options(
    "-ffile-prefix-map=${TORCH_INSTALL_ROOT}=torch"
    "-fmacro-prefix-map=${TORCH_INSTALL_ROOT}=torch")

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

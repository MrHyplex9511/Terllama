cmake_minimum_required(VERSION 3.16)
project(terllama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────
option(BUILD_BENCHMARK "Build the multi-kernel benchmark" ON)
option(BUILD_SERVER    "Build the API server" ON)
option(BUILD_DOWNLOADER "Build the model downloader" ON)

# ─── OpenMP ───────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ─── Architecture detection & flags ───────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(ARCH_FLAGS "-march=x86-64 -mtune=generic")
    elseif(MSVC)
        set(ARCH_FLAGS "/arch:SSE2")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(ARCH_FLAGS "-march=armv8-a+fp+simd")
    endif()
endif()

# ─── Third-party includes ─────────────────────────────────────────────────
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ─── Kernel objects (separate compile for ISA-specific flags) ─────────────
# Scalar
add_library(terllama_kernel_scalar OBJECT src/kernel_scalar.cpp)
target_compile_options(terllama_kernel_scalar PRIVATE ${ARCH_FLAGS} -O3)

# AVX2 (if x86)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    add_library(terllama_kernel_avx2 OBJECT src/kernel_avx2.cpp)
    target_compile_options(terllama_kernel_avx2 PRIVATE -mavx2 -mfma -O3)
else()
    add_library(terllama_kernel_avx2 OBJECT src/kernel_avx2.cpp)
    target_compile_options(terllama_kernel_avx2 PRIVATE ${ARCH_FLAGS} -O3)
endif()

# NEON (if ARM64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    add_library(terllama_kernel_neon OBJECT src/kernel_neon.cpp)
    target_compile_options(terllama_kernel_neon PRIVATE -march=armv8-a+fp+simd -O3)
endif()

# Dispatcher (always)
add_library(terllama_dispatcher OBJECT src/dispatcher.cpp)
target_compile_options(terllama_dispatcher PRIVATE ${ARCH_FLAGS} -O3)

# ─── Main binary (CLI + server + downloader) ─────────────────────────────
add_executable(terllama
    src/main.cpp
    src/server.cpp
    src/downloader.cpp
    $<TARGET_OBJECTS:terllama_kernel_scalar>
    $<TARGET_OBJECTS:terllama_kernel_avx2>
    $<TARGET_OBJECTS:terllama_dispatcher>
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    target_sources(terllama PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
endif()

target_include_directories(terllama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${THIRD_PARTY_DIR}
)
target_compile_options(terllama PRIVATE ${ARCH_FLAGS} -O3)
target_link_libraries(terllama PRIVATE
    OpenMP::OpenMP_CXX
    pthread
)
target_compile_definitions(terllama PRIVATE
    $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_DEFINITIONS>
)

# ─── Benchmark binary (optional) ─────────────────────────────────────────
if(BUILD_BENCHMARK)
    add_executable(terllama-bench
        src/benchmark.cpp
        $<TARGET_OBJECTS:terllama_kernel_scalar>
        $<TARGET_OBJECTS:terllama_kernel_avx2>
        $<TARGET_OBJECTS:terllama_dispatcher>
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        target_sources(terllama-bench PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
    endif()
    target_include_directories(terllama-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(terllama-bench PRIVATE ${ARCH_FLAGS} -O3)
    target_link_libraries(terllama-bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(terllama-bench PRIVATE
        $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# ─── Install ──────────────────────────────────────────────────────────────
install(TARGETS terllama RUNTIME DESTINATION bin)
if(BUILD_BENCHMARK)
    install(TARGETS terllama-bench RUNTIME DESTINATION bin)
endif()

# ─── Web UI assets ────────────────────────────────────────────────────────
install(DIRECTORY web/ DESTINATION share/terllama/web)

# ─── Print build summary ─────────────────────────────────────────────────
message(STATUS "Terllama build configuration:")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Architecture:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Flags:            ${ARCH_FLAGS}")
message(STATUS "  OpenMP:           ${OpenMP_CXX_FOUND}")
message(STATUS "  libcurl:          ${CURL_FOUND}")
message(STATUS "  Benchmark:        ${BUILD_BENCHMARK}")

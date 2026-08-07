cmake_minimum_required(VERSION 3.16)
project(terllama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── IPO / LTO (critical for inference perf) ──────────────────────────────
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
if(lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "LTO/IPO: enabled")
else()
    message(WARNING "LTO/IPO not supported: ${lto_error}")
endif()

# ─── Options ──────────────────────────────────────────────────────────────
option(BUILD_BENCHMARK "Build terllama-bench" ON)
option(BUILD_TESTING   "Build tests" ON)
option(BUILD_SERVER    "Build API server" ON)
option(BUILD_DOWNLOADER "Build model downloader" ON)

# ─── Dependencies ─────────────────────────────────────────────────────────
find_package(OpenMP QUIET)
find_package(CURL QUIET)
if(NOT CURL_FOUND)
    message(STATUS "libcurl: not found (downloader will use Python subprocess only)")
endif()

# ─── Architecture detection ───────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ARCH_X86_64 TRUE)
    message(STATUS "Arch: x86_64 (AVX2 auto-enabled)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(ARCH_ARM64 TRUE)
    message(STATUS "Arch: ARM64 (NEON auto-enabled)")
else()
    message(WARNING "Unknown arch: ${CMAKE_SYSTEM_PROCESSOR}, using scalar fallback")
endif()

# ─── Third-party includes ─────────────────────────────────────────────────
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(TESTS_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# ─── GigaToken (Rust tokenizer) ───────────────────────────────────────────
set(GIGATOKEN_DIR     ${THIRD_PARTY_DIR}/gigatoken)
set(GIGATOKEN_LIB     ${GIGATOKEN_DIR}/target/release/libgigatoken_rs.so)
set(GIGATOKEN_BUILD   ${CMAKE_BINARY_DIR}/libgigatoken_rs.so)

if(EXISTS ${GIGATOKEN_LIB})
    set(GIGATOKEN_FOUND YES)
else()
    set(GIGATOKEN_FOUND NO)
    message(STATUS "GigaToken .so: NOT FOUND — run 'cargo build --release -p gigatoken' in third_party/gigatoken/")
    message(STATUS "  Python tokenizer fallback will be used at runtime.")
endif()

# ─── Kernel object libraries (separate compile for ISA-specific flags) ────

# Scalar fallback (always)
add_library(terllama_kernel_scalar OBJECT src/kernel_scalar.cpp)
target_compile_options(terllama_kernel_scalar PRIVATE -O3 -ffast-math)

# AVX2 (x86_64 only)
if(ARCH_X86_64)
    add_library(terllama_kernel_avx2 OBJECT src/kernel_avx2.cpp)
    target_compile_options(terllama_kernel_avx2 PRIVATE -mavx2 -mfma -O3 -ffast-math)
    target_compile_definitions(terllama_kernel_avx2 PRIVATE __AVX2__)
else()
    # Dummy compilation on non-x86 targets (kernels use #ifdef guards)
    add_library(terllama_kernel_avx2 OBJECT src/kernel_avx2.cpp)
    target_compile_options(terllama_kernel_avx2 PRIVATE -O3)
endif()

# NEON (ARM64 only - header unavailable on x86)
if(ARCH_ARM64)
    add_library(terllama_kernel_neon OBJECT src/kernel_neon.cpp)
    target_compile_options(terllama_kernel_neon PRIVATE -march=armv8-a+fp+simd -O3 -ffast-math)
    target_compile_definitions(terllama_kernel_neon PRIVATE __ARM_NEON)
endif()

# MoTE kernel (always — compiles with inline dispatch)
add_library(terllama_kernel_mote OBJECT src/kernel_mote.cpp)
target_compile_options(terllama_kernel_mote PRIVATE -O3)
target_include_directories(terllama_kernel_mote PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Dispatcher (always)
add_library(terllama_dispatcher OBJECT src/dispatcher.cpp)
target_compile_options(terllama_dispatcher PRIVATE -O3)
target_include_directories(terllama_dispatcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ─── Core object library (shared between terllama + tests) ────────────────
add_library(terllama_core OBJECT
    src/main.cpp
    src/cli/commands.cpp
    src/core/inference.cpp
    src/core/logger.cpp
    src/core/tokenizer.cpp
    src/core/gigatoken_wrapper.cpp
    src/mote_builder.cpp
    src/server/handlers.cpp
    src/server.cpp
    src/downloader.cpp
    src/gguf_loader.cpp
    src/convert/hf_download.cpp
    src/convert/safetensors.cpp
    src/convert/als_decompose.cpp
    src/convert/export.cpp
)
target_include_directories(terllama_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}
)
target_compile_options(terllama_core PRIVATE -O3)
target_link_libraries(terllama_core PRIVATE
    pthread
    dl
)
if(OpenMP_CXX_FOUND)
    # OpenMP::OpenMP_CXX already carries the correct per-compiler flags
    # (-fopenmp for GCC/Clang, -Xclang -fopenmp for AppleClang).
    target_link_libraries(terllama_core PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CURL_FOUND)
    target_link_libraries(terllama_core PRIVATE CURL::libcurl)
    target_compile_definitions(terllama_core PRIVATE TERLLAMA_HAVE_CURL)
endif()

# ─── Main binary ──────────────────────────────────────────────────────────
add_executable(terllama
    $<TARGET_OBJECTS:terllama_core>
    $<TARGET_OBJECTS:terllama_kernel_scalar>
    $<TARGET_OBJECTS:terllama_kernel_avx2>
    $<TARGET_OBJECTS:terllama_kernel_mote>
    $<TARGET_OBJECTS:terllama_dispatcher>
)
if(ARCH_ARM64)
    target_sources(terllama PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
endif()
target_include_directories(terllama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}
)
target_link_libraries(terllama PRIVATE
    pthread
    dl
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(terllama PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CURL_FOUND)
    target_link_libraries(terllama PRIVATE CURL::libcurl)
endif()

# Post-build: copy GigaToken .so next to the binary so dlopen finds it
if(EXISTS ${GIGATOKEN_LIB})
    add_custom_command(TARGET terllama POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GIGATOKEN_LIB} ${CMAKE_BINARY_DIR}/libgigatoken_rs.so
        COMMENT "Copying libgigatoken_rs.so → ${CMAKE_BINARY_DIR}/"
    )
endif()

# ─── Benchmark binary ─────────────────────────────────────────────────────
if(BUILD_BENCHMARK)
    add_executable(terllama-bench
        src/benchmark.cpp
        src/core/logger.cpp
        $<TARGET_OBJECTS:terllama_kernel_scalar>
        $<TARGET_OBJECTS:terllama_kernel_avx2>
        $<TARGET_OBJECTS:terllama_kernel_mote>
        $<TARGET_OBJECTS:terllama_dispatcher>
    )
    if(ARCH_ARM64)
        target_sources(terllama-bench PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
    endif()
    target_include_directories(terllama-bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${THIRD_PARTY_DIR}
    )
    target_link_libraries(terllama-bench PRIVATE
        pthread
        dl
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(terllama-bench PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CURL_FOUND)
        target_link_libraries(terllama-bench PRIVATE CURL::libcurl)
    endif()
endif()

# ─── Tests ────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Install ──────────────────────────────────────────────────────────────
install(TARGETS terllama RUNTIME DESTINATION bin)
if(BUILD_BENCHMARK)
    install(TARGETS terllama-bench RUNTIME DESTINATION bin)
endif()
install(DIRECTORY web/ DESTINATION share/terllama/web)

# Install GigaToken shared library alongside the binary
if(EXISTS ${GIGATOKEN_LIB})
    install(FILES ${GIGATOKEN_LIB} DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
    message(STATUS "GigaToken .so: will install to bin/")
endif()

# ─── Build summary ────────────────────────────────────────────────────────
message(STATUS "╔══════════════════════════════════════════╗")
message(STATUS "║        Terllama Build Configuration      ║")
message(STATUS "╠══════════════════════════════════════════╣")
message(STATUS "║  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "║  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "║  Arch:          ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "║  LTO/IPO:       ${lto_supported}")
message(STATUS "║  OpenMP:        ${OpenMP_CXX_FOUND}")
message(STATUS "║  libcurl:       ${CURL_FOUND}")
message(STATUS "║  GigaToken:     ${GIGATOKEN_FOUND}")
message(STATUS "║  Build tests:   ${BUILD_TESTING}")
message(STATUS "║  Benchmark:     ${BUILD_BENCHMARK}")
message(STATUS "╚══════════════════════════════════════════╝")

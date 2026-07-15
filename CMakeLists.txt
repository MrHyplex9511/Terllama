cmake_minimum_required(VERSION 3.16)
project(terllama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────
option(BUILD_BENCHMARK "Build the multi-kernel benchmark" ON)

# ─── OpenMP ───────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ─── Architecture detection & flags ───────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    # Check for Intel compiler vs GCC/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Enable baseline x86-64; target-annotated functions handle the rest
        set(ARCH_FLAGS "-march=x86-64 -mtune=generic")
        # Optional: compile with full native optimization
        # set(ARCH_FLAGS "-march=native")
    elseif(MSVC)
        set(ARCH_FLAGS "/arch:SSE2")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(ARCH_FLAGS "-march=armv8-a+fp+simd")
    endif()
endif()

# ─── Main binary ──────────────────────────────────────────────────────────
add_executable(terllama
    main.cpp
)

target_include_directories(terllama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(terllama PRIVATE ${ARCH_FLAGS} -O3)
target_link_libraries(terllama PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(terllama PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_DEFINITIONS>)

# ─── Benchmark binary (optional) ─────────────────────────────────────────
if(BUILD_BENCHMARK)
    add_executable(terllama-bench
        benchmark.cpp
    )
    target_include_directories(terllama-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(terllama-bench PRIVATE ${ARCH_FLAGS} -O3)
    target_link_libraries(terllama-bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(terllama-bench PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# ─── Install ──────────────────────────────────────────────────────────────
install(TARGETS terllama RUNTIME DESTINATION bin)
if(BUILD_BENCHMARK)
    install(TARGETS terllama-bench RUNTIME DESTINATION bin)
endif()

# ─── Print build summary ─────────────────────────────────────────────────
message(STATUS "Terllama build configuration:")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Architecture:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Flags:            ${ARCH_FLAGS}")
message(STATUS "  OpenMP:           ${OpenMP_CXX_FOUND}")
message(STATUS "  Benchmark:        ${BUILD_BENCHMARK}")

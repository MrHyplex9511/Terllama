# ─── Catch2 ────────────────────────────────────────────────────────────────
#   catch_amalgamated.hpp — header (included by test sources)
#   catch_amalgamated.cpp — implementation + main() (compiled once, linked to
#                           each test binary)
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# LTO inlining drops Catch2 vtable symbols; disable for test binaries
set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

# ─── Test: kernel correctness ─────────────────────────────────────────────
add_executable(test_kernels
    test_kernels.cpp
    catch_amalgamated.cpp
    $<TARGET_OBJECTS:terllama_kernel_scalar>
    $<TARGET_OBJECTS:terllama_kernel_avx2>
    $<TARGET_OBJECTS:terllama_dispatcher>
)
if(ARCH_ARM64)
    target_sources(test_kernels PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
endif()
target_include_directories(test_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CATCH_INCLUDE_DIR}
)
target_link_libraries(test_kernels PRIVATE
    OpenMP::OpenMP_CXX
    pthread
)
add_test(NAME test_kernels COMMAND test_kernels)

# ─── Test: inference ops ──────────────────────────────────────────────────
add_executable(test_inference
    test_inference.cpp
    catch_amalgamated.cpp
    ${CMAKE_SOURCE_DIR}/src/core/inference.cpp
    ${CMAKE_SOURCE_DIR}/src/core/logger.cpp
    $<TARGET_OBJECTS:terllama_kernel_scalar>
    $<TARGET_OBJECTS:terllama_kernel_avx2>
    $<TARGET_OBJECTS:terllama_dispatcher>
)
if(ARCH_ARM64)
    target_sources(test_inference PRIVATE $<TARGET_OBJECTS:terllama_kernel_neon>)
endif()
target_include_directories(test_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CATCH_INCLUDE_DIR}
)
target_link_libraries(test_inference PRIVATE
    OpenMP::OpenMP_CXX
    pthread
)
add_test(NAME test_inference COMMAND test_inference)

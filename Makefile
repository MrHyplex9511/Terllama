# ═══════════════════════════════════════════════════════════════════════════
# Terllama Makefile
# ═══════════════════════════════════════════════════════════════════════════
# Each kernel compiled separately with its own -march flags.
# Dispatcher uses weak symbols: missing kernel .o → skip.
#
# Usage:
#   make all          # Build terllama + benchmark
#   make terllama     # Inference binary
#   make bench        # Benchmark
#   make clean
#   make help
# ═══════════════════════════════════════════════════════════════════════════

CXX         := g++
CXXFLAGS    := -std=c++17 -O3 -fopenmp -Wall -Wextra -Wno-unknown-pragmas -Wno-address
LDFLAGS     := -lm -fopenmp

SRC_DIR     := src
BUILD_DIR   := build
TARGET      := terllama
BENCH       := terllama-bench

# ISA detection
try_isa = $(shell echo 'int x;' | $(CXX) -x c++ $(1) -c - -o /dev/null 2>/dev/null && echo yes || echo no)

HAVE_SSE42  := $(call try_isa,-msse4.2)
HAVE_AVX    := $(call try_isa,-mavx)
HAVE_AVX2   := $(call try_isa,-mavx2 -mfma)
HAVE_AVX512 := $(call try_isa,-mavx512f -mavx512dq)

# Kernel objects
KERNEL_OBJS := $(BUILD_DIR)/kernel_scalar.o $(BUILD_DIR)/kernel_avx2.o

ifneq (,$(filter aarch64 arm64, $(shell uname -m)))
  KERNEL_OBJS += $(BUILD_DIR)/kernel_neon.o
endif

COMMON_OBJS := $(BUILD_DIR)/dispatcher.o

.PHONY: all build-terllama build-bench clean help

all: $(BUILD_DIR) $(TARGET) $(BENCH)

build-terllama: $(BUILD_DIR) $(TARGET)

build-bench: $(BUILD_DIR) $(BENCH)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Scalar fallback (any CPU)
$(BUILD_DIR)/kernel_scalar.o: $(SRC_DIR)/kernel_scalar.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# AVX2+FMA — primary optimized kernel
ifeq ($(HAVE_AVX2),yes)
$(BUILD_DIR)/kernel_avx2.o: $(SRC_DIR)/kernel_avx2.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma -c $< -o $@
else
$(BUILD_DIR)/kernel_avx2.o: $(SRC_DIR)/kernel_avx2.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif

# NEON for ARM64
$(BUILD_DIR)/kernel_neon.o: $(SRC_DIR)/kernel_neon.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -march=armv8-a+fp+simd -c $< -o $@

# Dispatcher
$(BUILD_DIR)/dispatcher.o: $(SRC_DIR)/dispatcher.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main inference binary
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/benchmark.o: $(SRC_DIR)/benchmark.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(BUILD_DIR)/main.o $(COMMON_OBJS) $(KERNEL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BENCH): $(BUILD_DIR)/benchmark.o $(COMMON_OBJS) $(KERNEL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(BENCH)

help:
	@echo "Terllama Build"
	@echo "C++ compiler:  $(CXX)"
	@echo "C++ flags:     $(CXXFLAGS)"
	@echo "AVX2+FMA:      $(HAVE_AVX2)"
	@echo "Targets:"
	@echo "  make all            Build everything"
	@echo "  make terllama       Inference binary"
	@echo "  make bench          Benchmark"
	@echo "  make clean"

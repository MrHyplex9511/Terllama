# ═══════════════════════════════════════════════════════════════════════════
# Terllama Makefile
# ═══════════════════════════════════════════════════════════════════════════
#
# Targets:
#   make all          Build terllama + terllama-bench
#   make terllama     Main binary (CLI + server + downloader)
#   make bench        Benchmark binary
#   make clean
# ═══════════════════════════════════════════════════════════════════════════

CXX         := g++
CXXFLAGS    := -std=c++17 -O3 -fopenmp -flto -Wall -Wextra -Wno-unknown-pragmas -Wno-address
LDFLAGS     := -lm -fopenmp -lpthread -flto

SRC_DIR     := src
BUILD_DIR   := build
TARGET      := terllama
BENCH       := terllama-bench
THIRD_PARTY := third_party

# ISA detection
try_isa = $(shell echo 'int x;' | $(CXX) -x c++ $(1) -c - -o /dev/null 2>/dev/null && echo yes || echo no)

HAVE_AVX2   := $(call try_isa,-mavx2 -mfma)

# Kernel objects
KERNEL_OBJS := $(BUILD_DIR)/kernel_scalar.o $(BUILD_DIR)/kernel_avx2.o $(BUILD_DIR)/kernel_mote.o

ifneq (,$(filter aarch64 arm64, $(shell uname -m)))
  KERNEL_OBJS += $(BUILD_DIR)/kernel_neon.o
endif

# All source objects for the main binary
MAIN_OBJS := $(BUILD_DIR)/main.o \
             $(BUILD_DIR)/commands.o \
             $(BUILD_DIR)/server.o \
             $(BUILD_DIR)/handlers.o \
             $(BUILD_DIR)/downloader.o \
             $(BUILD_DIR)/dispatcher.o \
             $(BUILD_DIR)/gguf_loader.o \
             $(BUILD_DIR)/inference.o \
             $(BUILD_DIR)/logger.o \
             $(BUILD_DIR)/tokenizer.o \
             $(BUILD_DIR)/mote_builder.o

.PHONY: all build-terllama build-bench clean help

all: $(BUILD_DIR) $(TARGET) $(BENCH)

build-terllama: $(BUILD_DIR) $(TARGET)

build-bench: $(BUILD_DIR) $(BENCH)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Scalar fallback
$(BUILD_DIR)/kernel_scalar.o: $(SRC_DIR)/kernel_scalar.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# AVX2+FMA
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

# MoTE kernel
$(BUILD_DIR)/kernel_mote.o: $(SRC_DIR)/kernel_mote.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# Dispatcher
$(BUILD_DIR)/dispatcher.o: $(SRC_DIR)/dispatcher.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CLI commands
$(BUILD_DIR)/commands.o: $(SRC_DIR)/cli/commands.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -I$(THIRD_PARTY) -c $< -o $@

# Main CLI (dispatches subcommands)
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -I$(THIRD_PARTY) -c $< -o $@

# Server (API + web)
$(BUILD_DIR)/server.o: $(SRC_DIR)/server.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -I$(THIRD_PARTY) -c $< -o $@

# Server handlers (route implementations — separated from server.cpp)
$(BUILD_DIR)/handlers.o: $(SRC_DIR)/server/handlers.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -I$(THIRD_PARTY) -c $< -o $@

# Downloader (HuggingFace model pull)
$(BUILD_DIR)/downloader.o: $(SRC_DIR)/downloader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# MoTE builder
$(BUILD_DIR)/mote_builder.o: $(SRC_DIR)/mote_builder.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# GGUF loader
$(BUILD_DIR)/gguf_loader.o: $(SRC_DIR)/gguf_loader.cpp $(SRC_DIR)/gguf_loader.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Core inference
$(BUILD_DIR)/inference.o: $(SRC_DIR)/core/inference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# Logger
$(BUILD_DIR)/logger.o: $(SRC_DIR)/core/logger.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# Tokenizer
$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/core/tokenizer.cpp $(SRC_DIR)/core/tokenizer.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

# Benchmark
$(BUILD_DIR)/benchmark.o: $(SRC_DIR)/benchmark.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main binary (CLI + server + downloader + kernels)
$(TARGET): $(MAIN_OBJS) $(KERNEL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Benchmark binary
$(BENCH): $(BUILD_DIR)/benchmark.o $(BUILD_DIR)/dispatcher.o $(BUILD_DIR)/logger.o $(KERNEL_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(BENCH)

help:
	@echo "Terllama Build"
	@echo "Compiler:      $(CXX)"
	@echo "Flags:         $(CXXFLAGS)"
	@echo "AVX2+FMA:      $(HAVE_AVX2)"
	@echo ""
	@echo "Targets:"
	@echo "  make all            Build everything"
	@echo "  make terllama       Main binary (CLI + server + downloader)"
	@echo "  make bench          Benchmark binary"
	@echo "  make clean"

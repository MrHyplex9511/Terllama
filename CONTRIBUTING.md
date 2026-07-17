# Contributing to Terllama

## Coding Style

### C++

- **Indentation:** 4 spaces (no tabs)
- **Braces:** Allman style (braces on next line) for functions; K&R (same line) for control flow
- **Naming:**
  - `snake_case` for functions and variables
  - `PascalCase` for types/classes/structs
  - `UPPER_CASE` for macros and constants
- **Comments:** `//` for single-line, `/* */` for block comments. Keep comments minimal and high-level — explain *why*, not *what*
- **Header order:** Own header → standard library → third-party → project headers (alphabetical within each group)
- **Line length:** 100 characters max
- **Include guards:** `#pragma once` (not `#ifndef`)

We include a `.clang-format` config at the repo root. Run before committing:

```bash
clang-format -i src/*.cpp src/*.h third_party/*.h
```

### Python (scripts/)

- **Indentation:** 4 spaces
- **Follow PEP 8**
- **Docstrings:** Triple-quote, module-level docstring explaining purpose
- **Type hints:** Encouraged for function signatures

### HTML/CSS/JS (web/)

- **Indentation:** 2 spaces
- **CSS:** Variables in `:root`, class names in `kebab-case`
- **JS:** `camelCase` for variables and functions, `'use strict'`, IIFE wrapper

## Branch Strategy

```
main          — Production-ready, reviewed, tagged releases
├─ feat/*     — New features (e.g., feat/avx512-kernel)
├─ fix/*      — Bug fixes (e.g., fix/rope-off-by-one)
├─ bench/*    — Benchmarking and performance work
└─ docs/*     — Documentation changes
```

- `main` must always compile and pass basic correctness checks
- Feature branches branch from `main`, merge back via PR
- Keep commits small and focused. One logical change per commit
- Commit messages: present tense, imperative mood (`"Add AVX-512 kernel"` not `"Added AVX-512"`)

## How to Add a New ISA Kernel

1. **Create the kernel function** in a new `src/kernel_<isa>.cpp` or add to existing dispatch:

   ```cpp
   // src/kernel_<isa>.cpp
   __attribute__((target("<isa-flags>")))
   void ternary_mul_<isa>(const uint32_t* const* term_data,
                           const int* alpha_exps,
                           int n_active, int out_f, int in_f,
                           const float* input, float* output) {
       // Your SIMD implementation
   }
   ```

2. **Add enum** in `kernel_dispatch.h`:

   ```cpp
   enum class CPUArch { ..., X86_64_<ISA> };
   ```

3. **Register in dispatcher** in `kernel_dispatch.h`:

   ```cpp
   case CPUArch::X86_64_<ISA>:
       ternary_mul_<isa>(...);
       return;
   ```

4. **Add to `detect_cpu_arch()`** — check `__builtin_cpu_supports()` for the new ISA.

5. **Add to Makefile** — compile with appropriate `-m<isa>` flags:

   ```makefile
   $(BUILD_DIR)/kernel_<isa>.o: src/kernel_<isa>.cpp | $(BUILD_DIR)
       $(CXX) $(CXXFLAGS) -m<isa> -c $< -o $@
   ```

6. **Add to `validate_all_kernels()`** — include the new kernel in the multi-kernel validator.

7. **Update CMakeLists.txt** — add object library target with ISA-specific compile flags.

8. **Benchmark** — run `make bench && ./terllama-bench` and compare throughput.

## Build Instructions

### Make (primary)

```bash
# Build everything
make

# Main binary only
make terllama

# Benchmark only
make bench

# Clean
make clean
```

**Dependencies:** C++17 compiler (g++ 9+ or clang 12+), OpenMP, make, pthreads.

ISA detection is automatic. Build output goes to `build/`.

### CMake (alternative)

```bash
mkdir build-cmake && cd build-cmake
cmake .. -DBUILD_BENCHMARK=ON
make -j$(nproc)
```

Options:

| Option | Default | Description |
|---|---|---|
| `BUILD_BENCHMARK` | `ON` | Build `terllama-bench` |
| `BUILD_SERVER` | `ON` | Build API server |
| `BUILD_DOWNLOADER` | `ON` | Build model downloader |

### Docker

```bash
docker build -t terllama .
```

See `Dockerfile` for multi-stage build details.

## Test Requirements

- **Correctness:** `make bench && ./terllama-bench` runs all kernels against the scalar reference. All kernels must produce `max_err < 1e-4`.
- **Export:** Python export script must produce deterministic output (uses `torch.manual_seed(42)`).
- **Server:** Start with `./terllama serve --port 8375`, test endpoints with curl.
- **Memory:** Valgrind or `-fsanitize=address` for new memory-heavy code.

Run the multi-kernel validator:

```bash
./terllama-bench --validate
```

## Pull Request Process

1. Open an issue describing the change (bug, feature, performance) before starting work.
2. Fork the repo and create a feature branch.
3. Make your changes. Keep commits small and focused.
4. Update docs if the API or behavior changes.
5. Run `make clean && make` — must compile without warnings.
6. Run `./terllama-bench` — all kernels must pass validation.
7. Open a PR against `main`. Describe what changed and why.
8. PRs need at least one review before merging.
9. Squash-merge into `main`.

## Code of Conduct

- Be constructive and respectful.
- Focus on the code, not the person.
- Performance and correctness over elegance.

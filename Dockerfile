# ═══════════════════════════════════════════════════════════════════════════
# Terllama — Multi-stage Docker build
# ═══════════════════════════════════════════════════════════════════════════
# Build:
#   docker build -t terllama .
#
# Run:
#   docker run -p 8375:8375 -v ~/.terllama:/home/terllama/.terllama terllama
#
# The server listens on port 8375 with an OpenAI-compatible API.
# Mount ~/.terllama to persist downloaded models across container restarts.
# ═══════════════════════════════════════════════════════════════════════════

# ─── Build stage ──────────────────────────────────────────────────────────
FROM alpine:3.19@sha256:b58899f069c47216f6002a6850143dc6fae0d35eb8b0df9300bbe6327b9c2171 AS builder

# C++ build deps
RUN apk add --no-cache build-base curl-dev linux-headers

# Rust + Python-dev for GigaToken (pyo3 needs libpython)
RUN apk add --no-cache python3-dev rust cargo
# Remove Python-exclusive pyo3 output from the Rust build
ENV PYO3_PRINT_CONFIG=0

WORKDIR /src
COPY . .

# Build GigaToken C API library (uses PYO3 + Python dev)
RUN cargo build --release -p gigatoken && \
    cp target/release/libgigatoken_rs.so /usr/local/lib/

# Static link: all .cpp in src/ except benchmark
# Runtime dispatch via weak symbols in dispatcher
RUN g++ -std=c++17 -O3 -fopenmp -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE \
    -I. -Ithird_party \
    src/main.cpp src/server.cpp src/downloader.cpp \
    src/dispatcher.cpp src/kernel_scalar.cpp src/kernel_avx2.cpp \
    -o /terllama -lm -fopenmp -lpthread -lcurl \
    -pie -Wl,-z,relro,-z,now \
    && strip /terllama

# ─── Runtime stage ────────────────────────────────────────────────────────
FROM alpine:3.19@sha256:b58899f069c47216f6002a6850143dc6fae0d35eb8b0df9300bbe6327b9c2171

# Runtime deps: OpenMP, libcurl, Python3 (for tokenizer helpers fallback)
RUN apk add --no-cache libgomp libcurl python3 py3-pip

# GigaToken C API library — dlopen'd by terllama at runtime
COPY --from=builder /usr/local/lib/libgigatoken_rs.so /usr/local/lib/

COPY --from=builder /terllama /usr/local/bin/terllama
COPY web /usr/local/share/terllama/web

# Run as a non-root user (least privilege). The models volume lives under
# $HOME/.terllama and is chowned below; a host bind-mount into it must be
# owned by uid 10001 (e.g. `docker run -v ~/.terllama:/home/terllama/.terllama`).
RUN adduser -D -u 10001 terllama \
    && mkdir -p /home/terllama/.terllama \
    && chown -R terllama:terllama /home/terllama/.terllama
USER terllama

ENV HOME=/home/terllama
ENV TERLLAMA_MODEL_DIR=/home/terllama/.terllama/models
ENV TERLLAMA_WEB_DIR=/usr/local/share/terllama/web

EXPOSE 8375
VOLUME /home/terllama/.terllama

CMD ["terllama", "serve", "--port", "8375"]

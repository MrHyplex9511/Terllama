# ═══════════════════════════════════════════════════════════════════════════
# Terllama — Multi-stage Docker build
# ═══════════════════════════════════════════════════════════════════════════
# Build:
#   docker build -t terllama .
#
# Run:
#   docker run -p 11434:11434 -v ~/.terllama:/root/.terllama terllama
#
# The server listens on port 11434 with an OpenAI-compatible API.
# Mount ~/.terllama to persist downloaded models across container restarts.
# ═══════════════════════════════════════════════════════════════════════════

# ─── Build stage ──────────────────────────────────────────────────────────
FROM alpine:3.19 AS builder

RUN apk add --no-cache build-base curl-dev linux-headers

WORKDIR /src
COPY . .

# Static link: all .cpp in src/ except benchmark
# Runtime dispatch via weak symbols in dispatcher
RUN g++ -std=c++17 -O3 -fopenmp -I. -Ithird_party \
    src/main.cpp src/server.cpp src/downloader.cpp \
    src/dispatcher.cpp src/kernel_scalar.cpp src/kernel_avx2.cpp \
    -o /terllama -lm -fopenmp -lpthread -lcurl \
    && strip /terllama

# ─── Runtime stage ────────────────────────────────────────────────────────
FROM alpine:3.19

# Runtime deps: OpenMP, libcurl, Python3 (for tokenizer helpers)
RUN apk add --no-cache libgomp libcurl python3 py3-pip

COPY --from=builder /terllama /usr/local/bin/terllama
COPY web /usr/local/share/terllama/web

EXPOSE 11434
VOLUME /root/.terllama

ENV TERLLAMA_MODEL_DIR=/root/.terllama/models
ENV TERLLAMA_WEB_DIR=/usr/local/share/terllama/web

CMD ["terllama", "serve", "--port", "11434"]

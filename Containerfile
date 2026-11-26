# ==============================================================================
# Mycelial Cellular Vector Engine — Fedora 45 (Bazzite 45 Compatible) Container
# ==============================================================================
ARG BASE_IMAGE=registry.fedoraproject.org/fedora-minimal:45

# Stage 1: 使用官方 Fedora 45 完整環境編譯 Rust 裸機細胞與母體引擎（本機免裝 Rust）
FROM registry.fedoraproject.org/fedora:45 AS builder

RUN dnf install -y --setopt=install_weak_deps=False \
        rust \
        cargo \
        rust-std-static-wasm32-unknown-unknown \
        gcc \
        glibc-devel \
    && dnf clean all

WORKDIR /build
COPY Cargo.toml ./
COPY src ./src

# 1. 先將瀏覽器前端核心編譯為真正的 `no_std` Rust WebAssembly (`wasm32-unknown-unknown`) 二進位模組
RUN rustc --target wasm32-unknown-unknown -O --crate-type=cdylib src/wasm_frontend.rs -o src/mycelial_client.wasm

# 2. 再編譯後端母體引擎（透過 `include_bytes!("mycelial_client.wasm")` 將前端 Rust Wasm 封裝於單一執行檔內）
RUN cargo build --release

# Stage 2: 使用與 Bazzite 45 同源之 Fedora 45 Minimal 作為極簡執行環境
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Mycelial Cellular Vector Engine"
LABEL org.opencontainers.image.description="Bare-metal micro-cell dataflow, 500-cell spiking neural graph & read-only HugePages vector engine on Fedora 45"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=builder /build/target/release/mycelial-vector-engine /usr/local/bin/mycelial-vector-engine

EXPOSE 8080 9090

ENTRYPOINT ["/usr/local/bin/mycelial-vector-engine"]


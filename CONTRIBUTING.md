# Contributing to SporeFlow

SporeFlow is intentionally small (4 Rust files, 3 crate dependencies: `libc`, `kvm-ioctls`, `kvm-bindings`) so anyone can read the whole codebase in 15 minutes and ship meaningful performance or architecture improvements.

## Building & Testing (Zero Host Rust Setup Needed)

You do not need `rustc` or `cargo` installed on your host. The `Containerfile` compiles both the `wasm32-unknown-unknown` frontend (`src/sporeflow_client.wasm`) and the release binary inside Fedora 45:

```bash
# Build with Podman or Docker
podman build -t sporeflow:dev -f Containerfile .

# Run CLI benchmark mode
podman run --rm sporeflow:dev

# Run Web Studio (:8080) + Raw Binary Conduit (:9090)
podman run --rm -it -p 8080:8080 -p 9090:9090 --name sporeflow-dev sporeflow:dev --web

# Run built-in pure Rust client roundtrip check against the running container
podman exec -it sporeflow-dev sporeflow --rust-client
```

If you have Rust installed locally (`rustup target add wasm32-unknown-unknown`):

```bash
rustc --target wasm32-unknown-unknown -O --crate-type=cdylib src/wasm_frontend.rs -o src/sporeflow_client.wasm
cargo build --release
```

## Coding Guidelines

- Keep `src/wasm_frontend.rs` strictly `#![no_std]` with zero external crates.
- Keep zero compiler warnings (`cargo build --release` and `rustc --target wasm32-unknown-unknown` must both compile cleanly).
- Avoid adding heavy web frameworks or serialization crates unless there is a clear, benchmarked win—minimal dependency footprint is a core design goal.
- Include before/after microsecond measurements in performance PRs (specify CPU model and container runtime).

## Good Areas for PRs

Check the [open issues](https://github.com/exefncs2/sporeflow/issues) tagged `help wanted` or `good first issue`, especially:
1. Linux ARM64 multi-arch builds & NEON SIMD vector kernels.
2. `/dev/shm` lock-free shared-memory ring IPC alongside `:9090`.
3. 64-bit KVM Long Mode polling cells for true hardware isolation.
4. Int8 / FP16 quantized vector pool operators.
5. Prometheus `/metrics` endpoint & CPU core affinity (`sched_setaffinity`).

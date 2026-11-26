# Mycelial Cellular Vector Engine (`mycelial-vector-engine` v0.4.0)

基於 **Fedora 45 / Bazzite 45 (KVM & User-Space Arena)** 之微秒級「計算機細胞學（Computational Cytology）」、**雙頁面互動式工作台（真實加速通道控制台 ＋ 原速度視覺比較實驗室）**，以及支援外部專案直接進出資料的 **REST / Binary API (`:8080`) 與零 HTTP 純二進位直連 Socket (`:9090`)**。

---

## 一、 核心架構與外部資料進出閉環

本容器提供完整的「**資料進（Data In）➔ 動態算力流（並流／串流／矩陣流）➔ 資料出（Data Out）**」閉環，外部專案無須撰寫複雜的 Pipeline 即可直接掛載加速：

1. **📥 資料進（灌入外部自訂基底向量庫）— `POST http://localhost:8080/v1/pool`**：
   - 支援 **JSON 向量陣列** 或 **Raw Float32 二進位串流 (`application/octet-stream`)**。
   - 引擎自動完成 L2 正規化並寫入 50MB 大頁記憶體，**寫入完畢當場呼叫 `mprotect(..., PROT_READ)` 在硬體頁表層級鎖死唯讀**（防竄改、零拷貝共享）。
2. **🎛️ 參數控（網頁拖曳方塊即時切換算力流與通道）— `http://localhost:8080`**：
   - **頁面 1（🚀 真實加速通道 Production Conduit）**：100% 零比較包袱，自由設定通過的**傳輸介質（64B Zero-Parse / SHM / Binary / JSON）**、**真實算力拓撲（🧮 8×8 矩陣流一次送 6 筆 / ⚡ 500 核 3 層神經串流 / 🌊 128 核平行並流）**與 **1~8 條費洛蒙動態通道**。
   - **頁面 2（🎬 原速度視覺比較實驗室 Benchmark Arena）**：提供與「🐢 傳統 JSON + 逐筆 Heap 資料搬移原速度 (1.00x)」同場競速的雙賽道動態視覺比較圖。
3. **📤 資料出 A（通用 HTTP / Binary 加速輸出接口）— `POST http://localhost:8080/v1/conduit`**：
   - 外部程式傳入單筆或微批次查詢向量（支援 JSON 或 `application/octet-stream`），容器直接套用當前網頁設定的算力流極速運算並回傳 Top-K 匹配 ID 與餘弦相似度，算完瞬間由巨噬細胞在 `~11 µs` 內將 `0x1000~0x303F` 物理抹除為 `0x00`。
4. **⚡ 資料出 B（零 HTTP 純二進位直連 Socket 接口）— `tcp://localhost:9090`**：
   - 免除一切 HTTP Header 與 JSON 解析開銷！外部程式連上 TCP `:9090` 後，直接寫入 **`3,072 Bytes`（6 筆 × 128 維 `f32` Little-Endian，或單筆 `512 Bytes`）**，即刻收回 **`56 Bytes` 二進位結果封包（`[best_ids: u32 * 6, best_scores: f32 * 6, compute_us: f32, zero_us: f32]`）**。

---

## 二、 實際程式碼結構與檔案對照（三檔分離 ＋ 外部接入腳本）

| 檔案路徑 | 職責與實際實作內容 |
| :--- | :--- |
| `src/config.rs` | 定義 16KB 細胞槽位常數（`0x0000`、`0x1000`、`0x2000`、`0x3000`）、`:8080` Web/API 埠與 `:9090` 純二進位直連埠、以及 64B 對齊之 `CellReceptorInput`（`3,136 Bytes <= 4096 Bytes`）。 |
| `src/core.rs` | 實作 `SharedReadOnlyVectorPool::ingest_custom_vectors`（自訂向量池灌入與 `mprotect(PROT_READ)` 硬體唯讀鎖定）、四大介質轉檔引擎、動態通道費洛蒙路由表，以及並流／串流／矩陣流三大零拷貝算力核心。 |
| `src/main.rs` | 實作雙頁面 Web 伺服器（`:8080`）、`POST /v1/pool`、`POST /v1/conduit`、以及 `spawn_direct_binary_conduit_server`（`:9090` 零 HTTP 純二進位直連 Socket 伺服器）。 |
| `src/wasm_frontend.rs` | `no_std` 純 Rust WebAssembly 前端核心（編譯為 `1.3 KB` 的 `wasm32-unknown-unknown` 模組），於瀏覽器端直接執行文字切塊與 `3,072B` 受體封包封裝。 |
| `compose.yaml` | 同機 Sidecar 加速通道範本（一鍵開啟 `:8080` 網頁/API 與 `:9090` 純二進位直連埠）。 |
| `.github/workflows/publish-image.yml` | GitHub Actions 自動化建置並發布公開映像檔至 `ghcr.io`（亦支援選配同步推送至 Docker Hub）。 |
| `Containerfile` | 純 Fedora 45 多階段建置（Stage 1 編譯 `wasm32-unknown-unknown` 與 Release Binary；Stage 2 封裝至 `fedora-minimal:45`，開放 `EXPOSE 8080 9090`）。 |

---

## 三、 一行指令啟動與外部專案接入（Docker / Podman）

### 1. 直接拉取公開映像檔啟動（推薦，免編譯、免裝 Rust）
```bash
# 掛載 /dev/kvm（Linux 主機完整版）
docker run --rm -it --device /dev/kvm -p 8080:8080 -p 9090:9090 ghcr.io/exefncs2/mycelial-vector-engine:latest --web

# 免特權快速體驗（macOS / Windows / 無 KVM 雲端主機，自動降級 User-Space 沙盒）
docker run --rm -it -p 8080:8080 -p 9090:9090 ghcr.io/exefncs2/mycelial-vector-engine:latest --web
```
或使用同機 Sidecar 範本 [compose.yaml](./compose.yaml)：
```bash
docker compose up -d
```

啟動後打開瀏覽器造訪 **`http://localhost:8080`**（內建 `1.3 KB` `no_std` Rust WebAssembly 前端 `wasm32-unknown-unknown`）：
- **頁面 1（🚀 真實加速通道）**：支援拖入任意文字檔（`.txt` / `.log` / `.md` / `.rs` / `.json`）觀察均分切塊與細胞/通道分配，一鍵灌入 `/v1/pool` 鎖死 `PROT_READ`，並支援 `:9090` 純 Rust 二進位直連測試。
- **頁面 2（🎬 原速度視覺比較實驗室）**：與傳統 JSON + 逐筆 Heap 資料搬移原速度進行雙賽道視覺競速。

### 2. 執行內建 100% 純 Rust 現實應用端實測（`--rust-client`）
```bash
docker exec -it mycelial-engine mycelial-vector-engine --rust-client
```

### 3. 本地從原始碼建置 Fedora 45 容器映像檔
```bash
podman build -t mycelial-vector-engine:fedora45 -f Containerfile .
podman run --rm -it --device /dev/kvm -p 8080:8080 -p 9090:9090 mycelial-vector-engine:fedora45 --web
```

### 4. 同機 Sidecar 適用邊界（實測說明）
- **最佳場景（淨賺 4.1 ~ 4.8 倍加速）**：同機 `localhost:9090` / 同一個 Docker Network Sidecar，處理數萬筆向量掃描或 6 筆微批次矩陣流（實測同機 TCP 往返開銷僅 `~57 µs`，佔比 `< 2.2%`）。
- **不適用場景**：跨網際網路公網呼叫（公網 RTT 數十毫秒會覆蓋掉算力優勢）或僅比對幾十筆向量的極小請求。

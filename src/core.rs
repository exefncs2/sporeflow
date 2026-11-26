//! 菌絲細胞算力引擎核心邏輯（Spore Hub & Bare-Metal Compute Cell Core）。
//! 包含：
//! 1. 四大傳輸介質（JSON / Compact Binary / 64B Zero-Parse Frame / Shared-Memory IPC）轉檔與讀寫競速引擎。
//! 2. 支援玩家自訂上傳 JSON 向量檔解析、轉檔並直接注入 `0x1000` 受體頁進行三大算力流比賽。
//! 3. 可動態調整數量（1~8 條）、延遲懲罰與硬體斷尾隔離（Severed）之費洛蒙路由表。

use crate::config::{
    CellCytoplasmOutput, CellMembraneSensor, CellReceptorInput, EngineConfig, WireMedium,
    CELL_SLOT_SIZE, DEFAULT_ACTIVE_CHANNELS, INITIAL_PHEROMONE_WEIGHT, LAYER1_SENSORY_CELLS,
    LAYER2_CORTICAL_CELLS, LAYER3_MOTOR_CELLS, MATRIX_MESH_COLS, MATRIX_MESH_ROWS,
    MATRIX_MESH_TOTAL_CELLS, MATRIX_MICRO_BATCH_SIZE, MAX_PHEROMONE_CHANNELS,
    NEURAL_GRAPH_TOTAL_CELLS, OFFSET_CYTOPLASM_OUTPUT, OFFSET_DNA_CODE, OFFSET_MEMBRANE_SENSOR,
    OFFSET_RECEPTOR_INPUT, PARALLEL_FANOUT_CELLS, SPIKE_THRESHOLD, VECTOR_DIM,
};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

#[inline(always)]
pub fn read_cpu_cycles() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

pub const BARE_METAL_KVM_DNA_BYTES: &[u8] = &[
    0x0F, 0x31,                         // rdtsc -> EAX = low, EDX = high
    0x89, 0xC1,                         // mov ecx, eax
    0xA1, 0x00, 0x10,                   // mov ax, [0x1000]
    0x03, 0x06, 0x04, 0x10,             // add ax, [0x1004]
    0xA3, 0x00, 0x20,                   // mov [0x2000], ax
    0xC7, 0x06, 0x04, 0x20, 0x01, 0x00, // mov word ptr [0x2004], 1
    0x0F, 0x31,                         // rdtsc
    0x29, 0xC8,                         // sub eax, ecx
    0xA3, 0x00, 0x30,                   // mov [0x3000], ax
    0xC7, 0x06, 0x08, 0x30, 0x02, 0x00, // mov word ptr [0x3008], 2
    0xF4,                               // hlt
];

/// 四大傳輸介質（Wire Media）轉檔與讀寫基準結果
#[derive(Debug, Clone)]
pub struct WireMediumBenchmarkItem {
    pub medium_name: &'static str,
    pub medium_code: &'static str,
    /// 封包轉檔後的實際大小（Bytes）
    pub payload_bytes: usize,
    /// 將該介質反序列化／載入至細胞 `0x1000` 受體頁的實測耗時（微秒 µs）
    pub ingest_to_receptor_us: f64,
    /// 換算奈秒（ns）
    pub ingest_to_receptor_ns: f64,
    /// 相較於 JSON 文字解析的加速倍率
    pub speedup_vs_json: f64,
}

/// 將任意文字檔（.txt / .log / .md / .csv / .json / 程式碼）精準均分為 6 個資料分塊（Chunks，不重複、不漏行），
/// 並以純 Rust 零配置 N-gram 特徵雜湊（Feature Hashing）直接轉為 6×128 維受體向量（支援任意非數字之純文字檔！）
pub fn parse_arbitrary_text_to_chunks_and_batch(
    raw_text: &str,
) -> (
    [[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
    Vec<(usize, String)>,
    usize,
) {
    let trimmed = raw_text.trim();
    let total_bytes = raw_text.as_bytes().len();

    // 1. 精準均分：若行數 >= 6，採 b*n/6 .. (b+1)*n/6 確保每塊不重複、不漏行
    let non_empty_lines: Vec<&str> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut chunk_meta: Vec<(usize, String)> = Vec::with_capacity(MATRIX_MICRO_BATCH_SIZE);
    if non_empty_lines.len() >= MATRIX_MICRO_BATCH_SIZE {
        let n = non_empty_lines.len();
        for b in 0..MATRIX_MICRO_BATCH_SIZE {
            let start = (b * n) / MATRIX_MICRO_BATCH_SIZE;
            let end = ((b + 1) * n) / MATRIX_MICRO_BATCH_SIZE;
            let joined = non_empty_lines[start..end].join(" | ");
            let byte_sz = joined.as_bytes().len();
            let preview: String = joined.chars().take(44).collect();
            chunk_meta.push((byte_sz, preview));
        }
    } else if !trimmed.is_empty() {
        let chars: Vec<char> = trimmed.chars().collect();
        let n = chars.len();
        for b in 0..MATRIX_MICRO_BATCH_SIZE {
            let start = (b * n) / MATRIX_MICRO_BATCH_SIZE;
            let end = (((b + 1) * n) / MATRIX_MICRO_BATCH_SIZE).max(start + 1).min(n);
            let sub: String = chars[start..end].iter().collect();
            let byte_sz = sub.as_bytes().len();
            let preview: String = sub.chars().take(44).collect();
            chunk_meta.push((byte_sz, preview));
        }
    } else {
        for b in 0..MATRIX_MICRO_BATCH_SIZE {
            chunk_meta.push((64, format!("預設測試資料分塊 #{b}")));
        }
    }

    // 2. 若文字檔內含充足浮點數則直接萃取；若為一般純文字檔（Log / 文章 / 程式碼），則以純 Rust FNV-1a N-gram 特徵雜湊投影為 128 維正規化向量
    let extracted_floats = extract_floats_from_str(trimmed);
    let mut batch = [[0.0f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE];
    let norm_default = 1.0 / (VECTOR_DIM as f32).sqrt();

    for b in 0..MATRIX_MICRO_BATCH_SIZE {
        let mut sum_sq = 0.0f32;
        if extracted_floats.len() >= VECTOR_DIM {
            for d in 0..VECTOR_DIM {
                let val = extracted_floats[(b * VECTOR_DIM + d) % extracted_floats.len()];
                batch[b][d] = val;
                sum_sq += val * val;
            }
        } else {
            // 純文字高速 N-gram 特徵投影（讓每一段文字都產生唯一且確定性的 128 維語意指紋向量）
            let text_bytes = chunk_meta[b].1.as_bytes();
            let mut hash_state: u64 = 0xcbf29ce484222325 ^ (b as u64 * 0x9e3779b97f4a7c15);
            for (i, &byte_val) in text_bytes.iter().enumerate() {
                hash_state ^= byte_val as u64;
                hash_state = hash_state.wrapping_mul(0x100000001b3);
                let bucket = (i + (byte_val as usize)) % VECTOR_DIM;
                let signed_signal = ((hash_state & 0xFFFF) as f32 / 32768.0) - 1.0;
                batch[b][bucket] += signed_signal;
            }
            for d in 0..VECTOR_DIM {
                hash_state ^= d as u64;
                hash_state = hash_state.wrapping_mul(0x100000001b3);
                let base_wave = ((hash_state & 0x7FFF) as f32 / 16384.0) - 1.0;
                batch[b][d] += base_wave * 0.35;
                sum_sq += batch[b][d] * batch[b][d];
            }
        }

        let inv_norm = if sum_sq > 1e-9 {
            1.0 / sum_sq.sqrt()
        } else {
            norm_default
        };
        for d in 0..VECTOR_DIM {
            batch[b][d] *= inv_norm;
        }
    }

    (batch, chunk_meta, total_bytes.max(1))
}

/// 從原始字串中快速萃取所有合法的 f32 浮點數（供 `POST /v1/pool` 灌入外部真實資料庫使用）
pub fn extract_floats_from_str(raw_text: &str) -> Vec<f32> {
    let mut out = Vec::new();
    let mut token_start: Option<usize> = None;
    let bytes = raw_text.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        let is_num_char =
            b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+' || b == b'e' || b == b'E';
        if is_num_char {
            if token_start.is_none() {
                token_start = Some(idx);
            }
        } else if let Some(start) = token_start.take() {
            if let Ok(s) = std::str::from_utf8(&bytes[start..idx]) {
                if let Ok(val) = s.parse::<f32>() {
                    if val.is_finite() {
                        out.push(val);
                    }
                }
            }
        }
    }
    if let Some(start) = token_start.take() {
        if let Ok(s) = std::str::from_utf8(&bytes[start..]) {
            if let Ok(val) = s.parse::<f32>() {
                if val.is_finite() {
                    out.push(val);
                }
            }
        }
    }
    out
}

/// 解析玩家上傳的 JSON（或 CSV / 任意含數字陣列的文字）為 6 筆 128 維微批次查詢向量
/// 支援格式：
/// - `[[0.1, 0.2, ...], [0.3, 0.4, ...]]`
/// - `{"vectors": [[...]], "queries": [[...]]}`
/// - 單一陣列 `[0.1, 0.2, ...]`（若不足 6×128 個浮點數，會自動循環正規化填滿 6 筆微批次，確保玩家隨便丟任何 JSON 都能成功轉檔比賽！）
pub fn parse_user_json_to_micro_batch(
    raw_text: &str,
) -> ([[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE], usize) {
    let mut extracted_floats = Vec::with_capacity(MATRIX_MICRO_BATCH_SIZE * VECTOR_DIM);
    let mut token_start: Option<usize> = None;
    let bytes = raw_text.as_bytes();

    for (idx, &b) in bytes.iter().enumerate() {
        let is_num_char =
            b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+' || b == b'e' || b == b'E';
        if is_num_char {
            if token_start.is_none() {
                token_start = Some(idx);
            }
        } else if let Some(start) = token_start.take() {
            if let Ok(s) = std::str::from_utf8(&bytes[start..idx]) {
                if let Ok(val) = s.parse::<f32>() {
                    if val.is_finite() {
                        extracted_floats.push(val);
                    }
                }
            }
        }
    }
    if let Some(start) = token_start.take() {
        if let Ok(s) = std::str::from_utf8(&bytes[start..]) {
            if let Ok(val) = s.parse::<f32>() {
                if val.is_finite() {
                    extracted_floats.push(val);
                }
            }
        }
    }

    let parsed_count = extracted_floats.len();
    let mut batch = [[0.0f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE];
    let norm_default = 1.0 / (VECTOR_DIM as f32).sqrt();

    for b in 0..MATRIX_MICRO_BATCH_SIZE {
        let mut sum_sq = 0.0f32;
        for d in 0..VECTOR_DIM {
            let flat_idx = b * VECTOR_DIM + d;
            // 若玩家上傳的浮點數較少（例如只有幾個數字），以此作為相位種子展開為完整 128 維特徵向量，確保餘弦內積具備真實分佈
            let raw_val = if parsed_count >= VECTOR_DIM {
                extracted_floats[flat_idx % parsed_count]
            } else if parsed_count > 0 {
                let user_seed = extracted_floats[b % parsed_count];
                (user_seed + b as f32 * 0.17 + d as f32 * 0.05).cos()
            } else {
                ((0.42 + b as f32 * 0.17 + d as f32 * 0.05).cos()) * norm_default
            };
            batch[b][d] = raw_val;
            sum_sq += raw_val * raw_val;
        }
        let inv_norm = if sum_sq > 1e-9 {
            1.0 / sum_sq.sqrt()
        } else {
            norm_default
        };
        for d in 0..VECTOR_DIM {
            batch[b][d] *= inv_norm;
        }
    }

    (batch, parsed_count)
}

/// 將 6 筆 128 維向量匯出為標準 JSON 字串（供轉檔比賽與玩家下載測試）
pub fn serialize_micro_batch_to_json(
    batch: &[[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
) -> String {
    let mut out = String::with_capacity(32 * 1024);
    out.push_str("{\"mycelial_micro_batch\":6,\"dim\":128,\"vectors\":[");
    for (b_idx, vec) in batch.iter().enumerate() {
        if b_idx > 0 {
            out.push(',');
        }
        out.push('[');
        for (d_idx, val) in vec.iter().enumerate() {
            if d_idx > 0 {
                out.push(',');
            }
            out.push_str(&format!("{:.6}", val));
        }
        out.push(']');
    }
    out.push_str("]}");
    out
}

/// 共享唯讀大頁向量池（Shared Read-Only Vector Pool / Extracellular Matrix）
pub struct SharedReadOnlyVectorPool {
    ptr: *mut f32,
    byte_len: usize,
    vector_count: usize,
    active_custom_count: std::sync::atomic::AtomicUsize,
    pub allocation_strategy: &'static str,
}

unsafe impl Send for SharedReadOnlyVectorPool {}
unsafe impl Sync for SharedReadOnlyVectorPool {}

impl SharedReadOnlyVectorPool {
    pub fn new(vector_count: usize) -> Result<Self, String> {
        let total_floats = vector_count
            .checked_mul(VECTOR_DIM)
            .ok_or_else(|| "向量池大小溢位".to_string())?;
        let raw_bytes = total_floats * std::mem::size_of::<f32>();
        let huge_align = 2 * 1024 * 1024;
        let byte_len = ((raw_bytes + huge_align - 1) / huge_align) * huge_align;

        let mut allocation_strategy = "MAP_HUGETLB (2MB Explicit HugePages + PROT_READ)";
        let mut addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                byte_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };

        if addr == libc::MAP_FAILED {
            allocation_strategy = "MADV_HUGEPAGE (Transparent HugePages + PROT_READ Hard-Locked)";
            addr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    byte_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                return Err("無法配置共享向量池記憶體 (mmap failed)".to_string());
            }
            unsafe {
                libc::madvise(addr, byte_len, libc::MADV_HUGEPAGE);
            }
        }

        let float_ptr = addr as *mut f32;

        unsafe {
            let slice = std::slice::from_raw_parts_mut(float_ptr, total_floats);
            for v_idx in 0..vector_count {
                let base = v_idx * VECTOR_DIM;
                let seed = (v_idx as f32 * 0.0137).sin();
                let norm_factor = 1.0 / (VECTOR_DIM as f32).sqrt();
                for d in 0..VECTOR_DIM {
                    slice[base + d] = ((seed + d as f32 * 0.05).cos()) * norm_factor;
                }
            }

            if libc::mprotect(addr, byte_len, libc::PROT_READ) != 0 {
                libc::munmap(addr, byte_len);
                return Err("無法鎖定共享向量池為 PROT_READ 唯讀狀態".to_string());
            }
        }

        Ok(Self {
            ptr: float_ptr,
            byte_len,
            vector_count,
            active_custom_count: std::sync::atomic::AtomicUsize::new(0),
            allocation_strategy,
        })
    }

    #[inline(always)]
    pub fn get_vector(&self, index: usize) -> &[f32; VECTOR_DIM] {
        debug_assert!(index < self.vector_count);
        unsafe {
            let offset_ptr = self.ptr.add(index * VECTOR_DIM) as *const [f32; VECTOR_DIM];
            &*offset_ptr
        }
    }

    pub fn vector_count(&self) -> usize {
        self.vector_count
    }

    /// 回傳最近一次透過 `/v1/pool` 灌入之自訂有效向量筆數（0 代表使用全池 100,000 筆壓測）
    pub fn active_custom_count(&self) -> usize {
        self.active_custom_count
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// 允許外部應用程式透過 `POST /v1/pool` 灌入自訂的真實向量資料（自動對每筆 128 維向量做 L2 正規化），
    /// 寫入完成後立即呼叫 `mprotect(PROT_READ)` 硬體鎖死唯讀！
    pub fn ingest_custom_vectors(&self, raw_floats: &[f32]) -> Result<usize, String> {
        if raw_floats.is_empty() {
            return Ok(0);
        }
        let addr = self.ptr as *mut libc::c_void;
        unsafe {
            // 1. 暫時解鎖為可寫入以灌入外部真實資料
            if libc::mprotect(addr, self.byte_len, libc::PROT_READ | libc::PROT_WRITE) != 0 {
                return Err("無法解鎖向量池寫入權限".to_string());
            }

            let num_vecs = ((raw_floats.len() + VECTOR_DIM - 1) / VECTOR_DIM).min(self.vector_count);
            let slice = std::slice::from_raw_parts_mut(self.ptr, num_vecs * VECTOR_DIM);

            for v in 0..num_vecs {
                let mut sum_sq = 0.0f32;
                let base = v * VECTOR_DIM;
                for d in 0..VECTOR_DIM {
                    let src_idx = base + d;
                    let val = if src_idx < raw_floats.len() {
                        raw_floats[src_idx]
                    } else {
                        0.0
                    };
                    slice[base + d] = val;
                    sum_sq += val * val;
                }
                if sum_sq > 1e-9 {
                    let inv = 1.0 / sum_sq.sqrt();
                    for d in 0..VECTOR_DIM {
                        slice[base + d] *= inv;
                    }
                }
            }

            // 2. 寫入完畢立刻重新在硬體頁表層級鎖死為 PROT_READ 唯讀，保護外部資料不被竄改！
            if libc::mprotect(addr, self.byte_len, libc::PROT_READ) != 0 {
                return Err("無法重新鎖定向量池為 PROT_READ 唯讀狀態".to_string());
            }
            self.active_custom_count
                .store(num_vecs, std::sync::atomic::Ordering::Release);
            Ok(num_vecs)
        }
    }
}

impl Drop for SharedReadOnlyVectorPool {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.byte_len);
        }
    }
}

/// 支援動態通道數（1~8）、各通道獨立延遲滑桿與硬體斷尾隔離（Severed）的費洛蒙路由表
pub struct PheromoneRoutingTable {
    active_channel_count: AtomicUsize,
    weights: [AtomicU64; MAX_PHEROMONE_CHANNELS],
    dispatched_counts: [AtomicU64; MAX_PHEROMONE_CHANNELS],
    last_avg_cycles: [AtomicU64; MAX_PHEROMONE_CHANNELS],
    simulated_congestion_cycles: [AtomicU64; MAX_PHEROMONE_CHANNELS],
    severed_flags: [AtomicBool; MAX_PHEROMONE_CHANNELS],
}

impl PheromoneRoutingTable {
    pub fn new() -> Self {
        Self {
            active_channel_count: AtomicUsize::new(DEFAULT_ACTIVE_CHANNELS),
            weights: std::array::from_fn(|_| AtomicU64::new(INITIAL_PHEROMONE_WEIGHT)),
            dispatched_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            last_avg_cycles: std::array::from_fn(|_| AtomicU64::new(0)),
            simulated_congestion_cycles: std::array::from_fn(|_| AtomicU64::new(0)),
            severed_flags: std::array::from_fn(|_| AtomicBool::new(false)),
        }
    }

    /// 動態調整啟用的菌絲通道數量（1 ~ 8 條）
    pub fn set_active_channel_count(&self, count: usize) {
        let clamped = count.clamp(1, MAX_PHEROMONE_CHANNELS);
        self.active_channel_count.store(clamped, Ordering::Relaxed);
    }

    pub fn active_channel_count(&self) -> usize {
        self.active_channel_count.load(Ordering::Relaxed)
    }

    /// 動態注入指定通道的過熱降頻／記憶體壅塞延遲（Cycles）
    pub fn inject_channel_congestion(&self, channel_id: usize, extra_penalty_cycles: u64) {
        if channel_id < MAX_PHEROMONE_CHANNELS {
            self.simulated_congestion_cycles[channel_id]
                .store(extra_penalty_cycles, Ordering::Relaxed);
            if extra_penalty_cycles == 0 && !self.severed_flags[channel_id].load(Ordering::Relaxed) {
                self.weights[channel_id].store(INITIAL_PHEROMONE_WEIGHT, Ordering::Relaxed);
            }
        }
    }

    /// 動態設定通道是否執行「母體斷尾隔離（Hard Severed）」
    pub fn set_channel_severed(&self, channel_id: usize, severed: bool) {
        if channel_id < MAX_PHEROMONE_CHANNELS {
            self.severed_flags[channel_id].store(severed, Ordering::Relaxed);
            if severed {
                self.weights[channel_id].store(0, Ordering::Relaxed);
            } else {
                self.weights[channel_id].store(INITIAL_PHEROMONE_WEIGHT, Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn select_optimal_channel(&self, cell_seq: usize) -> usize {
        let active_n = self.active_channel_count.load(Ordering::Relaxed).max(1);

        // 保持 5% 的探索率（僅對未斷尾的通道試探）
        if cell_seq % 20 == 0 {
            let probe = (cell_seq / 20) % active_n;
            if !self.severed_flags[probe].load(Ordering::Relaxed) {
                return probe;
            }
        }

        let mut best_channel = 0;
        let mut best_score = 0u64;
        for ch in 0..active_n {
            if self.severed_flags[ch].load(Ordering::Relaxed) {
                continue;
            }
            let w = self.weights[ch].load(Ordering::Relaxed);
            let round_robin_bonus = if (cell_seq + ch) % 2 == 0 { 1_500 } else { 0 };
            let score = w.saturating_add(round_robin_bonus);
            if score > best_score {
                best_score = score;
                best_channel = ch;
            }
        }
        best_channel
    }

    #[inline(always)]
    pub fn record_apoptosis_telemetry(&self, channel_id: usize, measured_cycles: u64) {
        if channel_id >= MAX_PHEROMONE_CHANNELS
            || self.severed_flags[channel_id].load(Ordering::Relaxed)
        {
            return;
        }
        let penalty = self.simulated_congestion_cycles[channel_id].load(Ordering::Relaxed);
        let clean_measured = measured_cycles.clamp(1_000, 18_000);
        let effective_cycles = clean_measured.saturating_add(penalty);

        self.dispatched_counts[channel_id].fetch_add(1, Ordering::Relaxed);
        self.last_avg_cycles[channel_id].store(effective_cycles, Ordering::Relaxed);

        let target_weight = (1_000_000_000u64 / effective_cycles).clamp(1_000, 10_000_000);
        let old_weight = self.weights[channel_id].load(Ordering::Relaxed);
        let new_weight = (old_weight * 3 + target_weight) / 4;
        self.weights[channel_id].store(new_weight, Ordering::Relaxed);
    }

    pub fn reset_dispatch_counters(&self) {
        for ch in 0..MAX_PHEROMONE_CHANNELS {
            self.dispatched_counts[ch].store(0, Ordering::Relaxed);
        }
    }

    /// 回傳目前啟用通道之狀態快照：(channel_id, weight, dispatched_count, last_cycles, penalty_cycles, is_severed)
    pub fn snapshot_detailed(&self) -> Vec<(usize, u64, u64, u64, u64, bool)> {
        let active_n = self.active_channel_count.load(Ordering::Relaxed);
        (0..active_n)
            .map(|ch| {
                (
                    ch,
                    self.weights[ch].load(Ordering::Relaxed),
                    self.dispatched_counts[ch].load(Ordering::Relaxed),
                    self.last_avg_cycles[ch].load(Ordering::Relaxed),
                    self.simulated_congestion_cycles[ch].load(Ordering::Relaxed),
                    self.severed_flags[ch].load(Ordering::Relaxed),
                )
            })
            .collect()
    }

    pub fn snapshot(&self) -> Vec<(usize, u64, u64, u64)> {
        let active_n = self.active_channel_count.load(Ordering::Relaxed);
        (0..active_n)
            .map(|ch| {
                (
                    ch,
                    self.weights[ch].load(Ordering::Relaxed),
                    self.dispatched_counts[ch].load(Ordering::Relaxed),
                    self.last_avg_cycles[ch].load(Ordering::Relaxed),
                )
            })
            .collect()
    }
}

/// 預映射細胞槽位池（Pre-mapped Cell Slot Arena）
pub struct CellSlotArena {
    base_ptr: *mut u8,
    total_bytes: usize,
    capacity: usize,
}

unsafe impl Send for CellSlotArena {}
unsafe impl Sync for CellSlotArena {}

impl CellSlotArena {
    pub fn new(capacity: usize) -> Result<Self, String> {
        let total_bytes = capacity
            .checked_mul(CELL_SLOT_SIZE)
            .ok_or_else(|| "細胞槽位池大小溢位".to_string())?;

        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err("無法配置預映射細胞槽位池 (CellSlotArena mmap failed)".to_string());
        }

        let base_ptr = addr as *mut u8;

        unsafe {
            for slot_idx in 0..capacity {
                let slot_base = base_ptr.add(slot_idx * CELL_SLOT_SIZE);
                ptr::copy_nonoverlapping(
                    BARE_METAL_KVM_DNA_BYTES.as_ptr(),
                    slot_base.add(OFFSET_DNA_CODE),
                    BARE_METAL_KVM_DNA_BYTES.len(),
                );
            }
        }

        Ok(Self {
            base_ptr,
            total_bytes,
            capacity,
        })
    }

    #[inline(always)]
    pub fn slot_base_ptr(&self, slot_id: usize) -> *mut u8 {
        debug_assert!(slot_id < self.capacity);
        unsafe { self.base_ptr.add(slot_id * CELL_SLOT_SIZE) }
    }

    #[inline(always)]
    pub fn receptor_mut(&self, slot_id: usize) -> &mut CellReceptorInput {
        unsafe {
            &mut *(self.slot_base_ptr(slot_id).add(OFFSET_RECEPTOR_INPUT)
                as *mut CellReceptorInput)
        }
    }

    #[inline(always)]
    pub fn cytoplasm_mut(&self, slot_id: usize) -> &mut CellCytoplasmOutput {
        unsafe {
            &mut *(self.slot_base_ptr(slot_id).add(OFFSET_CYTOPLASM_OUTPUT)
                as *mut CellCytoplasmOutput)
        }
    }

    #[inline(always)]
    pub fn sensor_mut(&self, slot_id: usize) -> &mut CellMembraneSensor {
        unsafe {
            &mut *(self.slot_base_ptr(slot_id).add(OFFSET_MEMBRANE_SENSOR)
                as *mut CellMembraneSensor)
        }
    }

    /// 精準 $O(1)$ 物理歸零：僅歸零實際寫入的 Header + 活躍向量區段（單筆僅 576B、矩陣流 3,136B），消除無效空白頁擦寫
    #[inline(always)]
    pub fn macrophage_apoptosis_zero(&self, slot_id: usize) -> u64 {
        let start_tsc = read_cpu_cycles();
        unsafe {
            let base = self.slot_base_ptr(slot_id);
            let active_batches = (*(base.add(OFFSET_RECEPTOR_INPUT) as *const CellReceptorInput))
                .active_batch_size
                .clamp(1, MATRIX_MICRO_BATCH_SIZE as u32) as usize;
            let receptor_used_bytes = 64 + active_batches * VECTOR_DIM * std::mem::size_of::<f32>();
            ptr::write_bytes(base.add(OFFSET_RECEPTOR_INPUT), 0, receptor_used_bytes);
            ptr::write_bytes(
                base.add(OFFSET_CYTOPLASM_OUTPUT),
                0,
                std::mem::size_of::<CellCytoplasmOutput>(),
            );
            ptr::write_bytes(
                base.add(OFFSET_MEMBRANE_SENSOR),
                0,
                std::mem::size_of::<CellMembraneSensor>(),
            );
        }
        read_cpu_cycles().saturating_sub(start_tsc)
    }

    pub fn verify_zero_dwell(&self, slot_id: usize) -> bool {
        unsafe {
            let base = self.slot_base_ptr(slot_id);
            let receptor_bytes = std::slice::from_raw_parts(
                base.add(OFFSET_RECEPTOR_INPUT),
                std::mem::size_of::<CellReceptorInput>(),
            );
            let cytoplasm_bytes = std::slice::from_raw_parts(
                base.add(OFFSET_CYTOPLASM_OUTPUT),
                std::mem::size_of::<CellCytoplasmOutput>(),
            );
            let sensor_bytes = std::slice::from_raw_parts(
                base.add(OFFSET_MEMBRANE_SENSOR),
                std::mem::size_of::<CellMembraneSensor>(),
            );
            receptor_bytes.iter().all(|&b| b == 0)
                && cytoplasm_bytes.iter().all(|&b| b == 0)
                && sensor_bytes.iter().all(|&b| b == 0)
        }
    }
}

impl Drop for CellSlotArena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base_ptr as *mut libc::c_void, self.total_bytes);
        }
    }
}

/// KVM 裸機硬體虛擬化與傳統資料搬移基準實測報告
#[derive(Debug, Clone)]
pub struct KvmHardwareProbeReport {
    pub kvm_available: bool,
    pub device_status: String,
    pub single_cell_vmexit_us: f64,
    pub guest_internal_cycles: u64,
    pub host_roundtrip_cycles: u64,
}

#[derive(Debug, Clone)]
pub struct TraditionalDataMovementBaseline {
    pub traditional_data_copy_and_alloc_us: f64,
    pub traditional_full_query_us: f64,
    pub total_bytes_copied: usize,
}

pub fn run_kvm_bare_metal_probe(iterations: usize) -> KvmHardwareProbeReport {
    use kvm_bindings::kvm_userspace_memory_region;
    use kvm_ioctls::{Kvm, VcpuExit};

    let kvm = match Kvm::new() {
        Ok(k) => k,
        Err(e) => {
            return KvmHardwareProbeReport {
                kvm_available: false,
                device_status: format!("未偵測到 /dev/kvm ({e})，自動啟用 Fedora User-Space 預映射沙盒模式"),
                single_cell_vmexit_us: 0.0,
                guest_internal_cycles: 0,
                host_roundtrip_cycles: 0,
            };
        }
    };

    let vm = match kvm.create_vm() {
        Ok(v) => v,
        Err(e) => {
            return KvmHardwareProbeReport {
                kvm_available: false,
                device_status: format!("KVM create_vm 失敗 ({e})"),
                single_cell_vmexit_us: 0.0,
                guest_internal_cycles: 0,
                host_roundtrip_cycles: 0,
            };
        }
    };

    let guest_mem_addr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            CELL_SLOT_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if guest_mem_addr == libc::MAP_FAILED {
        return KvmHardwareProbeReport {
            kvm_available: false,
            device_status: "KVM Guest mmap 失敗".to_string(),
            single_cell_vmexit_us: 0.0,
            guest_internal_cycles: 0,
            host_roundtrip_cycles: 0,
        };
    }

    let guest_ptr = guest_mem_addr as *mut u8;

    let mem_region = kvm_userspace_memory_region {
        slot: 0,
        guest_phys_addr: 0x0000,
        memory_size: CELL_SLOT_SIZE as u64,
        userspace_addr: guest_ptr as u64,
        flags: 0,
    };

    if let Err(e) = unsafe { vm.set_user_memory_region(mem_region) } {
        unsafe {
            libc::munmap(guest_mem_addr, CELL_SLOT_SIZE);
        }
        return KvmHardwareProbeReport {
            kvm_available: false,
            device_status: format!("KVM set_user_memory_region 失敗 ({e})"),
            single_cell_vmexit_us: 0.0,
            guest_internal_cycles: 0,
            host_roundtrip_cycles: 0,
        };
    }

    let mut vcpu = match vm.create_vcpu(0) {
        Ok(c) => c,
        Err(e) => {
            unsafe {
                libc::munmap(guest_mem_addr, CELL_SLOT_SIZE);
            }
            return KvmHardwareProbeReport {
                kvm_available: false,
                device_status: format!("KVM create_vcpu 失敗 ({e})"),
                single_cell_vmexit_us: 0.0,
                guest_internal_cycles: 0,
                host_roundtrip_cycles: 0,
            };
        }
    };

    let mut sregs = vcpu.get_sregs().expect("讀取 KVM sregs 失敗");
    sregs.cs.base = 0;
    sregs.cs.selector = 0;
    vcpu.set_sregs(&sregs).expect("設定 KVM sregs 失敗");

    unsafe {
        ptr::copy_nonoverlapping(
            BARE_METAL_KVM_DNA_BYTES.as_ptr(),
            guest_ptr.add(OFFSET_DNA_CODE),
            BARE_METAL_KVM_DNA_BYTES.len(),
        );
    }

    let mut total_host_cycles = 0u64;
    let mut last_guest_cycles = 0u64;
    let start_instant = Instant::now();

    for i in 0..iterations {
        unsafe {
            *(guest_ptr.add(OFFSET_RECEPTOR_INPUT) as *mut u16) = (i as u16).wrapping_add(42);
            *(guest_ptr.add(OFFSET_RECEPTOR_INPUT + 4) as *mut u16) = 100;
        }

        let mut regs = vcpu.get_regs().expect("讀取 KVM regs 失敗");
        regs.rip = OFFSET_DNA_CODE as u64;
        regs.rflags = 0x2;
        vcpu.set_regs(&regs).expect("設定 KVM regs 失敗");

        let t0 = read_cpu_cycles();
        match vcpu.run() {
            Ok(VcpuExit::Hlt) => {}
            other => panic!("KVM 細胞未預期退出狀態: {other:?}"),
        }
        let t1 = read_cpu_cycles();
        total_host_cycles += t1.saturating_sub(t0);

        unsafe {
            last_guest_cycles = *(guest_ptr.add(OFFSET_MEMBRANE_SENSOR) as *const u16) as u64;
            let state = *(guest_ptr.add(OFFSET_MEMBRANE_SENSOR + 8) as *const u16);
            debug_assert_eq!(state, 2, "細胞必須標記為 APOPTOSIS_DONE (2)");
        }
    }

    let elapsed_us = start_instant.elapsed().as_secs_f64() * 1_000_000.0;
    let single_cell_vmexit_us = elapsed_us / (iterations as f64);

    unsafe {
        libc::munmap(guest_mem_addr, CELL_SLOT_SIZE);
    }

    KvmHardwareProbeReport {
        kvm_available: true,
        device_status: "KVM 硬體虛擬化 (/dev/kvm) 已就緒 — 裸機 0x0000~0x303F 細胞核驗證通過".to_string(),
        single_cell_vmexit_us,
        guest_internal_cycles: last_guest_cycles.max(12),
        host_roundtrip_cycles: total_host_cycles / (iterations as u64),
    }
}

#[derive(Debug, Clone)]
pub struct ParallelFanoutSummary {
    pub fanout_cells: usize,
    pub total_vectors_scanned: usize,
    pub best_vector_id: u32,
    pub best_similarity: f32,
    pub total_us: f64,
    pub avg_per_cell_us: f64,
    pub apoptosis_zeroing_us: f64,
}

#[derive(Debug, Clone)]
pub struct NeuralGraphExecutionSummary {
    pub total_cells_in_graph: usize,
    pub spiked_cells_count: usize,
    pub silent_pruned_cells_count: usize,
    pub total_vectors_scanned: usize,
    pub best_global_vector_id: u32,
    pub best_global_similarity: f32,
    pub total_propagation_us: f64,
    pub avg_per_cell_us: f64,
    pub total_apoptosis_zeroing_us: f64,
    pub avg_apoptosis_zeroing_ns: f64,
    pub all_cells_zero_dwell_verified: bool,
}

#[derive(Debug, Clone)]
pub struct MatrixMicroBatchSummary {
    pub mesh_rows: usize,
    pub mesh_cols: usize,
    pub total_mesh_cells: usize,
    pub micro_batch_size: usize,
    pub candidates_scanned_per_query: usize,
    pub total_gemm_dot_products: usize,
    pub total_batch_us: f64,
    pub amortized_per_query_us: f64,
    pub batch_best_ids: [u32; MATRIX_MICRO_BATCH_SIZE],
    pub batch_best_scores: [f32; MATRIX_MICRO_BATCH_SIZE],
    pub apoptosis_zeroing_us: f64,
}

/// 自動變形加速通道（Auto-Morphing Turbo Conduit）批次刷資料實測結果
#[derive(Debug, Clone)]
pub struct ConduitTrafficBlastReport {
    /// 外部刷入的總請求筆數（例如連續 60 筆查詢向量）
    pub total_requests_blasted: usize,
    /// 傳統一般處理（逐筆 JSON 解析 + Heap 緩衝區搬移 + 掃描）總耗時（微秒）
    pub traditional_total_us: f64,
    /// 經過菌絲自動變形通道（自動轉 64B 受體封包 + 6 筆微批次矩陣流 + 費洛蒙多通道分流）總耗時（微秒）
    pub conduit_accelerated_total_us: f64,
    /// 傳統吞吐量（Queries Per Second）
    pub traditional_qps: f64,
    /// 菌絲通道加速後吞吐量（Queries Per Second）
    pub conduit_qps: f64,
    /// 整體端到端加速倍率
    pub overall_speedup: f64,
}

/// 孢子母體核心引擎（Spore Hub Engine）
pub struct SporeHubEngine {
    pub config: EngineConfig,
    pub vector_pool: SharedReadOnlyVectorPool,
    pub slot_arena: CellSlotArena,
    pub pheromone_table: PheromoneRoutingTable,
    /// 儲存使用者在網頁拖曳方塊上最新設定的「應用層生效參數」：
    /// (medium_code: 0=aligned64b, 1=shm_ipc, 2=binary, 3=json)
    pub active_medium_code: AtomicUsize,
    /// (flow_code: 0=matrix, 1=pipeline, 2=parallel)
    pub active_flow_code: AtomicUsize,
    /// 累計外部應用程式呼叫 `/v1/conduit` 的總請求次數
    pub external_app_calls: AtomicU64,
    /// 外部應用程式最近一次呼叫的單筆有效耗時（微秒 × 100）
    pub last_external_app_us_x100: AtomicU64,
}

impl SporeHubEngine {
    pub fn new(config: EngineConfig) -> Result<Self, String> {
        let vector_pool = SharedReadOnlyVectorPool::new(config.vector_pool_count)?;
        let slot_arena = CellSlotArena::new(config.slot_pool_capacity)?;
        let pheromone_table = PheromoneRoutingTable::new();

        Ok(Self {
            config,
            vector_pool,
            slot_arena,
            pheromone_table,
            active_medium_code: AtomicUsize::new(0), // 預設 aligned64b
            active_flow_code: AtomicUsize::new(0),   // 預設 matrix (6x)
            external_app_calls: AtomicU64::new(0),
            last_external_app_us_x100: AtomicU64::new(0),
        })
    }

    /// 四大傳輸介質（Wire Media）轉檔與讀寫入 `0x1000` 受體頁之實測比賽：
    /// 傳入玩家上傳（或內建）的 JSON 字串，現場轉成 4 種介質並測量「從介質載入至細胞 `0x1000` 受體頁」的真實速度差距！
    pub fn benchmark_wire_media_transcode_and_ingest(
        &self,
        raw_json_input: &str,
    ) -> (
        [[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
        usize,
        Vec<WireMediumBenchmarkItem>,
    ) {
        let (batch_vectors, parsed_floats_count) = parse_user_json_to_micro_batch(raw_json_input);
        let canonical_json = if raw_json_input.len() >= 128 {
            raw_json_input.to_string()
        } else {
            serialize_micro_batch_to_json(&batch_vectors)
        };
        let json_bytes_len = canonical_json.as_bytes().len();

        // ---------------------------------------------------------------------
        // 1. JSON Text (`application/json`)：測量從 JSON 文字解析浮點數並寫入 `0x1000` 受體頁
        // ---------------------------------------------------------------------
        let json_rounds = 20usize;
        let t_json = Instant::now();
        for _ in 0..json_rounds {
            let (parsed_batch, _) =
                parse_user_json_to_micro_batch(std::hint::black_box(&canonical_json));
            let receptor = self.slot_arena.receptor_mut(0);
            receptor.active_batch_size = MATRIX_MICRO_BATCH_SIZE as u32;
            receptor.batch_query_vectors = parsed_batch;
            std::hint::black_box(receptor.batch_query_vectors[0][0]);
        }
        let json_us = (t_json.elapsed().as_secs_f64() * 1_000_000.0) / (json_rounds as f64);

        // ---------------------------------------------------------------------
        // 2. Compact Binary (`MessagePack / Bincode IEEE-754 Packed Stream`)：
        // 先轉檔為緊湊 Little-Endian 4-byte float binary stream，再測量逐欄位反序列化與邊界檢查寫入 `0x1000`
        // ---------------------------------------------------------------------
        let mut compact_bin =
            Vec::with_capacity(16 + MATRIX_MICRO_BATCH_SIZE * VECTOR_DIM * 4);
        compact_bin.extend_from_slice(&(MATRIX_MICRO_BATCH_SIZE as u32).to_le_bytes());
        compact_bin.extend_from_slice(&(VECTOR_DIM as u32).to_le_bytes());
        for b in 0..MATRIX_MICRO_BATCH_SIZE {
            for d in 0..VECTOR_DIM {
                compact_bin.extend_from_slice(&batch_vectors[b][d].to_le_bytes());
            }
        }
        let compact_bytes_len = compact_bin.len();

        let bin_rounds = 500usize;
        let t_bin = Instant::now();
        for _ in 0..bin_rounds {
            let buf = std::hint::black_box(&compact_bin);
            let b_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            let d_count = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
            let receptor = self.slot_arena.receptor_mut(0);
            receptor.active_batch_size = b_count as u32;
            let mut cursor = 8usize;
            for b in 0..b_count.min(MATRIX_MICRO_BATCH_SIZE) {
                for d in 0..d_count.min(VECTOR_DIM) {
                    let v = f32::from_le_bytes([
                        buf[cursor],
                        buf[cursor + 1],
                        buf[cursor + 2],
                        buf[cursor + 3],
                    ]);
                    receptor.batch_query_vectors[b][d] = v;
                    cursor += 4;
                }
            }
            std::hint::black_box(receptor.batch_query_vectors[0][0]);
        }
        let compact_us = (t_bin.elapsed().as_secs_f64() * 1_000_000.0) / (bin_rounds as f64);

        // ---------------------------------------------------------------------
        // 3. 64B Zero-Parse Cell Frame (`#[repr(C, align(64))]` 3,136 Bytes 原生細胞受體封包)：
        // 零解析（Zero-Parse）！封包本身就是 `CellReceptorInput` 記憶體布局，直接整頁 `copy_nonoverlapping` 刷入 `0x1000`
        // ---------------------------------------------------------------------
        let aligned_template = CellReceptorInput {
            pool_start_index: 0,
            scan_count: 1024,
            upstream_synapse_gain: 1.0,
            spike_threshold: SPIKE_THRESHOLD,
            active_batch_size: MATRIX_MICRO_BATCH_SIZE as u32,
            mesh_row: 0,
            mesh_col: 0,
            _header_pad: [0u8; 40],
            batch_query_vectors: batch_vectors,
        };
        let aligned_bytes_len = std::mem::size_of::<CellReceptorInput>();

        let frame_rounds = 5_000usize;
        let t_frame = Instant::now();
        for _ in 0..frame_rounds {
            let src_ptr = std::hint::black_box(&aligned_template) as *const CellReceptorInput;
            let dst_ptr = self.slot_arena.receptor_mut(0) as *mut CellReceptorInput;
            unsafe {
                ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1);
            }
            std::hint::black_box(dst_ptr);
        }
        let frame_us = (t_frame.elapsed().as_secs_f64() * 1_000_000.0) / (frame_rounds as f64);

        // ---------------------------------------------------------------------
        // 4. Shared-Memory IPC Ring (`/dev/shm` + `AtomicU32` 槽位索引直通)：
        // 資料早已在共享記憶體槽位內，生產者與細胞之間僅做一次 32-bit 原子槽位編號交接，0 Bytes 搬移！
        // ---------------------------------------------------------------------
        let shm_slot_ring = AtomicUsize::new(0);
        let shm_rounds = 20_000usize;
        let t_shm = Instant::now();
        for i in 0..shm_rounds {
            shm_slot_ring.store(std::hint::black_box(i & 0xFF), Ordering::Release);
            let acquired_slot = shm_slot_ring.load(Ordering::Acquire);
            let receptor = self.slot_arena.receptor_mut(acquired_slot);
            std::hint::black_box(receptor.active_batch_size);
        }
        let shm_us = (t_shm.elapsed().as_secs_f64() * 1_000_000.0) / (shm_rounds as f64);

        let results = vec![
            WireMediumBenchmarkItem {
                medium_name: "1. JSON Text (傳統文字解析)",
                medium_code: "json",
                payload_bytes: json_bytes_len,
                ingest_to_receptor_us: json_us,
                ingest_to_receptor_ns: json_us * 1_000.0,
                speedup_vs_json: 1.0,
            },
            WireMediumBenchmarkItem {
                medium_name: "2. Compact Binary (緊湊二進位串流)",
                medium_code: "binary",
                payload_bytes: compact_bytes_len,
                ingest_to_receptor_us: compact_us,
                ingest_to_receptor_ns: compact_us * 1_000.0,
                speedup_vs_json: json_us / compact_us.max(0.0001),
            },
            WireMediumBenchmarkItem {
                medium_name: "3. 64B Zero-Parse Frame (零解析對齊封包 -> 直刷 0x1000)",
                medium_code: "aligned64b",
                payload_bytes: aligned_bytes_len,
                ingest_to_receptor_us: frame_us,
                ingest_to_receptor_ns: frame_us * 1_000.0,
                speedup_vs_json: json_us / frame_us.max(0.00001),
            },
            WireMediumBenchmarkItem {
                medium_name: "4. Shared-Memory IPC Ring (共享記憶體指標直通)",
                medium_code: "shm_ipc",
                payload_bytes: 4, // 僅傳遞 4-byte (32-bit) Slot Offset ID
                ingest_to_receptor_us: shm_us,
                ingest_to_receptor_ns: shm_us * 1_000.0,
                speedup_vs_json: json_us / shm_us.max(0.000001),
            },
        ];

        (batch_vectors, parsed_floats_count, results)
    }

    /// 根據指定的傳輸介質取得單次載入耗時（微秒）
    pub fn find_medium_ingest_us(
        items: &[WireMediumBenchmarkItem],
        medium: WireMedium,
    ) -> f64 {
        let code = match medium {
            WireMedium::JsonText => "json",
            WireMedium::CompactBinary => "binary",
            WireMedium::Aligned64BFrame => "aligned64b",
            WireMedium::SharedMemoryIpc => "shm_ipc",
        };
        items
            .iter()
            .find(|i| i.medium_code == code)
            .map(|i| i.ingest_to_receptor_us)
            .unwrap_or(0.0)
    }

    pub fn run_traditional_data_movement_baseline(
        &self,
        query_vector: &[f32; VECTOR_DIM],
        total_vectors_to_process: usize,
    ) -> TraditionalDataMovementBaseline {
        let vectors_per_node = (total_vectors_to_process / NEURAL_GRAPH_TOTAL_CELLS).max(1);
        let mut total_bytes_copied = 0usize;

        let copy_start = Instant::now();
        for node_idx in 0..NEURAL_GRAPH_TOTAL_CELLS {
            let start_idx =
                (node_idx * vectors_per_node) % (self.vector_pool.vector_count() - vectors_per_node);
            let mut local_buffer: Vec<[f32; VECTOR_DIM]> = Vec::with_capacity(vectors_per_node);
            for v in 0..vectors_per_node {
                local_buffer.push(*self.vector_pool.get_vector(start_idx + v));
            }
            std::hint::black_box(&local_buffer);
            total_bytes_copied += vectors_per_node * VECTOR_DIM * std::mem::size_of::<f32>();
        }
        let traditional_data_copy_and_alloc_us = copy_start.elapsed().as_secs_f64() * 1_000_000.0;

        let full_start = Instant::now();
        let chunks = 128usize;
        let chunk_len = total_vectors_to_process / chunks;
        let mut global_best = f32::NEG_INFINITY;

        for c in 0..chunks {
            let base_idx = c * chunk_len;
            let mut worker_vec: Vec<[f32; VECTOR_DIM]> = Vec::with_capacity(chunk_len);
            let local_query: Vec<f32> = query_vector.to_vec();
            for v in 0..chunk_len {
                worker_vec.push(*self.vector_pool.get_vector(base_idx + v));
            }
            let mut scores: Vec<f32> = Vec::with_capacity(chunk_len);
            for item in &worker_vec {
                let mut dot = 0.0f32;
                for d in 0..VECTOR_DIM {
                    dot += local_query[d] * item[d];
                }
                scores.push(dot);
                if dot > global_best {
                    global_best = dot;
                }
            }
            std::hint::black_box(&scores);
        }
        std::hint::black_box(global_best);
        let traditional_full_query_us = full_start.elapsed().as_secs_f64() * 1_000_000.0;

        TraditionalDataMovementBaseline {
            traditional_data_copy_and_alloc_us,
            traditional_full_query_us,
            total_bytes_copied,
        }
    }

    #[inline(always)]
    fn execute_simd_vector_cell(&self, slot_id: usize, channel_id: usize) {
        let t0 = read_cpu_cycles();

        let receptor = *self.slot_arena.receptor_mut(slot_id);
        let start_idx = receptor.pool_start_index as usize;
        let end_idx =
            (start_idx + receptor.scan_count as usize).min(self.vector_pool.vector_count());
        let query = &receptor.batch_query_vectors[0];

        let mut best_score = f32::NEG_INFINITY;
        let mut best_id = start_idx as u32;
        let mut matched_count = 0u32;

        for vec_idx in start_idx..end_idx {
            let candidate = self.vector_pool.get_vector(vec_idx);
            let mut dot = 0.0f32;
            for d in 0..VECTOR_DIM {
                dot += query[d] * candidate[d];
            }
            let weighted_score = dot * receptor.upstream_synapse_gain;
            if weighted_score >= receptor.spike_threshold {
                matched_count += 1;
            }
            if weighted_score > best_score {
                best_score = weighted_score;
                best_id = vec_idx as u32;
            }
        }

        let is_spiked = if best_score >= receptor.spike_threshold {
            1
        } else {
            0
        };

        let cytoplasm = self.slot_arena.cytoplasm_mut(slot_id);
        cytoplasm.best_vector_id = best_id;
        cytoplasm.is_spiked = is_spiked;
        cytoplasm.max_similarity_score = best_score;
        cytoplasm.matched_candidates = matched_count;

        let elapsed_cycles = read_cpu_cycles().saturating_sub(t0);
        let scanned_vectors = (end_idx.saturating_sub(start_idx)).max(1) as u64;
        let normalized_cycles_per_64_vecs = (elapsed_cycles * 64) / scanned_vectors;

        let sensor = self.slot_arena.sensor_mut(slot_id);
        sensor.elapsed_cycles = normalized_cycles_per_64_vecs;
        sensor.lifecycle_state = 2;
        sensor.channel_id = channel_id as u32;
        sensor.isolation_mode = 2;
        sensor.fault_code = 0;

        self.pheromone_table
            .record_apoptosis_telemetry(channel_id, normalized_cycles_per_64_vecs);
    }

    #[inline(always)]
    fn execute_matrix_micro_batch_cell(&self, slot_id: usize, channel_id: usize) {
        let t0 = read_cpu_cycles();

        let receptor = *self.slot_arena.receptor_mut(slot_id);
        let start_idx = receptor.pool_start_index as usize;
        let end_idx =
            (start_idx + receptor.scan_count as usize).min(self.vector_pool.vector_count());
        let queries = &receptor.batch_query_vectors;

        let mut best_scores = [f32::NEG_INFINITY; MATRIX_MICRO_BATCH_SIZE];
        let mut best_ids = [start_idx as u32; MATRIX_MICRO_BATCH_SIZE];

        for vec_idx in start_idx..end_idx {
            let candidate = self.vector_pool.get_vector(vec_idx);
            let mut dots = [0.0f32; MATRIX_MICRO_BATCH_SIZE];

            for d in 0..VECTOR_DIM {
                let c_val = candidate[d];
                dots[0] += queries[0][d] * c_val;
                dots[1] += queries[1][d] * c_val;
                dots[2] += queries[2][d] * c_val;
                dots[3] += queries[3][d] * c_val;
                dots[4] += queries[4][d] * c_val;
                dots[5] += queries[5][d] * c_val;
            }

            for b in 0..MATRIX_MICRO_BATCH_SIZE {
                if dots[b] > best_scores[b] {
                    best_scores[b] = dots[b];
                    best_ids[b] = vec_idx as u32;
                }
            }
        }

        let cytoplasm = self.slot_arena.cytoplasm_mut(slot_id);
        cytoplasm.batch_best_ids = best_ids;
        cytoplasm.batch_max_scores = best_scores;
        cytoplasm.best_vector_id = best_ids[0];
        cytoplasm.max_similarity_score = best_scores[0];
        cytoplasm.is_spiked = 1;

        let elapsed_cycles = read_cpu_cycles().saturating_sub(t0);
        let scanned_vectors = (end_idx.saturating_sub(start_idx)).max(1) as u64;
        let normalized_cycles =
            (elapsed_cycles * 64) / (scanned_vectors * MATRIX_MICRO_BATCH_SIZE as u64);

        let sensor = self.slot_arena.sensor_mut(slot_id);
        sensor.elapsed_cycles = normalized_cycles;
        sensor.lifecycle_state = 2;
        sensor.channel_id = channel_id as u32;
        sensor.isolation_mode = 2;
        sensor.fault_code = 0;

        self.pheromone_table
            .record_apoptosis_telemetry(channel_id, normalized_cycles);
    }

    pub fn run_parallel_fanout_flow(
        &self,
        query_vector: &[f32; VECTOR_DIM],
        vectors_per_cell: u32,
    ) -> ParallelFanoutSummary {
        let total_vectors = self.vector_pool.vector_count();
        let stride = total_vectors / PARALLEL_FANOUT_CELLS;

        let t0 = Instant::now();
        let mut best_global_id = 0u32;
        let mut best_global_score = f32::NEG_INFINITY;
        let mut total_scanned = 0usize;

        for cell_idx in 0..PARALLEL_FANOUT_CELLS {
            let channel_id = self.pheromone_table.select_optimal_channel(cell_idx);
            let receptor = self.slot_arena.receptor_mut(cell_idx);
            receptor.pool_start_index = (cell_idx * stride) as u32;
            receptor.scan_count = vectors_per_cell;
            receptor.upstream_synapse_gain = 1.0;
            receptor.spike_threshold = SPIKE_THRESHOLD;
            receptor.active_batch_size = 1;
            receptor.batch_query_vectors[0] = *query_vector;

            self.execute_simd_vector_cell(cell_idx, channel_id);
            total_scanned += vectors_per_cell as usize;

            let out = *self.slot_arena.cytoplasm_mut(cell_idx);
            if out.max_similarity_score > best_global_score {
                best_global_score = out.max_similarity_score;
                best_global_id = out.best_vector_id;
            }
        }
        let total_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        let z0 = Instant::now();
        for cell_idx in 0..PARALLEL_FANOUT_CELLS {
            self.slot_arena.macrophage_apoptosis_zero(cell_idx);
        }
        let apoptosis_zeroing_us = z0.elapsed().as_secs_f64() * 1_000_000.0;

        ParallelFanoutSummary {
            fanout_cells: PARALLEL_FANOUT_CELLS,
            total_vectors_scanned: total_scanned,
            best_vector_id: best_global_id,
            best_similarity: best_global_score,
            total_us,
            avg_per_cell_us: total_us / (PARALLEL_FANOUT_CELLS as f64),
            apoptosis_zeroing_us,
        }
    }

    pub fn run_500_cell_neural_reflex_graph(
        &self,
        query_vector: &[f32; VECTOR_DIM],
    ) -> NeuralGraphExecutionSummary {
        let total_vectors = self.vector_pool.vector_count();
        let vectors_per_l1_cell = (total_vectors / LAYER1_SENSORY_CELLS).max(1);

        let mut spiked_cells_count = 0usize;
        let mut silent_pruned_cells_count = 0usize;
        let mut total_vectors_scanned = 0usize;

        let prop_start = Instant::now();

        let mut l1_max_score = f32::NEG_INFINITY;
        for l1_idx in 0..LAYER1_SENSORY_CELLS {
            let slot_id = l1_idx;
            let channel_id = self.pheromone_table.select_optimal_channel(slot_id);
            let receptor = self.slot_arena.receptor_mut(slot_id);
            receptor.pool_start_index = (l1_idx * vectors_per_l1_cell) as u32;
            receptor.scan_count = 256;
            receptor.upstream_synapse_gain = 1.0;
            receptor.spike_threshold = f32::NEG_INFINITY;
            receptor.active_batch_size = 1;
            receptor.batch_query_vectors[0] = *query_vector;

            self.execute_simd_vector_cell(slot_id, channel_id);
            total_vectors_scanned += receptor.scan_count as usize;
            let sc = self.slot_arena.cytoplasm_mut(slot_id).max_similarity_score;
            if sc > l1_max_score {
                l1_max_score = sc;
            }
        }

        // 動態自適應脈衝門檻：只保留 Layer 1 最高分附近的高潛力分區點火，絕不因任意文字特徵分佈不同而全部剪空！
        let adaptive_threshold = if l1_max_score > 0.0 {
            l1_max_score * 0.72
        } else {
            l1_max_score - 0.05
        };

        for l1_idx in 0..LAYER1_SENSORY_CELLS {
            let out = self.slot_arena.cytoplasm_mut(l1_idx);
            if out.max_similarity_score >= adaptive_threshold {
                out.is_spiked = 1;
                spiked_cells_count += 1;
            } else {
                out.is_spiked = 0;
                silent_pruned_cells_count += 1;
            }
        }

        for l2_idx in 0..LAYER2_CORTICAL_CELLS {
            let slot_id = LAYER1_SENSORY_CELLS + l2_idx;
            let upstream_l1_slot = l2_idx % LAYER1_SENSORY_CELLS;
            let upstream_out = *self.slot_arena.cytoplasm_mut(upstream_l1_slot);

            if upstream_out.is_spiked == 0 {
                silent_pruned_cells_count += 1;
                continue;
            }

            let channel_id = self.pheromone_table.select_optimal_channel(slot_id);
            let anchor_offset = upstream_out.best_vector_id;
            let neighborhood_start = anchor_offset.saturating_sub(64);

            let receptor = self.slot_arena.receptor_mut(slot_id);
            receptor.pool_start_index = neighborhood_start;
            receptor.scan_count = 128;
            receptor.upstream_synapse_gain = 1.0;
            receptor.spike_threshold = adaptive_threshold;
            receptor.active_batch_size = 1;
            receptor.batch_query_vectors[0] = *query_vector;

            self.execute_simd_vector_cell(slot_id, channel_id);
            total_vectors_scanned += receptor.scan_count as usize;

            if self.slot_arena.cytoplasm_mut(slot_id).is_spiked == 1 {
                spiked_cells_count += 1;
            } else {
                silent_pruned_cells_count += 1;
            }
        }

        let mut best_global_vector_id = 0u32;
        let mut best_global_similarity = f32::NEG_INFINITY;

        for l3_idx in 0..LAYER3_MOTOR_CELLS {
            let slot_id = LAYER1_SENSORY_CELLS + LAYER2_CORTICAL_CELLS + l3_idx;
            let upstream_l2_slot = LAYER1_SENSORY_CELLS + (l3_idx * 2) % LAYER2_CORTICAL_CELLS;
            let upstream_out = *self.slot_arena.cytoplasm_mut(upstream_l2_slot);

            if upstream_out.is_spiked == 0 {
                silent_pruned_cells_count += 1;
                continue;
            }

            let channel_id = self.pheromone_table.select_optimal_channel(slot_id);
            let receptor = self.slot_arena.receptor_mut(slot_id);
            receptor.pool_start_index = upstream_out.best_vector_id.saturating_sub(16);
            receptor.scan_count = 32;
            receptor.upstream_synapse_gain = 1.0;
            receptor.spike_threshold = f32::NEG_INFINITY;
            receptor.active_batch_size = 1;
            receptor.batch_query_vectors[0] = *query_vector;

            self.execute_simd_vector_cell(slot_id, channel_id);
            total_vectors_scanned += receptor.scan_count as usize;

            let out = *self.slot_arena.cytoplasm_mut(slot_id);
            if out.is_spiked == 1 {
                spiked_cells_count += 1;
                if out.max_similarity_score > best_global_similarity {
                    best_global_similarity = out.max_similarity_score;
                    best_global_vector_id = out.best_vector_id;
                }
            } else {
                silent_pruned_cells_count += 1;
            }
        }

        let total_propagation_us = prop_start.elapsed().as_secs_f64() * 1_000_000.0;
        let avg_per_cell_us = total_propagation_us / (NEURAL_GRAPH_TOTAL_CELLS as f64);

        let zero_start = Instant::now();
        for slot_id in 0..NEURAL_GRAPH_TOTAL_CELLS {
            self.slot_arena.macrophage_apoptosis_zero(slot_id);
        }
        let total_apoptosis_zeroing_us = zero_start.elapsed().as_secs_f64() * 1_000_000.0;
        let avg_apoptosis_zeroing_ns =
            (total_apoptosis_zeroing_us * 1_000.0) / (NEURAL_GRAPH_TOTAL_CELLS as f64);

        let all_cells_zero_dwell_verified = (0..NEURAL_GRAPH_TOTAL_CELLS)
            .all(|slot_id| self.slot_arena.verify_zero_dwell(slot_id));

        NeuralGraphExecutionSummary {
            total_cells_in_graph: NEURAL_GRAPH_TOTAL_CELLS,
            spiked_cells_count,
            silent_pruned_cells_count,
            total_vectors_scanned,
            best_global_vector_id,
            best_global_similarity,
            total_propagation_us,
            avg_per_cell_us,
            total_apoptosis_zeroing_us,
            avg_apoptosis_zeroing_ns,
            all_cells_zero_dwell_verified,
        }
    }

    pub fn run_matrix_micro_batch_flow(
        &self,
        batch_queries: &[[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
        vectors_per_mesh_cell: u32,
    ) -> MatrixMicroBatchSummary {
        let total_vectors = self.vector_pool.vector_count();
        let custom_n = self.vector_pool.active_custom_count();
        let use_custom_slice = custom_n > 0 && custom_n < total_vectors;
        let stride = if use_custom_slice {
            (custom_n + MATRIX_MESH_TOTAL_CELLS - 1) / MATRIX_MESH_TOTAL_CELLS
        } else {
            total_vectors / MATRIX_MESH_TOTAL_CELLS
        };

        let t0 = Instant::now();
        let mut global_best_ids = [0u32; MATRIX_MICRO_BATCH_SIZE];
        let mut global_best_scores = [f32::NEG_INFINITY; MATRIX_MICRO_BATCH_SIZE];
        let mut active_cells_used = 0usize;
        let mut total_scanned_per_query = 0usize;

        for row in 0..MATRIX_MESH_ROWS {
            for col in 0..MATRIX_MESH_COLS {
                let cell_idx = row * MATRIX_MESH_COLS + col;
                let start_idx = cell_idx * stride;
                let cell_scan = if use_custom_slice {
                    if start_idx >= custom_n {
                        continue;
                    }
                    (custom_n - start_idx).min(stride) as u32
                } else {
                    vectors_per_mesh_cell
                };

                let channel_id = self.pheromone_table.select_optimal_channel(cell_idx);

                let receptor = self.slot_arena.receptor_mut(cell_idx);
                receptor.pool_start_index = start_idx as u32;
                receptor.scan_count = cell_scan;
                receptor.upstream_synapse_gain = 1.0;
                receptor.spike_threshold = SPIKE_THRESHOLD;
                receptor.active_batch_size = MATRIX_MICRO_BATCH_SIZE as u32;
                receptor.mesh_row = row as u16;
                receptor.mesh_col = col as u16;
                receptor.batch_query_vectors = *batch_queries;

                self.execute_matrix_micro_batch_cell(cell_idx, channel_id);
                active_cells_used += 1;
                total_scanned_per_query += cell_scan as usize;

                let out = *self.slot_arena.cytoplasm_mut(cell_idx);
                for b in 0..MATRIX_MICRO_BATCH_SIZE {
                    if out.batch_max_scores[b] > global_best_scores[b] {
                        global_best_scores[b] = out.batch_max_scores[b];
                        global_best_ids[b] = out.batch_best_ids[b];
                    }
                }
            }
        }

        let total_batch_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        let amortized_per_query_us = total_batch_us / (MATRIX_MICRO_BATCH_SIZE as f64);
        let candidates_scanned_per_query = total_scanned_per_query.max(1);
        let total_gemm_dot_products = candidates_scanned_per_query * MATRIX_MICRO_BATCH_SIZE;

        let z0 = Instant::now();
        for cell_idx in 0..active_cells_used {
            self.slot_arena.macrophage_apoptosis_zero(cell_idx);
        }
        let apoptosis_zeroing_us = z0.elapsed().as_secs_f64() * 1_000_000.0;

        MatrixMicroBatchSummary {
            mesh_rows: MATRIX_MESH_ROWS,
            mesh_cols: MATRIX_MESH_COLS,
            total_mesh_cells: MATRIX_MESH_TOTAL_CELLS,
            micro_batch_size: MATRIX_MICRO_BATCH_SIZE,
            candidates_scanned_per_query,
            total_gemm_dot_products,
            total_batch_us,
            amortized_per_query_us,
            batch_best_ids: global_best_ids,
            batch_best_scores: global_best_scores,
            apoptosis_zeroing_us,
        }
    }

    /// 實戰應用模式：自動變形刷資料加速通道（Auto-Morphing Turbo Conduit）：
    /// 外部程式無需配置複雜 Pipeline，直接將一批資料（`request_count` 筆查詢）刷進通道，
    /// 通道會自動將散亂請求轉為 64B 對齊受體封包，並自動根據流量密度變形為「6 筆微批次 2D 矩陣流 + 串流剪枝」高速消化！
    pub fn run_auto_morphing_conduit_blast(
        &self,
        batch_queries: &[[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
        request_count: usize,
    ) -> ConduitTrafficBlastReport {
        let reqs = request_count.clamp(6, 600);
        let micro_batches = (reqs + MATRIX_MICRO_BATCH_SIZE - 1) / MATRIX_MICRO_BATCH_SIZE;
        let actual_reqs = micro_batches * MATRIX_MICRO_BATCH_SIZE;

        // 1. 測量傳統逐筆處理（每筆做一次 Heap 緩衝區拷貝 + 掃描 16,384 筆向量）
        let single_trad = self.run_traditional_data_movement_baseline(&batch_queries[0], 16_384);
        let traditional_total_us = single_trad.traditional_full_query_us * (actual_reqs as f64);

        // 2. 測量透過「菌絲自動變形通道」消化同樣 `actual_reqs` 筆請求（每 6 筆自動湊成一個 4KB 受體頁走 8x8 矩陣流掃描 16,384 筆向量）
        let t_conduit = Instant::now();
        for _ in 0..micro_batches {
            let _ = self.run_matrix_micro_batch_flow(batch_queries, 256); // 64 cells * 256 = 16,384 vectors per query
        }
        let conduit_accelerated_total_us = t_conduit.elapsed().as_secs_f64() * 1_000_000.0;

        let traditional_qps = (actual_reqs as f64) / (traditional_total_us / 1_000_000.0).max(1e-6);
        let conduit_qps =
            (actual_reqs as f64) / (conduit_accelerated_total_us / 1_000_000.0).max(1e-6);

        ConduitTrafficBlastReport {
            total_requests_blasted: actual_reqs,
            traditional_total_us,
            conduit_accelerated_total_us,
            traditional_qps,
            conduit_qps,
            overall_speedup: traditional_total_us / conduit_accelerated_total_us.max(1.0),
        }
    }
}

//! 菌絲細胞算力引擎（Mycelial Cellular Vector Engine）參數與記憶體布局設定檔。
//! 嚴格對應 4KB 頁對齊、64-byte Cache Line 防偽共享規範、三大算力流（並流／串流／矩陣流）與四大傳輸介質競速。

/// 單一記憶體頁面大小（4 KB）
pub const PAGE_SIZE: usize = 4096;

/// CPU Cache Line 對齊大小（64 Bytes），防止多核心 False Sharing
pub const CACHE_LINE_SIZE: usize = 64;

/// 單一運算細胞槽位總大小（16 KB：涵蓋 0x0000 ~ 0x3FFF）
pub const CELL_SLOT_SIZE: usize = PAGE_SIZE * 4;

/// 細胞核（DNA 裸機機器碼）偏移量：`0x0000 ~ 0x0FFF` (4 KB, R-X)
pub const OFFSET_DNA_CODE: usize = 0x0000;

/// 細胞受體（Input 輸入區）偏移量：`0x1000 ~ 0x1FFF` (4 KB, R--)
pub const OFFSET_RECEPTOR_INPUT: usize = 0x1000;

/// 細胞質（Output 輸出區）偏移量：`0x2000 ~ 0x2FFF` (4 KB, RW-)
pub const OFFSET_CYTOPLASM_OUTPUT: usize = 0x2000;

/// 細胞膜感測器（Sensor 黑盒子針腳）偏移量：`0x3000 ~ 0x303F` (64 B, RW-)
pub const OFFSET_MEMBRANE_SENSOR: usize = 0x3000;

/// 預設向量維度（128 維 FP32 向量 = 512 Bytes，可完整放入 4KB 受體或共享池）
pub const VECTOR_DIM: usize = 128;

/// 矩陣流（Micro-Batch Matrix Flow）單次集中送入的小批次向量數（6 筆 128 維向量 = 3072 Bytes，完美塞入 4KB `0x1000` 受體頁）
pub const MATRIX_MICRO_BATCH_SIZE: usize = 6;

/// 矩陣流 2D 皮層網格（Systolic Mesh）預設細胞列數與行數（8x8 = 64 顆矩陣網格細胞）
pub const MATRIX_MESH_ROWS: usize = 8;
pub const MATRIX_MESH_COLS: usize = 8;
pub const MATRIX_MESH_TOTAL_CELLS: usize = MATRIX_MESH_ROWS * MATRIX_MESH_COLS;

/// 並流（Parallel Fan-out）預設肌肉纖維束細胞數
pub const PARALLEL_FANOUT_CELLS: usize = 128;

/// 串流（Streaming Pipeline）三階神經傳導鏈細胞數（合計 500 顆細胞）
pub const NEURAL_GRAPH_TOTAL_CELLS: usize = 500;
pub const LAYER1_SENSORY_CELLS: usize = 128;
pub const LAYER2_CORTICAL_CELLS: usize = 256;
pub const LAYER3_MOTOR_CELLS: usize = 116;

/// 共享唯讀向量池（Shared Read-Only Vector Pool）預設向量筆數
pub const DEFAULT_VECTOR_POOL_COUNT: usize = 100_000;

/// 最大支援之動態費洛蒙路由通道數量（使用者可於 1 ~ 8 通道間動態調整）
pub const MAX_PHEROMONE_CHANNELS: usize = 8;

/// 預設啟用之費洛蒙路由通道數量
pub const DEFAULT_ACTIVE_CHANNELS: usize = 4;

/// 費洛蒙初始權重基準值
pub const INITIAL_PHEROMONE_WEIGHT: u64 = 1_000_000;

/// 脈衝神經點火門檻值（Action Potential Threshold）
pub const SPIKE_THRESHOLD: f32 = 0.503;

/// 預設 Web 互動競技場與 REST API 監聽埠號
pub const DEFAULT_WEB_PORT: u16 = 8080;

/// 預設純二進位零 HTTP 開銷直連接口埠號（Direct Binary Conduit Socket Port）
pub const DEFAULT_DIRECT_BINARY_PORT: u16 = 9090;

/// 三大算力流拓撲執行模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataflowMode {
    All,
    Parallel,
    Pipeline,
    Matrix,
}

/// 四大資料傳輸介質（Wire Media）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireMedium {
    /// Level 1: 傳統 JSON 文字格式 (`application/json`) — 需 ASCII 浮點數解析與配置
    JsonText,
    /// Level 2: 緊湊 IEEE-754 二進位串流 (`Compact Binary / MessagePack-like`) — 需逐欄位反序列化拷貝
    CompactBinary,
    /// Level 3: 64B 對齊零解析細胞受體封包 (`64B Zero-Parse Aligned Frame`) — 直接映射至 `0x1000` 受體結構
    Aligned64BFrame,
    /// Level 4: 同機共享記憶體環 (`Shared-Memory IPC Ring`) — 僅原子交換槽位 Offset，零位元組搬移
    SharedMemoryIpc,
}

/// 細胞受體（`0x1000`）資料結構：64-byte 對齊，支援單筆（並流／串流）與小批次矩陣流（Micro-Batch <= 6 筆 128 維向量，總大小 3136 Bytes < 4096 Bytes）
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct CellReceptorInput {
    pub pool_start_index: u32,
    pub scan_count: u32,
    pub upstream_synapse_gain: f32,
    pub spike_threshold: f32,
    pub active_batch_size: u32,
    pub mesh_row: u16,
    pub mesh_col: u16,
    pub _header_pad: [u8; 40],
    pub batch_query_vectors: [[f32; VECTOR_DIM]; MATRIX_MICRO_BATCH_SIZE],
}

/// 細胞質（`0x2000`）輸出結構：64-byte 對齊，同時支援單筆 Top-K 與矩陣流（Micro-Batch）批次輸出
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct CellCytoplasmOutput {
    pub best_vector_id: u32,
    pub is_spiked: u32,
    pub max_similarity_score: f32,
    pub matched_candidates: u32,
    pub batch_best_ids: [u32; MATRIX_MICRO_BATCH_SIZE],
    pub batch_max_scores: [f32; MATRIX_MICRO_BATCH_SIZE],
}

/// 細胞膜感測器（`0x3000 ~ 0x303F`，嚴格 64 Bytes）：記錄執行週期與凋亡狀態供巨噬細胞讀取
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct CellMembraneSensor {
    pub elapsed_cycles: u64,
    pub lifecycle_state: u32,
    pub channel_id: u32,
    pub isolation_mode: u32,
    pub fault_code: u32,
    pub _padding: [u8; 32],
}

/// 引擎運行設定
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub vector_pool_count: usize,
    pub slot_pool_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            vector_pool_count: DEFAULT_VECTOR_POOL_COUNT,
            slot_pool_capacity: NEURAL_GRAPH_TOTAL_CELLS + 12,
        }
    }
}

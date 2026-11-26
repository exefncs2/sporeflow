//! 運行於瀏覽器前端的 100% 純 Rust WebAssembly 細胞受體封裝核心 (`wasm32-unknown-unknown`)。
//! 負責在瀏覽器端直接將使用者上傳／輸入的任意文字檔，於 Wasm 線性記憶體內高速執行 FNV-1a N-gram 特徵投影，
//! 直接產出 3,072 Bytes (6 筆 × 128 維 f32 Little-Endian) 的受體二進位封包，供前端以 `application/octet-stream` 零解析直刷後端！

#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const VECTOR_DIM: usize = 128;
const MICRO_BATCH: usize = 6;
const MAX_TEXT_INPUT: usize = 65536;
const OUT_BINARY_LEN: usize = MICRO_BATCH * VECTOR_DIM * 4; // 3,072 Bytes

static mut INPUT_TEXT_BUF: [u8; MAX_TEXT_INPUT] = [0u8; MAX_TEXT_INPUT];
static mut OUTPUT_RECEPTOR_BIN: [u8; OUT_BINARY_LEN] = [0u8; OUT_BINARY_LEN];
static mut LAST_WASM_CHECKSUM: u32 = 0;

/// 取得 Wasm 線性記憶體中之文字輸入緩衝區指標
#[no_mangle]
pub extern "C" fn wasm_input_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(INPUT_TEXT_BUF) as *mut u8
}

/// 取得 Wasm 線性記憶體中之 3,072 Bytes 受體二進位封包輸出指標
#[no_mangle]
pub extern "C" fn wasm_output_ptr() -> *const u8 {
    core::ptr::addr_of!(OUTPUT_RECEPTOR_BIN) as *const u8
}

/// 取得最近一次 Wasm 二進位封裝之 32-bit FNV 特徵校驗碼
#[no_mangle]
pub extern "C" fn wasm_last_checksum() -> u32 {
    unsafe { LAST_WASM_CHECKSUM }
}

/// 純 `no_std` 快速平方根近似（Newton-Raphson），供 Wasm 內部 L2 向量正規化使用
#[inline(always)]
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut i = x.to_bits();
    i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5 - 0.5 * x * y * y);
    let y = y * (1.5 - 0.5 * x * y * y);
    1.0 / y
}

/// 在瀏覽器端直接由 Rust WebAssembly 將長為 `text_len` 的任意文字切為 6 個分塊，
/// 並烘焙為 3,072 Bytes 的 `0x1000` 細胞受體二進位封包（回傳輸出封包 Byte 長度 = 3072）
#[no_mangle]
pub extern "C" fn wasm_bake_text_to_receptor_frame(text_len: usize) -> usize {
    let valid_len = if text_len > MAX_TEXT_INPUT {
        MAX_TEXT_INPUT
    } else {
        text_len
    };

    unsafe {
        let text_slice = &INPUT_TEXT_BUF[..valid_len];
        let chunk_step = if valid_len >= MICRO_BATCH {
            valid_len / MICRO_BATCH
        } else {
            1
        };

        let mut global_checksum: u32 = 0x811c9dc5;

        for b in 0..MICRO_BATCH {
            let start = (b * chunk_step).min(valid_len);
            let end = if b + 1 == MICRO_BATCH {
                valid_len
            } else {
                ((b + 1) * chunk_step).min(valid_len)
            };
            let sub = if start < end {
                &text_slice[start..end]
            } else {
                text_slice
            };

            let mut vec_f32 = [0.0f32; VECTOR_DIM];
            let mut hash_state: u64 = 0xcbf29ce484222325 ^ (b as u64 * 0x9e3779b97f4a7c15);

            for (i, &byte_val) in sub.iter().enumerate() {
                hash_state ^= byte_val as u64;
                hash_state = hash_state.wrapping_mul(0x100000001b3);
                global_checksum ^= byte_val as u32;
                global_checksum = global_checksum.wrapping_mul(0x01000193);
                let bucket = (i + (byte_val as usize)) % VECTOR_DIM;
                let signed_signal = ((hash_state & 0xFFFF) as f32 / 32768.0) - 1.0;
                vec_f32[bucket] += signed_signal;
            }

            let mut sum_sq = 0.0f32;
            for d in 0..VECTOR_DIM {
                hash_state ^= d as u64;
                hash_state = hash_state.wrapping_mul(0x100000001b3);
                let base_wave = ((hash_state & 0x7FFF) as f32 / 16384.0) - 1.0;
                vec_f32[d] += base_wave * 0.35;
                sum_sq += vec_f32[d] * vec_f32[d];
            }

            let inv_norm = if sum_sq > 1e-9 {
                1.0 / fast_sqrt(sum_sq)
            } else {
                0.08838835
            };

            for d in 0..VECTOR_DIM {
                let normalized = vec_f32[d] * inv_norm;
                let bytes = normalized.to_le_bytes();
                let out_off = (b * VECTOR_DIM + d) * 4;
                OUTPUT_RECEPTOR_BIN[out_off] = bytes[0];
                OUTPUT_RECEPTOR_BIN[out_off + 1] = bytes[1];
                OUTPUT_RECEPTOR_BIN[out_off + 2] = bytes[2];
                OUTPUT_RECEPTOR_BIN[out_off + 3] = bytes[3];
            }
        }

        LAST_WASM_CHECKSUM = global_checksum;
    }

    OUT_BINARY_LEN
}

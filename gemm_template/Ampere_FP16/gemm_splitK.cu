#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "cuda_runtime_api.h"
#include <algorithm>

using namespace nvcuda;
using namespace std;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define UINT2(pointer) (reinterpret_cast<uint2*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

typedef enum{
    HGEMMAlignedV1,
    HGEMMAlignedV2,
    HGEMMAlignedV3,
    HGEMMAlignedV4,
    HGEMMAlignedV5
} F16F16GemmTCAlgo_t;

void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
// all cache
#define CP_ASYNC_CA(dst, src, Bytes) \ 
    asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

//only L2 cache
#define CP_ASYNC_CG(dst, src, Bytes) \ 
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))


#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)


//#define K_STAGE 5 // 2 - 5
//#define BLOCK_ROWS 256 //256  32
//#define BLOCK_COLS 128 //128  32
#define BLOCK_K    32 

//#define WARP_ROWS 64 // 128 64 32 16 
//#define WARP_COLS 64 // 128 64 32 16
#define BLOCK_STRIDE 1 // < M / BLOCK_ROWS

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32
#define THREAD_COPY_BYTES 16
#define BLOCK_ROW_WARPS (BLOCK_COLS / WARP_COLS) // BLOCK_COLS / WARP_COLS 2
#define BLOCK_COL_WARPS (BLOCK_ROWS / WARP_ROWS)  // BLOCK_ROWS / WARP_ROWS 4

#define BLOCK_ROW_TILES (BLOCK_COLS / MMA_N)  // BLOCK_COLS / MMA_N 16
#define BLOCK_COL_TILES (BLOCK_ROWS / MMA_M)  // BLOCK_ROWS / MMA_M 16

#define WARP_ROW_TILES (WARP_COLS / MMA_N)  // WARP_COLS / MMA_N 8
#define WARP_COL_TILES (WARP_ROWS / MMA_M)  // WARP_ROWS / MMA_M 4

#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS 8
#define THREADS_PER_BLOCK WARP_SIZE * WARPS_PER_BLOCK  // WARP_SIZE * WARPS_PER_BLOCK 256

#define CHUNK_K (BLOCK_K / MMA_K)  // 32 / MMA_K 2

#define CHUNK_LINE_BYTES (CHUNK_K * MMA_K * sizeof(half))          // CHUNK_K * MMA_K * sizeof(half) 64
#define CHUNK_COPY_LINES_PER_WARP (WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES)  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES 8
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP 4

#define AB_SMEM_STRIDE (CHUNK_K * MMA_K)  // CHUNK_K * MMA_K 32

#define C_SMEM_STRIDE BLOCK_COLS  // BLOCK_COLS 128
#define C_SMEM_OFFSET WARP_COLS   // WARP_COLS 64


#define SMEM_BANK_ROWS (128 / (AB_SMEM_STRIDE * sizeof(half)))  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))
#define SMEM_WARP_OFFSET (BLOCK_ROWS + BLOCK_COLS)

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define ARRAY_OFFSET_INSMEM (PERMUTED_COLS * SMEM_BANK_ROWS) 

__global__ void mmaPermuted_Coalesced_32Stride(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                  size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K / 2, MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1)) : (blockIdx.y);
    const size_t block_tile_j = (blockIdx.z * (gridDim.x / 2) + blockIdx.x / 2); //swizzle
    const size_t block_tile_k = blockIdx.x % 2;

    extern __shared__ half smem[][AB_SMEM_STRIDE]; // AB_SMEM_STRIDE = 32

    const size_t warp_id = threadIdx.x / WARP_SIZE; // warp_size = 32
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS; // BLOCK_ROWS = 256

    int RA[2][WARP_COL_TILES][4];
    int RB[2][WARP_ROW_TILES][2];
    int RC[WARP_COL_TILES][WARP_ROW_TILES][2] = {0};

    const half *A_warp_ptr = &A[block_tile_i * BLOCK_ROWS * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id + block_tile_k * (K / 2);

    const half *B_warp_ptr = &B[block_tile_j * BLOCK_COLS * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id + block_tile_k * (K / 2);

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    //

    size_t stage_store_smem = 0;
    int4 *A_lane_ptr = nullptr;
    int4 *B_lane_ptr = nullptr;

    // -------------------------------- load first buffer -----------------------------------------
    #pragma unroll
    for(int stage_load_id = 0; stage_load_id < (K_STAGE - 1); ++stage_load_id){ // read 3 buffer
        //------------------Load gloabl to smem----------------------------
        A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K + stage_store_smem * BLOCK_K) +
                            (lane_id % CHUNK_COPY_LINE_LANES);
        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;
        A_smem_idx += stage_store_smem * (SMEM_WARP_OFFSET);

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K + stage_store_smem * BLOCK_K) +
                            (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;
        B_smem_idx += stage_store_smem * (SMEM_WARP_OFFSET);

        #pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            int A_smem_ptr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                ((lane_id % CHUNK_COPY_LINE_LANES) +
                (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                    CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES; 

            CP_ASYNC_CG(A_smem_ptr, A_lane_ptr, THREAD_COPY_BYTES); 

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        #pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            int B_smem_ptr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) + 
                ((lane_id % CHUNK_COPY_LINE_LANES) +
                (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                    CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES;
            CP_ASYNC_CG(B_smem_ptr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        stage_store_smem = (stage_store_smem + 1) % K_STAGE;
    }

    CP_ASYNC_WAIT_GROUP((K_STAGE-2));
    __syncthreads();

    size_t stage_store_id = (K_STAGE - 1);
    size_t stage_use_id = 0;
    size_t reg_store_id = 0;
    size_t reg_use_id = 0;

    //-------------------------------load first register-----------------------------------
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M + (stage_use_id) * (SMEM_WARP_OFFSET);
        int A_smem_lane_addr = __cvta_generic_to_shared(
            &smem[A_smem_idx + lane_id % 16]
                    [((lane_id / 16) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);

        LDMATRIX_X4(RA[reg_store_id][i][0], RA[reg_store_id][i][1], RA[reg_store_id][i][2], RA[reg_store_id][i][3], A_smem_lane_addr);
    }

    #pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; j ++) {
        size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + (stage_use_id) * (SMEM_WARP_OFFSET);

        int B_smem_lane_addr = __cvta_generic_to_shared(
            &smem[B_smem_idx + lane_id % 8]
                    [( ((lane_id / 8) % 2) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);
        LDMATRIX_X2(RB[reg_store_id][j][0], RB[reg_store_id][j][1], B_smem_lane_addr);
    
    }

    reg_store_id = (reg_store_id + 1) % 2;
    

    //-----------------------------------------------main loop-----------------------------
    #pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < (K_tiles); tile_k += CHUNK_K) {
        //
        //-------------------------------load second register-----------------------------------
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M + (stage_use_id) * (SMEM_WARP_OFFSET);
            int A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                        [(MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                        AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_id][i][0], RA[reg_store_id][i][1], RA[reg_store_id][i][2], RA[reg_store_id][i][3], A_smem_lane_addr);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; j ++) {
            size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + (stage_use_id) * (SMEM_WARP_OFFSET);
            int B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                    [(MMA_K + ((lane_id / 8) % 2) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);
            LDMATRIX_X2(RB[reg_store_id][j][0], RB[reg_store_id][j][1], B_smem_lane_addr);
        }

        //---------------------------------cal first register-----------------------------------
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j; // Z permute for max util for regA
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                          RA[reg_use_id][i][0], RA[reg_use_id][i][1], RA[reg_use_id][i][2], RA[reg_use_id][i][3], 
                          RB[reg_use_id][j_s][0], RB[reg_use_id][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        //-----------------------------------Load last buffer----------------------------
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES); 

        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;
        A_smem_idx += (stage_store_id) * (SMEM_WARP_OFFSET);

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES); 

        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;
        B_smem_idx += (stage_store_id) * (SMEM_WARP_OFFSET);

        #pragma unroll
        for (size_t i = 0; i < A_smem_iters / 2; ++i) {
            
            int A_smem_ptr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
              ((lane_id % CHUNK_COPY_LINE_LANES) +
               (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                  CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES; 

            CP_ASYNC_CG(A_smem_ptr, A_lane_ptr, THREAD_COPY_BYTES); 

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        #pragma unroll
        for (size_t i = 0; i < B_smem_iters / 2; ++i) {
            
            int B_smem_ptr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) + 
               ((lane_id % CHUNK_COPY_LINE_LANES) +
               (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                  CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES;
            CP_ASYNC_CG(B_smem_ptr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        #pragma unroll
        for (size_t i = A_smem_iters / 2; i < A_smem_iters; ++i) {
            
            int A_smem_ptr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
              ((lane_id % CHUNK_COPY_LINE_LANES) +
               (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                  CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES; 

            CP_ASYNC_CG(A_smem_ptr, A_lane_ptr, THREAD_COPY_BYTES); 

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        #pragma unroll
        for (size_t i = B_smem_iters / 2; i < B_smem_iters; ++i) {
            
            int B_smem_ptr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) + 
               ((lane_id % CHUNK_COPY_LINE_LANES) +
               (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                  CHUNK_COPY_LINE_LANES * THREAD_COPY_BYTES;
            CP_ASYNC_CG(B_smem_ptr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
        //---------------------------------------------------------------------------------------
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(K_STAGE-2);
        __syncthreads();

        stage_use_id = (stage_use_id + 1) % K_STAGE;
        stage_store_id = (stage_store_id + 1) % K_STAGE;
        reg_use_id = (reg_use_id + 1) % 2;
        reg_store_id = (reg_use_id + 1) % 2;

        //-------------------------------load first register-----------------------------------
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M + (stage_use_id) * (SMEM_WARP_OFFSET);
            int A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                        [((lane_id / 16) * 8 +
                        (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                        AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_id][i][0], RA[reg_store_id][i][1], RA[reg_store_id][i][2], RA[reg_store_id][i][3], A_smem_lane_addr);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; j ++) {
            size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + (stage_use_id) * (SMEM_WARP_OFFSET);

            int B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                    [(((lane_id / 8) % 2) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);
            LDMATRIX_X2(RB[reg_store_id][j][0], RB[reg_store_id][j][1], B_smem_lane_addr);
        
        }

        //------------------------------cal second register
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j; // Z permute for max util for regA
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                          RA[reg_use_id][i][0], RA[reg_use_id][i][1], RA[reg_use_id][i][2], RA[reg_use_id][i][3], 
                          RB[reg_use_id][j_s][0], RB[reg_use_id][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
        
        reg_store_id = (reg_store_id + 1) % 2;
        reg_use_id = (reg_use_id + 1) % 2;
    }

    //-------------------------------------epolgue--------------------------------------------------------
    #pragma unroll
    for (size_t stage = 0; stage < (K_STAGE - 2); ++stage){
        #pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {         
            //-------------------------------load smem to reg-----------------------------------
            #pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M + (stage_use_id) * (SMEM_WARP_OFFSET);
                int A_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[A_smem_idx + lane_id % 16]
                            [(((k_step + 1) % 2) * MMA_K + (lane_id / 16) * 8 +
                            (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                            AB_SMEM_STRIDE]);

                LDMATRIX_X4(RA[reg_store_id][i][0], RA[reg_store_id][i][1], RA[reg_store_id][i][2], RA[reg_store_id][i][3], A_smem_lane_addr);
            }

            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; j ++) {
                size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + (stage_use_id) * (SMEM_WARP_OFFSET);
                int B_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[B_smem_idx + lane_id % 8]
                    [(((k_step + 1) % 2) * MMA_K + ((lane_id / 8) % 2) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);
                LDMATRIX_X2(RB[reg_store_id][j][0], RB[reg_store_id][j][1], B_smem_lane_addr);
            }
            #pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j; // Z permute for max util for regA
                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                          RA[reg_use_id][i][0], RA[reg_use_id][i][1], RA[reg_use_id][i][2], RA[reg_use_id][i][3], 
                          RB[reg_use_id][j_s][0], RB[reg_use_id][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
            reg_store_id = (reg_store_id + 1) % 2;
            reg_use_id = (reg_use_id + 1) % 2;
            if (k_step == 0){
                stage_use_id = (stage_use_id + 1) % K_STAGE;
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {         
        //-------------------------------load smem to reg-----------------------------------
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M + (stage_use_id) * (SMEM_WARP_OFFSET);
            int A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                        [(k_step * MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                        AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_id][i][0], RA[reg_store_id][i][1], RA[reg_store_id][i][2], RA[reg_store_id][i][3], A_smem_lane_addr);
        }

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; j ++) {
            size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + (stage_use_id) * (SMEM_WARP_OFFSET);
            int B_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[B_smem_idx + lane_id % 8]
                    [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                    (lane_id % 16 % (ARRAY_OFFSET_INSMEM)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                    AB_SMEM_STRIDE]);
            LDMATRIX_X2(RB[reg_store_id][j][0], RB[reg_store_id][j][1], B_smem_lane_addr);
        }
        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j; // Z permute for max util for regA
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                        RA[reg_use_id][i][0], RA[reg_use_id][i][1], RA[reg_use_id][i][2], RA[reg_use_id][i][3], 
                        RB[reg_use_id][j_s][0], RB[reg_use_id][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
        reg_store_id = (reg_store_id + 1) % 2;
        reg_use_id = (reg_use_id + 1) % 2;
    }

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
    #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j; // Z permute for max util for regA
            HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_use_id][i][0], RA[reg_use_id][i][1], RA[reg_use_id][i][2], RA[reg_use_id][i][3], 
                    RB[reg_use_id][j_s][0], RB[reg_use_id][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
    }
    __syncthreads();
//direct write back to global memory
    int C_row_block = block_tile_i * BLOCK_ROWS;
    int C_col_block = block_tile_j * BLOCK_COLS;
    int C_row_warp = warp_id / BLOCK_ROW_WARPS * WARP_ROWS;
    int C_col_warp = warp_id % BLOCK_ROW_WARPS * WARP_COLS;
    half* c_global_ptr = &C[(C_row_block + C_row_warp) * N + C_col_block + C_col_warp];
#pragma unroll
    for(size_t i = 0; i < WARP_COL_TILES;++i){
#pragma unroll
        for(size_t j = 0; j < WARP_ROW_TILES; ++j){
	        atomicAdd((half2*)(c_global_ptr + i * MMA_M * N + lane_id / 4 * N + lane_id % 4 * 2 + j * MMA_N), *(half2*)(&RC[i][j][0]));
            atomicAdd((half2*)(c_global_ptr + i * MMA_M * N + lane_id / 4 * N + 8 * N + lane_id % 4 * 2 + j * MMA_N), *(half2*)(&RC[i][j][1]));
        }
    }
}


size_t initMmaPermuted() {
    int dev_id = 0;
    (cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    (cudaGetDeviceProperties(&dev_prop, dev_id));

    int smem_max_size = std::max((SMEM_WARP_OFFSET) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));

    (dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    (cudaFuncSetAttribute(mmaPermuted_Coalesced_32Stride, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaPermuted(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaPermuted();
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE * 2, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));
    mmaPermuted_Coalesced_32Stride<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}


template<F16F16GemmTCAlgo_t algo = HGEMMAlignedV1>
void myF16F16GemmTCWarp(half *a, half *b, half *c, int M, int N, int K) {

    if(algo == HGEMMAlignedV2){
        mmaPermuted(a, b, c, M, N, K);
    }
}
float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *h_b_t, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_b_t = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++){
        h_a[i] = (half)(float(rand()) / float(RAND_MAX));
    }
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    
    for (int i =0; i < N; ++i){
        for(int j = 0; j < K; ++j){
		h_b_t[i * K + j] = h_b[j * N + i];
        }
    }

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_t, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}
float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K, int repeat) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    for (int i = 0; i < 10; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return sec;
}
int main(int arg, char* argv[]){
    const int test_num = 5;
    const int M_list[test_num] = {16, 512, 1024, 4096, 8192};
    const int N_list[test_num] = {8 , 512, 1024, 4096, 8192};
    const int K_list[test_num] = {16, 128, 1024, 4096, 8192};
    const int outer_repeat = 1, inner_repeat = 10;

    for (int j = 4; j < test_num; j++){
        int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
        N = (N % BLOCK_COLS == 0) ? N : ((N / BLOCK_COLS + 1) * BLOCK_COLS);
	    K = (K % BLOCK_K == 0) ? K : ((K / BLOCK_K + 1) * BLOCK_K);
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance(
                myF16F16GemmTCWarp<HGEMMAlignedV2>, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / avg_sec;

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
    }
    return 0;
}



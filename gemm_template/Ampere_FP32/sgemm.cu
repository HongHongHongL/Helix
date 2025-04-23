#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer))[0])

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

// all cache
#define CP_ASYNC_CA(dst, src, Bytes) \ 
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

//only L2 cache
#define CP_ASYNC_CG(dst, src, Bytes) \ 
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

//---------configurable----
//#define BM 4 
//#define BN 32 
//#define BK 32 
//#define TM 1
//#define TN 1
//#define COPY_A_SHM_REG_FLOAT 1
//#define COPY_B_SHM_REG_FLOAT 1
//-----------------------
// WARP_ROW_THREAD * TN <= BN
// WARP_COL_THREAD * TM <= BM
//#define WARP_ROW_THREAD 2 
#define WARP_COL_THREAD (32 / WARP_ROW_THREAD)
#define COPY_PER_THREAD_BYTES 16
#define COPY_COUNT_PER_THREAD (COPY_PER_THREAD_BYTES / sizeof(float)) // 4
#define WARP_M (WARP_ROW_THREAD * TM) // 
#define WARP_N (WARP_COL_THREAD * TN) // 

#define COL_WARP_NUM (BM % WARP_M == 0 ? (BM / WARP_M) : (BM / WARP_M + 1)) // 4
#define ROW_WARP_NUM (BN % WARP_N == 0 ? (BN / WARP_N) : (BN / WARP_N + 1)) // 2
#define WARP_NUM (ROW_WARP_NUM * COL_WARP_NUM)

#define WARP_M_LOOP (TM / COPY_A_SHM_REG_FLOAT)
#define WARP_N_LOOP (TN / COPY_B_SHM_REG_FLOAT)
#define WARP_ONCE_M (WARP_M / WARP_M_LOOP)
#define WARP_ONCE_N (WARP_N / WARP_N_LOOP)

#define PER_M_LINE_THREADS (BM % COPY_COUNT_PER_THREAD == 0 ? (BM / COPY_COUNT_PER_THREAD) : (BM / COPY_COUNT_PER_THREAD + 1)) // 32
#define PER_M_WARP_COPY_LINES (BK % WARP_NUM == 0 ? (BK / WARP_NUM) : (BK / WARP_NUM + 1)) // BK / PER_M_WARP_COPY_LINES / WARP_PER_BLOCK
#define ONCE_M_WARP_COPY_LINES ((32 / PER_M_LINE_THREADS) < PER_M_WARP_COPY_LINES ? (32 / PER_M_LINE_THREADS) : PER_M_WARP_COPY_LINES) // BK / PER_M_WARP_COPY_LINES / WARP_PER_BLOCK

#define PER_N_LINE_THREADS (BN % COPY_COUNT_PER_THREAD == 0 ? (BN / COPY_COUNT_PER_THREAD) : (BN / COPY_COUNT_PER_THREAD + 1)) // 32
#define PER_N_WARP_COPY_LINES (BK % WARP_NUM == 0 ? (BK / WARP_NUM) : (BK / WARP_NUM + 1))   // 1
#define ONCE_N_WARP_COPY_LINES ((32 / PER_N_LINE_THREADS) < PER_N_WARP_COPY_LINES ? (32 / PER_N_LINE_THREADS) : PER_N_WARP_COPY_LINES)   // 1

#define BLOCK_STRIDE 1 
#define mstage 2
__global__ void mySgemmCPAsyncTemplate(float * a, float * b, float * c, const int M, const int N, const int K){
    //const int bx = blockIdx.x;
    //const int by = blockIdx.y;
    const int bx = (blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : (blockIdx.y);
    const int by = (blockIdx.z * gridDim.x + blockIdx.x);
    const int tid = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    __shared__ float s_a[mstage][BK][BM];
    __shared__ float s_b[mstage][BK][BN];

    float r_comp_a[2][TM];
    float r_comp_b[2][TN];
    float r_c[TM][TN] = {0.0};

    //constexpr int A_smem_iters = BK / PER_M_WARP_COPY_LINES / WARP_NUM;
    //constexpr int B_smem_iters = BK / PER_N_WARP_COPY_LINES / WARP_NUM;
    constexpr int A_smem_iters = PER_M_WARP_COPY_LINES / ONCE_M_WARP_COPY_LINES; 
    constexpr int B_smem_iters = PER_N_WARP_COPY_LINES / ONCE_N_WARP_COPY_LINES;

    int load_a_gmem_row = warp_id * PER_M_WARP_COPY_LINES + (tid / PER_M_LINE_THREADS) % ONCE_M_WARP_COPY_LINES;
    int load_a_gmem_col = by * BM + (tid % PER_M_LINE_THREADS) * COPY_COUNT_PER_THREAD;
   
    int load_b_gmem_row = warp_id * PER_N_WARP_COPY_LINES + (tid / PER_N_LINE_THREADS) % ONCE_N_WARP_COPY_LINES;
    int load_b_gmem_col = bx * BN + (tid % PER_N_LINE_THREADS) * COPY_COUNT_PER_THREAD;

    int store_a_smem_row = load_a_gmem_row;
    int store_a_smem_col = (tid % PER_M_LINE_THREADS) * COPY_COUNT_PER_THREAD;

    int store_b_smem_row = load_b_gmem_row;
    int store_b_smem_col = (tid % PER_N_LINE_THREADS) * COPY_COUNT_PER_THREAD;

    int store_smem_idx = 0;
    int load_smem_idx = 0;
    int store_reg_idx = 0;
    int load_reg_idx = 0;

    int load_a_smem_row, load_a_smem_col, load_b_smem_row, load_b_smem_col;
    #pragma unroll
    for(int iter_a = 0;iter_a < A_smem_iters; ++iter_a){
        int A_smem_ptr = __cvta_generic_to_shared(&s_a[store_smem_idx][(store_a_smem_row + ONCE_M_WARP_COPY_LINES * iter_a) % BK][store_a_smem_col % BM]);
        int4 *A_lane_ptr = (int4*)(&a[(load_a_gmem_row + ONCE_M_WARP_COPY_LINES * iter_a) * M + load_a_gmem_col]);
	CP_ASYNC_CA(A_smem_ptr, A_lane_ptr, 16);
        //FLOAT4(s_a[store_a_smem_row % BK][store_a_smem_col % BM]) = FLOAT4(a[(load_a_gmem_row) * M + load_a_gmem_col]);
        //{
        //    float* aa = reinterpret_cast<float*>(&(a[(load_a_gmem_row) * M + load_a_gmem_col]));
        //    printf("AA : tid  : %d, gmem row: %d, gmem col : %d,  %.4f, %.4f, %.4f, %.4f\n", threadIdx.x,  load_a_gmem_row, load_a_gmem_col, float(aa[0]), float(aa[1]), float(aa[2]), float(aa[3]));
        //}
    }
    store_a_smem_row += BK;
    load_a_gmem_row += BK;
    #pragma unroll 
    for(int iter_b = 0; iter_b < B_smem_iters; ++iter_b){
        int B_smem_ptr = __cvta_generic_to_shared(&s_b[store_smem_idx][(store_b_smem_row + ONCE_N_WARP_COPY_LINES * iter_b) % BK][store_b_smem_col % BN]);
        int4* B_lane_ptr = (int4*)(&b[(load_b_gmem_row + ONCE_N_WARP_COPY_LINES * iter_b) * N + load_b_gmem_col]);
        CP_ASYNC_CG(B_smem_ptr, B_lane_ptr, 16);
    }   
    store_b_smem_row += BK;
    load_b_gmem_row += BK;

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    store_smem_idx = (store_smem_idx + 1) % mstage;

    for(int iter_k = 1; iter_k < (K / BK); iter_k++){
        load_a_smem_row = 0;
        //load_a_smem_col = warp_id / WARP_ROW_NUM * WARP_M + tid / 16 * 8 + (tid % 16) % 2 * (TM / WARP_M_LOOP);
        load_a_smem_col = warp_id / ROW_WARP_NUM * WARP_M + tid / WARP_COL_THREAD * (TM / WARP_M_LOOP);
        load_b_smem_row = 0;
        //load_b_smem_col = warp_id % WARP_ROW_NUM * WARP_N + (tid % 16 / 2) * (TN / WARP_N_LOOP);
        load_b_smem_col = (warp_id % ROW_WARP_NUM) * WARP_N + (tid % WARP_COL_THREAD) * (TN / WARP_N_LOOP);
        store_reg_idx = 0;
        load_reg_idx = 0;

        #pragma unroll
        for(int ai = 0; ai < (TM / COPY_A_SHM_REG_FLOAT); ++ai){
            #if (COPY_A_SHM_REG_FLOAT == 4)
	        FLOAT4(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT4(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
            #elif (COPY_A_SHM_REG_FLOAT == 2)
                FLOAT2(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT2(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
            #else
                FLOAT(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
            #endif
        }
        #pragma unroll
        for(int bi = 0; bi < (TN / COPY_B_SHM_REG_FLOAT); ++bi){
           #if (COPY_B_SHM_REG_FLOAT == 4)
	       FLOAT4(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT4(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #elif (COPY_B_SHM_REG_FLOAT == 2)
	       FLOAT2(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT2(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #else
	       FLOAT(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #endif
        }
        store_reg_idx ^= 1;
        
        #pragma unroll
	for(int iter_a = 0;iter_a < A_smem_iters; ++iter_a){
	    int A_smem_ptr = __cvta_generic_to_shared(&s_a[store_smem_idx][(store_a_smem_row + ONCE_M_WARP_COPY_LINES * iter_a) % BK][store_a_smem_col % BM]);
	    int4 *A_lane_ptr = (int4*)(&a[(load_a_gmem_row + ONCE_M_WARP_COPY_LINES * iter_a) * M + load_a_gmem_col]);
            CP_ASYNC_CA(A_smem_ptr, A_lane_ptr, 16);
            //FLOAT4(s_a[store_a_smem_row % BK][store_a_smem_col % BM]) = FLOAT4(a[(load_a_gmem_row) * M + load_a_gmem_col]);
	}
        store_a_smem_row += BK;
        load_a_gmem_row += BK;

        #pragma unroll
        for(int iter_b = 0; iter_b < B_smem_iters; ++iter_b){
	    int B_smem_ptr = __cvta_generic_to_shared(&s_b[store_smem_idx][(store_b_smem_row + ONCE_N_WARP_COPY_LINES * iter_b) % BK][store_b_smem_col % BN]);
	    int4* B_lane_ptr = (int4*)(&b[(load_b_gmem_row + ONCE_N_WARP_COPY_LINES * iter_b) * N + load_b_gmem_col]);
	    CP_ASYNC_CG(B_smem_ptr, B_lane_ptr, 16);
            //FLOAT4(s_b[store_b_smem_row % BK][store_b_smem_col % BN]) = FLOAT4(b[(load_b_gmem_row) * N + load_b_gmem_col]);
	}
        store_b_smem_row += BK;
        load_b_gmem_row += BK;

        #pragma unroll
        for(int j = 1; j < BK; ++j){
	   load_a_smem_row = j;
	   load_b_smem_row = j;
           #pragma unroll
	   for(int ai = 0; ai < (TM / COPY_A_SHM_REG_FLOAT); ++ai){
               #if (COPY_A_SHM_REG_FLOAT == 4)
	           FLOAT4(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT4(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
               #elif (COPY_A_SHM_REG_FLOAT == 2)
                   FLOAT2(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT2(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
               #else
                   FLOAT(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
	       #endif
	   }
           #pragma unroll
	   for(int bi = 0; bi < (TN / COPY_B_SHM_REG_FLOAT); ++bi){
	       #if (COPY_B_SHM_REG_FLOAT == 4)
	           FLOAT4(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT4(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	       #elif (COPY_B_SHM_REG_FLOAT == 2)
		   FLOAT2(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT2(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	       #else
		   FLOAT(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	       #endif
	   }
           #pragma unroll
	   for(int tm = 0; tm < TM; ++tm){
               #pragma unroll
	       for(int tn = 0; tn < TN; ++tn){
	           r_c[tm][tn] += r_comp_a[load_reg_idx][tm] * r_comp_b[load_reg_idx][tn];
	       }
	   }
	   store_reg_idx ^= 1;
	   load_reg_idx ^= 1;
	}
        #pragma unroll
	for(int tm = 0; tm < TM; ++tm){
            #pragma unroll
	    for(int tn = 0; tn < TN; ++tn){
	        r_c[tm][tn] += r_comp_a[load_reg_idx][tm] * r_comp_b[load_reg_idx][tn];
	    }
	}
	CP_ASYNC_COMMIT_GROUP();
	CP_ASYNC_WAIT_GROUP(0);
	__syncthreads();
	load_smem_idx = (load_smem_idx + 1) % mstage;
	store_smem_idx = (store_smem_idx + 1) % mstage;
    }

    load_a_smem_row = 0;
    //load_a_smem_col = warp_id / WARP_ROW_NUM * WARP_M + tid / 16 * 8 + (tid % 16) % 2 * (TM / WARP_M_LOOP);
    load_a_smem_col = (warp_id / ROW_WARP_NUM) * WARP_M + tid / WARP_COL_THREAD * (TM / WARP_M_LOOP);
    load_b_smem_row = 0;
    //load_b_smem_col = warp_id % WARP_ROW_NUM * WARP_N + (tid % 16 / 2) * (TN / WARP_N_LOOP);
    load_b_smem_col = (warp_id % ROW_WARP_NUM) * WARP_N + (tid % WARP_COL_THREAD) * (TN / WARP_N_LOOP);
    store_reg_idx = 0;
    load_reg_idx = 0;

    #pragma unroll
    for(int ai = 0; ai < (TM / COPY_A_SHM_REG_FLOAT); ++ai){
           #if (COPY_A_SHM_REG_FLOAT == 4)
	       FLOAT4(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT4(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
           #elif (COPY_A_SHM_REG_FLOAT == 2)
               FLOAT2(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT2(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
           #else
               FLOAT(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
	   #endif
    }
    #pragma unroll
    for(int bi = 0; bi < (TN / COPY_B_SHM_REG_FLOAT); ++bi){
           #if (COPY_B_SHM_REG_FLOAT == 4)
	       FLOAT4(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT4(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #elif (COPY_B_SHM_REG_FLOAT == 2)
	       FLOAT2(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT2(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #else
	       FLOAT(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	    #endif
	   
    }
    store_reg_idx ^= 1;

    #pragma unroll
    for(int j = 1; j < BK ; ++j){
      load_a_smem_row = j;
      load_b_smem_row = j;
      #pragma unroll
      for(int ai = 0; ai < (TM / COPY_A_SHM_REG_FLOAT); ++ai){
           #if (COPY_A_SHM_REG_FLOAT == 4)
	       FLOAT4(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT4(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
           #elif (COPY_A_SHM_REG_FLOAT == 2)
               FLOAT2(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT2(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
           #else
               FLOAT(r_comp_a[store_reg_idx][0 + ai * COPY_A_SHM_REG_FLOAT]) = FLOAT(s_a[load_smem_idx][load_a_smem_row][load_a_smem_col + ai * WARP_ONCE_M]);
	   #endif
      }
      #pragma unroll
      for(int bi = 0; bi < (TN / COPY_B_SHM_REG_FLOAT); ++bi){
           #if (COPY_B_SHM_REG_FLOAT == 4)
	       FLOAT4(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT4(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #elif (COPY_B_SHM_REG_FLOAT == 2)
	       FLOAT2(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT2(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #else
	       FLOAT(r_comp_b[store_reg_idx][0 + bi * COPY_B_SHM_REG_FLOAT]) = FLOAT(s_b[load_smem_idx][load_b_smem_row][load_b_smem_col + bi * WARP_ONCE_N]);
	   #endif
      }
      for(int tm = 0; tm < TM; ++tm){
          for(int tn = 0; tn < TN; ++tn){
              r_c[tm][tn] += r_comp_a[load_reg_idx][tm] * r_comp_b[load_reg_idx][tn];
          }
      }  
      store_reg_idx ^= 1;
      load_reg_idx ^= 1;
    } 
    for(int tm = 0; tm < TM; ++tm){
        for(int tn = 0; tn < TN; ++tn){
            r_c[tm][tn] += r_comp_a[load_reg_idx][tm] * r_comp_b[load_reg_idx][tn];
        }
    }

    //int store_c_global_row = by * BM + warp_id / WARP_ROW_NUM * WARP_M + (tid / 16) * 8 + (tid % 16) % 2 * (TM / WARP_M_LOOP);
    int store_c_global_row = by * BM + warp_id / ROW_WARP_NUM * WARP_M + (tid / WARP_COL_THREAD) * (TM / WARP_M_LOOP);
    //int store_c_global_col = bx * BN + warp_id % WARP_ROW_NUM * WARP_N + (tid % 16 / 2)  * (TN / WARP_N_LOOP);
    int store_c_global_col = bx * BN + warp_id % ROW_WARP_NUM * WARP_N + (tid % WARP_COL_THREAD)  * (TN / WARP_N_LOOP);

    for(int i = 0; i < WARP_M_LOOP; ++i){
        for(int j = 0; j < WARP_N_LOOP; ++j){
            for(int tm = 0; tm < COPY_A_SHM_REG_FLOAT; ++tm){
		#if (COPY_B_SHM_REG_FLOAT == 4)
	            FLOAT4(c[(store_c_global_row + i * (WARP_ONCE_M) + tm) * N + store_c_global_col + j * WARP_ONCE_N]) = FLOAT4(r_c[i * (TM / WARP_M_LOOP) + tm][j * (TN / WARP_N_LOOP)]);
		#elif (COPY_B_SHM_REG_FLOAT == 2)
		    FLOAT2(c[(store_c_global_row + i * (WARP_ONCE_M) + tm) * N + store_c_global_col + j * WARP_ONCE_N]) = FLOAT2(r_c[i * (TM / WARP_M_LOOP) + tm][j * (TN / WARP_N_LOOP)]);
		#else
		    FLOAT(c[(store_c_global_row + i * (WARP_ONCE_M) + tm) * N + store_c_global_col + j * WARP_ONCE_N]) = FLOAT(r_c[i * (TM / WARP_M_LOOP) + tm][j * (TN / WARP_N_LOOP)]);
		#endif
	    }
	}
    }
}

float testMaxError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_t_a, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_t_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
        //h_a[i] = float(i) / 10;
    for (int i = 0; i < K * N; i++)
        //h_b[i] = float(i) / 10;
        h_b[i] = rand() / float(RAND_MAX);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
	    h_t_a[j * M + i] = h_a[i * K + j];
	}
    }

    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_t_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    //for(int i = 0; i < M; ++i){
    //    for(int j = 0; j < N; ++j){
    //        printf("%f, ", h_c[i * N + j]);
    //    }
    //    printf("\n");
    //}
    //printf("===============================\n");
    //for(int i = 0; i < M; ++i){
    //    for(int j = 0; j < N; ++j){
    //        printf("%f, ", h_d_c[i * N + j]);
    //    }
    //    printf("\n");
    //}

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testCublasMaxError(const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *h_t_a, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_t_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
	//h_a[i] = float(i) / 100;
        h_a[i] = rand() / float(RAND_MAX);

    for (int i = 0; i < K * N; i++)
	//h_b[i] = float(i) / 100;
        h_b[i] = rand() / float(RAND_MAX);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
	    h_t_a[i * K + j] = h_a[j * M + i];
	}
    }

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_t_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

int main(int arg, char* argv[]) {

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    //const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    //const int M_list[1] = {1024};
    //const int N_list[1] = {1024};
    //const int K_list[1] = {1024};
    const int outer_repeat = 1, inner_repeat = 10;
    {
        //printf("\nKernal = cublas\n");

        //{
        //    const int TESTNUM = 15;

        //    for (int i = 14; i < TESTNUM; i++) {
        //        //const int M = M_list[i], N = N_list[i], K = K_list[i];
        //	const int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
        //	

        //        double max_sec = 0.0;
        //        double min_sec = DBL_MAX;
        //        double total_sec = 0.0;

        //        for (int j = 0; j < outer_repeat; j++) {
        //            double this_sec = testCublasPerformance(M, N, K, inner_repeat);
        //            max_sec = max(max_sec, this_sec);
        //            min_sec = min(min_sec, this_sec);
        //            total_sec += this_sec;
        //        }

        //        double avg_sec = total_sec / outer_repeat;
        //        double avg_Gflops = ((double)M) * N * K * 2 / 1e9 / avg_sec;

        //        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
        //    }
        //}
    }
    {
        printf("\nKernal = mySgemmCPAsyncTemplate\n");

        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) =
            mySgemmCPAsyncTemplate;

        {
            const int M = 128, N = 128, K = 128;
            //dim3 blockDim(((BN / TN) * (BM / TM)));
            //dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            int threads_per_block = (BN / TN) * (BM / TM);
            dim3 blockDim(threads_per_block); //16, 16
            dim3 gridDim(BLOCK_STRIDE, div_ceil(N, BN), div_ceil(M, BM * BLOCK_STRIDE)); //blocks

            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 15;

            for (int i = 14; i < TESTNUM; i++) {
                //const int M = M_list[i], N = N_list[i], K = K_list[i];
                const int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);

		int threads_per_block = (BN / TN) * (BM / TM);
                dim3 blockDim(threads_per_block); //16, 16
                dim3 gridDim(BLOCK_STRIDE, div_ceil(N, BN), div_ceil(M, BM * BLOCK_STRIDE)); //blocks

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1e9 / avg_sec;

                // printf("%12.8lf\n", avg_sec);
                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    return 0;
}


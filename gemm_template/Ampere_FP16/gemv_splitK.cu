#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "cuda_runtime_api.h"
#include <algorithm>
#include <stdint.h>

using namespace nvcuda;
using namespace std;

#define CUTLASS_ENABLE_L2_PREFETCH False 
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define UINT2(pointer) (reinterpret_cast<uint2*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

typedef enum{
    HGEMMAlignedV5,
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

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

__device__ void LDG_L1_128bit_LAST(int4& dst, const uint8_t* ptr, bool pred_guard=true) \
{
    uint4 &data = reinterpret_cast<uint4 &>(dst);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
}



__device__ void LDG_L1_128bit_ALWAYS(int4& dst, const uint8_t* ptr, bool pred_guard=true) \
{
  uint4 &data = reinterpret_cast<uint4 &>(dst);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#else
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#endif
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));

}


#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

//#define WARP_SIZE 32
//#define WARPS_PER_BLOCK 4 
//#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK


//#define NUM_PER_THREAD 8
//#define ROW_THREAD 4                     // 16
//#define COL_THREAD (WARP_SIZE / ROW_THREAD) // 2
//#define COL_NUM_COUNT (COL_THREAD * NUM_PER_THREAD) // 16 half
//#define ROW_NUM_COUNT (ROW_THREAD) // 16 line
//#define BLOCK_ROW (ROW_THREAD * WARPS_PER_BLOCK)
//
//#define K_TILE 128 
//
//#define N_TILE 8 

template<int N_TILE_, int K_TILE_, int NUM_PER_THREAD_, int ROW_THREAD_, int WARPS_PER_BLOCK_>
struct params{

    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_;
    static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

    static constexpr int N_TILE =N_TILE_; 
    static constexpr int K_TILE = K_TILE_;
    static constexpr int NUM_PER_THREAD = NUM_PER_THREAD_;
    static constexpr int ROW_THREAD = ROW_THREAD_;
    static constexpr int COL_THREAD = (WARP_SIZE / ROW_THREAD_) ;
    static constexpr int COL_NUM_COUNT = (COL_THREAD * NUM_PER_THREAD_) ;
    static constexpr int ROW_NUM_COUNT = ROW_THREAD_ ;
    static constexpr int BLOCK_ROW = ROW_THREAD_ * WARPS_PER_BLOCK ;
};

// vectorize B
template<typename T, typename params> \ 
__global__ void mySGemvKernel_Stage4(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C, size_t M, size_t N, size_t K){

    constexpr size_t ROW_THREAD = params::ROW_THREAD;
    constexpr size_t COL_THREAD = params::COL_THREAD;
    constexpr size_t COL_NUM_COUNT = params::COL_NUM_COUNT;
    constexpr size_t N_TILE = params::N_TILE;
    constexpr size_t K_TILE = params::K_TILE;
    constexpr size_t WARP_SIZE = params::WARP_SIZE;
    constexpr size_t WARPS_PER_BLOCK = params::WARPS_PER_BLOCK;
    constexpr size_t NUM_PER_THREAD = params::NUM_PER_THREAD;  


    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_THREAD;
    const size_t warp_row = warpid * ROW_THREAD;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = (K / gridDim.y) / K_TILE;
    constexpr size_t in_K_iters = K_TILE / COL_NUM_COUNT;

    half tmp[params::N_TILE] = {0.0};
    //printf("gridDim.y : %d \n", gridDim.y);
    int4 r_a[in_K_iters];
    int4 r_b[in_K_iters][N_TILE]; 
    size_t B_idx = (tid % COL_THREAD) * NUM_PER_THREAD + blockIdx.z * K + blockIdx.y * (K / gridDim.y); 
    size_t A_idx = (block_row + warp_row + tid_row) * K + (tid % COL_THREAD) * NUM_PER_THREAD + blockIdx.y * (K / gridDim.y); 

    for (size_t i = 0; i < out_K_iters; ++i){
        //#pragma unroll
        for (size_t j = 0; j < (in_K_iters); ++j){
            LDG_L1_128bit_LAST(r_a[j], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            for(size_t n = 0; n < N_TILE; ++n){
                LDG_L1_128bit_ALWAYS(r_b[j][n], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j + K * n])));
            }
        }

        for(size_t j = 0; j < (in_K_iters); ++j){
            for(size_t n = 0; n < N_TILE; ++n){
                //#pragma unroll 
                for(size_t k = 0; k < NUM_PER_THREAD; ++k){
                    tmp[n] += (reinterpret_cast<half*>(&r_a[j]))[k] * (reinterpret_cast<half*>(&r_b[j][n]))[k];
                }
            }
        }
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));

    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        for(size_t n = 0; n < N_TILE; ++n){
            tmp[n] += __shfl_xor_sync(0xFFFFFFFF, tmp[n], i, 32);}
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));
    //if (tid % 2 == 0){
    size_t C_idx = block_row + warp_row + tid_row;
    //#pragma unroll

    //if (N_TILE % 8 == 0){
    //    *(int4*)(&C[C_idx * N + blockIdx.z * 8]) = *(int4*)(&tmp[0]);
    //}else{ 
    if (tid % COL_THREAD == 0){
        for(size_t n = 0; n < N_TILE; ++n)
        {
            //*(int4*)(&C[C_idx * N + blockIdx.z * 8]) = *(int4*)(&tmp[0]);
            atomicAdd(&(C[C_idx * N + n + blockIdx.z * N_TILE]), tmp[n]);
            //C[C_idx * N + n + blockIdx.z * N_TILE] = tmp[n];
        }
    }
}

template<typename T, typename params>
void gemv(T* a, T* b, T* c, int M, int N, int K, int split_k){
    dim3 block(params::THREADS_PER_BLOCK);
    dim3 grid(M / params::BLOCK_ROW, split_k, div_ceil(N, params::N_TILE));
    mySGemvKernel_Stage4<T, params><<<grid, block>>>(a, b, c, M, N, K);
}

template<F16F16GemmTCAlgo_t algo = HGEMMAlignedV5>
void myF16F16GemmTCWarp(half *a, half *b, half *c, int M, int N, int K) {
    if(algo == HGEMMAlignedV5){
        constexpr int k_tile = 128;
        constexpr int row_thread = 4;
        constexpr int num_per_thread = 8;
        constexpr int warps_per_block = 4; 

        int block_num = div_ceil(M, (row_thread * warps_per_block));
        int sms = div_ceil(block_num, 16);
        int split_K = div_ceil(sms, 108) < 4 ? div_ceil(sms, 108) : 4;
        if(N <= 4){
            gemv<half, params<1, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
        }else if(N <= 8){
            if(N % 2 == 0){
                gemv<half, params<2, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }else{
                gemv<half, params<1, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }
        }else{
            if(N % 8 == 0){
                gemv<half, params<8, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }else if(N % 4 == 0){
                gemv<half, params<4, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }else if(N % 2 == 0){
                gemv<half, params<2, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }else{
                gemv<half, params<1, k_tile, num_per_thread, row_thread, warps_per_block>>(a, b, c, M, N, K, split_K);
            }
        }
    }
}
float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    cudaMemset(&d_c, 0, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++){
        h_a[i] = (half)(float(rand()) / float(RAND_MAX));
        //h_a[i] = (half)(float(i) / 100);
    }
    for (int i = 0; i < K * N; i++)
        //h_b[i] = (half)(rand() / float(RAND_MAX));
        h_b[i] = (half)(float(i)/ 10000);

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    for (int i =0; i < N; ++i){
        for(int j = 0; j < K; ++j){
           h_b[i * K + j] = (half)((float)(j * N + i) / 10000);
        }
    }

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            printf("%f , ", float(h_c[i * N + j]));
        }
        printf("\n----------------------\n");
    }
    printf("\n\n\n\n\n\n\n");
    for(int i = 0; i < N; ++i){ 
        for(int j = 0; j < M; ++j){
            printf("%f , ", float(h_d_c[i * N + j]));
        }
        printf("\n======================\n");
    }
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
 
    //warmup 
    for(int i = 0; i < 10;i++){
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

    float msec = 0.f;
    cudaEventElapsedTime(&msec, start, end);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return double(msec / repeat);
}
int main(int arg, char* argv[]){
    printf("\nalgo = HGEMMAlignedV1\n");
    const int test_num = 1;
    //const int test_num = 16;
    //// [1, 4096, 1024],  [1, 2304, 768], [1, 768, 3072]
    //const int M_list[test_num] = {8192, 50, 50, 500, 256,  1024, 768, 1024, 4096, 2304, 768, 3072, 4096, 4096, 11008};
    ////const int N_list[test_num] = {1,   1,   1,   1,    1   , 1,    1,   1,    1,    1    , 1};
    //const int N_list[test_num] = {5,   5,   5,   5,    5   , 5,    5,   5, 5, 5, 5};
    ////const int N_list[test_num] = {8,   8,   8,   8,    8   , 8,    8,   8,    8,    8,   8};
    //const int K_list[test_num] = {14208, 64, 64, 128, 128, 256, 1024, 768, 4096, 1024, 768, 3072, 768, 4096, 11008, 4096};
    const int outer_repeat = 1, inner_repeat = 10;
    //const int total_N = 5;
    //{
    //    //const int M = 1024, N = 1, K = 1024;
    //    //const int M = 16, N = 2, K = 32;
    //    int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
    //    float max_error = testF16F16GemmMaxError(
    //        myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K);
    //    printf("Max Error = %f\n", max_error);
    //}


    printf("----------------cache A for B------------------------\n");
    for (int j = 0; j < test_num; j++){
        //int M = M_list[j], N = total_N, K = K_list[j];
        int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
        //int M = 1024 * j, N = 1, K = 1024 * j;

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance(
                myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

       printf("M N K = %6d %6d %6d, ", M, N, K);
       printf("Time = %12.8lf %12.8lf %12.8lf ms, ", min_sec, avg_sec, max_sec);
       printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
	// printf("%10.4lf\n", avg_Gflops);
    }
    ////printf("-------------------cache A for B double-buff---------------------\n");
    //for (int j = 0; j < test_num; j++){
    //    //int M = M_list[j], N = total_N, K = K_list[j];
    //    int M = std::atoi(argv[2]), N = std::atoi(argv[1]), K = std::atoi(argv[3]);
    //    //int M = 1024 * j, N = 1, K = 1024 * j;

    //    double max_sec = 0.0;
    //    double min_sec = DBL_MAX;
    //    double total_sec = 0.0;

    //    for (int k = 0; k < outer_repeat; k++) {
    //        double this_sec = testF16F16GemmPerformance(
    //            myF16F16GemmTCWarp<HGEMMAlignedV6>, M, N, K, inner_repeat);
    //        max_sec = max(max_sec, this_sec);
    //        min_sec = min(min_sec, this_sec);
    //        total_sec += this_sec;
    //    }

    //    double avg_sec = total_sec / outer_repeat;
    //    double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

    //    printf("M N K = %6d %6d %6d, ", M, N, K);
    //    printf("Time = %12.8lf %12.8lf %12.8lf ms, ", min_sec, avg_sec, max_sec);
    //    printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
    //}
    return 0;
}

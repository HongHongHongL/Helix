#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;

#define CHECK_CUBLAS(Expr) { \
    int err = (Expr); \
    if (err != 0) { \
        printf("cuBLAS error %d at line %d\n", err, __LINE__); \
    } \
}

void gemm(cublasHandle_t handle,
          int m,
          int n,
          int k,
          const void *alpha,
          const void *beta,
          cudaDataType_t input_type,
          const void *A,
          const void *B,
          cudaDataType_t output_type,
          void *C,
#if __CUDACC_VER_MAJOR__ >= 11
          cublasComputeType_t compute_type,
#else
          cudaDataType_t compute_type,
#endif
          int algo) {
    cublasStatus_t res = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        alpha, B, input_type, n, A, input_type, k,
        beta, C, output_type, n, compute_type, static_cast<cublasGemmAlgo_t>(algo));
    CHECK_CUBLAS(res);
}

int main(int arg, char* argv[]) {
    int test_num = 1;
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    if (m < n) {
        std::swap(m, n);
    }
    for(int i = 0; i < test_num; ++i){
        half alpha = __float2half(1.0);
        half beta = __float2half(0.0);

        cudaDataType_t input_type = CUDA_R_16F;
        cudaDataType_t output_type = CUDA_R_16F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
    cudaDataType_t compute_type = CUDA_R_16F;
#endif

        double gopss = 0.f;
        int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        int end_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for(int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo){
            int iter = 10;

            void *A, *B, *C;
            cudaMalloc(&A, m * k * sizeof(half));
            cudaMalloc(&B, k * n * sizeof(half));
            cudaMalloc(&C, m * n * sizeof(half));

            cublasHandle_t handle;
            cublasCreate(&handle);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

             //warmup
            for(int i = 0 ;i < 10; ++i){
                gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
                     output_type, C, compute_type, algo);
            } 
            cudaEventRecord(start);
            for (int i = 0; i < iter; ++i) {
                gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
                     output_type, C, compute_type, algo);
            }
            cudaEventRecord(stop);

            float time_ms = 0.f;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            long ops = (long)m * n * k * 2;
            double gops = ((double)ops / 1e12) / ((double)time_ms / iter / 1e3);
            gopss = gops > gopss ? gops : gopss;

            cudaFree(A);
            cudaFree(B);
            cudaFree(C);
            printf("CBLAS - M : %d, N : %d, K : %d, %f ms, %f Tflops\n", m, n, k, (time_ms/iter), gopss);
        }
    }
}




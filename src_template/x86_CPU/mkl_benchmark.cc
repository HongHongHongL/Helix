#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>

#include <mkl.h>


// M, K, N
int benchmark_dims[][3] = {
#define CASE(...) __VA_ARGS__,
#include "CASES.def"
#undef CASE 
};

#define _CONCAT(x, y) x##y
#define CONCAT(x, y) _CONCAT(x, y)

enum CaseCount {
#define CASE(...) CONCAT(item, __COUNTER__),
#include "CASES.def"
CASE_NUM,
#undef CASE 
};


#define WARMUP_STEPS 20
#define BENCHMARK_STEPS 100
#define ALLIGNMENT 64
#define ALPHA 1.0
#define BETA 0.0


void random_initialize(float *data , size_t size)
{
  std::random_device dev;
  std::mt19937 rng(dev()); 
  std::uniform_int_distribution<std::mt19937::result_type> dist(-1,1);

  for (size_t i = 0; i < size; ++ i) {
    data[i] = dist(rng);
  }
}

int main()
{
  mkl_set_num_threads(48);
  for (int i = 0; i < CASE_NUM; ++ i) {
    const int M = benchmark_dims[i][0];
    const int K = benchmark_dims[i][1];
    const int N = benchmark_dims[i][2];

//    std::cout << "M=" << M << " K=" << K << " N=" << N << ": ";

    float *A = (float *)(aligned_alloc(ALLIGNMENT, M * K * sizeof(float)));
    float *B = (float *)(aligned_alloc(ALLIGNMENT, K * N * sizeof(float)));
    float *C = (float *)(aligned_alloc(ALLIGNMENT, M * N * sizeof(float)));

    random_initialize(A, M * K);
    random_initialize(B, K * N);
    random_initialize(C, M * N);

    // Warmup
    for (size_t i = 0; i < WARMUP_STEPS; ++i) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K/*lda*/, B, N/*ldb*/, BETA, C, N/*ldc*/);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < BENCHMARK_STEPS; ++i) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K/*lda*/, B, N/*ldb*/, BETA, C, N/*ldc*/);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
//    std::cout << M << " " << N << " " << K << " " << duration.count() / BENCHMARK_STEPS << std::endl;
//    std::cout << M << " " << duration.count() * 1000 / BENCHMARK_STEPS << std::endl;
    std::cout << 2ll * M * N * K * 1e-9 / (duration.count() / BENCHMARK_STEPS) << std::endl;
  }
}
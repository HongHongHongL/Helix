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

#define CUDA_KERNEL_LOOP(i, n) \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

const int CAFFE_CUDA_NUM_THREADS = 1024;
inline int CAFFE_GET_BLOCKS(const int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
__global__ void im2col_gpu_kernel(
        const int n, Dtype* data_im,
        const int height, const int width, const int ksize, const int pad,
        const int stride, const int height_col, const int width_col,
        Dtype* data_col,
        const int data_im_size,
        const int data_col_size,
        const int batch_size
        ){
        for(int batch_index = 0; batch_index < batch_size; batch_index++){
                for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x){
                        int w_out = index % width_col; //col
                        int h_index = index / width_col; // col index

                        int h_out = h_index % height_col; // row
                        int channel_in = h_index / height_col; // row index

                        int channel_out = channel_in * ksize * ksize;
                        int h_in = h_out * stride - pad;
                        int w_in = w_out * stride - pad;
                        Dtype* data_col_ptr = data_col;
                        data_col_ptr += batch_index * data_col_size + (channel_out * height_col + h_out) * width_col + w_out;
                        Dtype* data_im_ptr = data_im;
                        data_im_ptr += batch_index * data_im_size + (channel_in * height + h_in) * width + w_in;

                        for (int i = 0; i < ksize; ++i) {
                                for (int j = 0; j < ksize; ++j) {
                                        int h = h_in + i;
                                        int w = w_in + j;
                                        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                                                data_im_ptr[i * width + j]  : half(0);
                                        data_col_ptr += height_col * width_col;
                                }
                        }

                }
        }
}


template <typename Dtype>
void im2col_gpu(Dtype* data_im, int channels,
                                int height, int width, int ksize, int pad,
                                int stride, Dtype* data_col, int batch_size)
{
        int height_col = (height + 2 * pad - ksize) / stride + 1;
        int width_col = (width + 2 * pad - ksize) / stride + 1;
        int num_kernels = channels * height_col * width_col; // col: M

        int data_im_size = height*width*channels;
        int data_col_size = num_kernels*ksize*ksize;
        dim3 grid(CAFFE_GET_BLOCKS(num_kernels), 1, 1);
        dim3 block(CAFFE_CUDA_NUM_THREADS);
        // NOLINT_NEXT_LINE(whitespace/operators)
        im2col_gpu_kernel<Dtype><<<grid, // num_kernels/16, means each thread process 16 elements
                block>>>(
                num_kernels, data_im, height, width, ksize, pad, stride, height_col,
                width_col, data_col, data_im_size, data_col_size, batch_size);
        //CUDA_POST_KERNEL_CHECK;
}


template<F16F16GemmTCAlgo_t algo = HGEMMAlignedV1>
void myF16F16GemmTCWarp(half* dev_image, int channels,
                                int height, int width, int ksize, int pad,
                                int stride, half* dev_col, int batch_size) {

    if(algo == HGEMMAlignedV2){
        im2col_gpu<half>(dev_image, channels, height, width, ksize, pad, stride, dev_col, batch_size);
    }
}

float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (half *, int, int, int, int, int, int, half*, int),
    int batch_size, int channels, int height, int width, int ksize, int num_kernels, int pad, int stride, int repeat) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    
    int K = ksize*ksize*channels; //KH * KW * IC
    int M = num_kernels;         // OC
    int N = height_col*width_col; // 
    
    int image_size = height * width * channels;
    int images_size = image_size * batch_size;
    
    int kernels_size = M * K;
    int col_size = N*K;
    int result_size = M * N * batch_size;
    half *dev_image, *dev_col;
    (cudaMalloc((void**)&dev_image, images_size* sizeof(half)));
    //(cudaMemcpy(dev_image, data_im, images_size * sizeof(float), cudaMemcpyHostToDevice));
    (cudaMalloc((void**)&dev_col, N * K *batch_size * sizeof(half)));

    int Batch_N = N * batch_size;
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(dev_image, channels, height, width, ksize, pad, stride, dev_col, batch_size);
    }
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(dev_image, channels, height, width, ksize, pad, stride, dev_col, batch_size);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;
    cudaFree(dev_image);
    cudaFree(dev_col);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return sec * 1e6;
}

int main(int arg, char* argv[]){
    const int test_num = 5;
    const int outer_repeat = 1, inner_repeat = 10;
    for (int j = 4; j < test_num; j++){
        //int M = M_list[j], N = 8, K = K_list[j];
        //int M = 1024, N = 1024, K = 32;
        //int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
        int batch = std::atoi(argv[1]);
        int IC = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        int W = std::atoi(argv[4]);
        int OC = std::atoi(argv[5]);
        int KH = std::atoi(argv[6]);
        int KW = std::atoi(argv[7]);
    int stride = std::atoi(argv[8]);
        int pad = std::atoi(argv[9]);
    

        //int batch = 1, IC = 3, H = 224, W = 224, OC = 64, KH = 3;
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;
        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance(
                myF16F16GemmTCWarp<HGEMMAlignedV2>, batch, IC, H, W, KH, OC, pad, stride, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;

        //printf("M N K = %6d %6d %6d, ", M, N, K);
        //printf("Time = %12.8lf %12.8lf %12.8lf us \n", min_sec, avg_sec, max_sec);
        //printf("%f\n", min_sec, avg_sec, max_sec);
        printf("%f\n", avg_sec / 1e9);
        //printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        //if(avg_Gflops < 300) printf("%f\n", avg_Gflops);
        //else printf("0\n");
    }
    return 0;
}
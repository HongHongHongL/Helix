#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#include <cuda_fp16.h>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <numeric>
#include <functional>



//#define MY_DEBUG
//#define MY_PRINT

using namespace std;

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

void print(const float *data, int n, int c, int h, int w) {
    std::vector<float> buffer(1 << 20);
    CUDA_CALL(cudaMemcpy(
          buffer.data(), data,
          n * c * h * w * sizeof(float),
          cudaMemcpyDeviceToHost));
    int a = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < c; ++j) {
        std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
        for (int k = 0; k < h; ++k) {
          for (int l = 0; l < w; ++l) {
            // std::cout << std::setw(4) << std::right << buffer[a];
            std::cout << float(buffer[a]) << " ";
            ++a;
          }
          std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl;
  }

int main(int argc, char *argv[]) {

  const int inputParamNum = 12;
  int in_n, in_h, in_w, in_c, filt_k, filt_h, filt_w, filt_c, is_nchw, str_h, str_w, pad_h, pad_w;

  string log_file;
  ofstream log;
  bool log_bool = true;
  
  if(argc == inputParamNum){
    //input
    in_n = std::stoi(argv[1]);
    in_c = std::stoi(argv[2]);
    in_h = std::stoi(argv[3]);
    in_w = std::stoi(argv[4]);

    //filter
    filt_k = std::stoi(argv[5]);
    filt_h = std::stoi(argv[6]);
    filt_w = std::stoi(argv[7]);
    filt_c = in_c;
    str_h = std::stoi(argv[8]);
    str_w = str_h;
    pad_h = std::stoi(argv[9]);
    pad_w = pad_h;

    log_file = string(argv[10]);
    //  log(log_file, ios::out); 
     log.open(log_file);
    is_nchw = std::stoi(argv[11]);
    
    //stride
    // params.stride_width = std::stoi(argv[8]);
    // params.stride_height = std::stoi(argv[9]);
  }
  else if(argc == inputParamNum - 1) {
    in_n = std::stoi(argv[1]);
    in_c = std::stoi(argv[2]);
    in_h = std::stoi(argv[3]);
    in_w = std::stoi(argv[4]);

    //filter
    filt_k = std::stoi(argv[5]);
    filt_h = std::stoi(argv[6]);
    filt_w = std::stoi(argv[7]);
    str_h = std::stoi(argv[8]);
    str_w = str_h;
    pad_h = std::stoi(argv[9]);
    pad_w = pad_h;
    is_nchw = std::stoi(argv[10]);
    filt_c = in_c;
    log_bool = false;
  }
  else {
      printf("The number of parameters should be %d instead of %d\n", inputParamNum, argc);
      printf("./a.out n h w cin cout fh fw log_file\n");
      exit(0);
  }

  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

#ifdef MY_DEBUG
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;
#endif

  // // convolution
  // int pad_h = filt_h / 2;
  // int pad_w = filt_w / 2;
  // const int str_h = 1;
  // const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;

  #ifdef MY_DEBUG
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;
#endif

  vector<int> in_dims = {in_n, in_c, in_h, in_w};
  vector<int> w_dims = {filt_k, filt_c, filt_h, filt_w};
  cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;

  if(is_nchw == 0){
      tensor_format = CUDNN_TENSOR_NHWC;
  }

  int ni,ci,hi,wi; 
  if(tensor_format == CUDNN_TENSOR_NCHW){
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
  }else{
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
  }

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, tensor_format, CUDNN_DATA_FLOAT,
        static_cast<int>(in_dims[ni]), static_cast<int>(in_dims[ci]), static_cast<int>(in_dims[hi]), static_cast<int>(in_dims[wi])));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, tensor_format,
        static_cast<int>(w_dims[ni]), static_cast<int>(w_dims[ci]), static_cast<int>(w_dims[hi]), static_cast<int>(w_dims[wi])));
  cudnnConvolutionDescriptor_t conv_desc;

  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  //if(tensor_format == CUDNN_TENSOR_NCHW){
  //    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
  //      conv_desc, in_desc, filt_desc,
  //      &out_n, &out_c, &out_h, &out_w));
  //}
  //}else{
  out_n = in_dims[0];
  out_c = w_dims[0];
  out_h = 1 + (in_dims[2] + 2 * pad_h - (((w_dims[2] - 1) * dil_h) + 1)) / str_h;
  out_w = 1 + (in_dims[3] + 2 * pad_w - (((w_dims[3] - 1) * dil_w) + 1)) / str_w;
  //}
  vector<int> out_dims = {out_n, out_c, out_h, out_w};
#ifdef MY_DEBUG
  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;
#endif

  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH));

  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
        // CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, tensor_format, CUDNN_DATA_FLOAT,
        out_dims[ni], out_dims[ci], out_dims[hi], out_dims[wi]));


  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  //int returned_algo_count = 0;
  //cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  //CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
  //    cudnn, in_desc, filt_desc,
  //    conv_desc, out_desc,
  //    CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count, perf_results));
  //algo = perf_results[0].algo;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));


  void *ws_data;
  if (ws_size > 0) {
    cudaMalloc(&ws_data, ws_size);
  }

  #ifdef MY_DEBUG
  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;
  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;
  #endif

  float *in_data, *filt_data, *out_data;

  CUDA_CALL(cudaMalloc(
         &in_data, in_n * in_c * in_h * in_w * sizeof(float)));
        //&in_data, std::accumulate(in_dims.begin(), in_dims.end(), 1, multiplies<int>()) * sizeof(float)));
  CUDA_CALL(cudaMalloc(
         &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
      //&filt_data, std::accumulate(w_dims.begin(), w_dims.end(), 1, multiplies<int>()) * sizeof(float)));
  CUDA_CALL(cudaMalloc(
        // &out_data, out_n * out_c * out_h * out_w * sizeof(float)));
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  int output_tensor_size = out_n * out_c * out_h * out_w;
  
  int64_t fmas = output_tensor_size * int64_t(filt_h * filt_w * filt_c);

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  //dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  //dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 0.01);

  float runtime_ms = 0;
  int iter_num = 100;
  for (int iteration = 0; iteration < iter_num ; ++iteration) {
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0]);

  for (int iteration = 0; iteration < iter_num ; ++iteration) {
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));
  }

  cudaEventRecord(events[1]);
  cudaEventSynchronize(events[1]);
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  runtime_ms = double(runtime_ms) / double(iter_num) * 1000;
  double gflops = 2.0 * double(fmas) / double(1.0e9) / (runtime_ms / 1000);

  if(log_bool) {
    log << runtime_ms << std::endl;
    log << gflops << std::endl;
    log << ws_size << std::endl;
    log.close();
  }
  else {
    // printf("%f\n", runtime_ms/1000000);
    printf("%f\n", gflops);
    //std::cout << "Workspace size: " << ws_size << std::endl;
  }
  // printf("time:%f\n", runtime_ms);
  // printf("gflops:%f\n", gflops);
  // results
#ifdef MY_PRINT
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);
#endif

  // finalizing
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  //CUDA_CALL(cudaFree(in_data));
  //CUDA_CALL(cudaFree(filt_data));
  //CUDA_CALL(cudaFree(ws_data));
  //CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroy(cudnn));

  
  return 0;
}


import argparse

from Ampere_FP16.build_Ampere_FP16 import build_Helix_Ampere_FP16_gemm_kernel, build_cublas_FP16_gemm_kernel, build_cudnn_FP16_conv_kernel
from Ampere_FP32.build_Ampere_FP32 import build_Helix_Ampere_FP32_gemm_kernel, build_cublas_FP32_gemm_kernel, build_cudnn_FP32_conv_kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get runtime cost of an operation")
    parser.add_argument("-b", "--backend", type=str, choices=['Ampere_FP16', 'Ampere_FP32'], default='Ampere_FP16', help="backend")
    args = parser.parse_args()

    if args.backend == 'Ampere_FP16':
        build_Helix_Ampere_FP16_gemm_kernel()
        build_cublas_FP16_gemm_kernel()
        build_cudnn_FP16_conv_kernel()
    elif args.backend == 'Ampere_FP32':
        build_Helix_Ampere_FP32_gemm_kernel()
        build_cublas_FP32_gemm_kernel()
        build_cudnn_FP32_conv_kernel()
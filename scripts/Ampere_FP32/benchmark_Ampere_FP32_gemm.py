import math
import os

from utils.get_benchmark_shape_list import get_gemm_op_MNKList
from Ampere_FP32_gemm import get_Ampere_FP32_gemm_Helix_result, get_Ampere_FP32_gemm_cublas_result

root_path = os.getcwd()

def Ampere_FP32_Helix_op_level_GEMM_benchmark(backend):
    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = get_gemm_op_MNKList()
    for M, N, K in MNKList:
        _, helix_tflops = get_Ampere_FP32_gemm_Helix_result(prof_dict, M, N, K, backend)
        print(f'{M}x{N}x{K}: Helix: {helix_tflops:.2f} TFLOPS')

def Ampere_FP32_cublas_op_level_GEMM_benchmark():
    MNKList = get_gemm_op_MNKList()
    for M, N, K in MNKList:
        _, cublas_tflops = get_Ampere_FP32_gemm_cublas_result(M, N, K)
        print(f'{M}x{N}x{K}: cublas: {cublas_tflops:.2f} TFLOPS')

if __name__ == "__main__":

    Ampere_FP32_Helix_op_level_GEMM_benchmark(backend="cuda")
    Ampere_FP32_Helix_op_level_GEMM_benchmark(backend="tvm")
    Ampere_FP32_cublas_op_level_GEMM_benchmark()
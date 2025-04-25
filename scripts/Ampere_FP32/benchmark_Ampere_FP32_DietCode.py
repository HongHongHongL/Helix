import math
import os

from utils.get_benchmark_shape_list import get_gemm_op_MNKList

root_path = os.getcwd()

def cost_model(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        num_block = math.ceil(M / k[0]) * math.ceil(N / k[1])
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def get_Helix_result(prof_dict, M, N, K):
    best_config = cost_model(prof_dict, M, N, K)
    cmd = f'{root_path}/build/bin_fp32/sgemm_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]}_{best_config[5]}_{best_config[6]}_{best_config[7]} {math.ceil(M / best_config[0]) * best_config[0]} {math.ceil(N / best_config[1]) * best_config[1]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

def get_cublas_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp32/cublas_f32 {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])
    tflops = 2 * M * N * K / 1e9 / cost

    return tflops

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = get_gemm_op_MNKList()
    for M, N, K in MNKList:
        helix_tflops = get_Helix_result(prof_dict, M, N, K)
        cublas_tflops = get_cublas_result(M, N, K)
        print(f'{M}x{N}x{K}: Helix: {helix_tflops:.2f} TFLOPS, cublas: {cublas_tflops:.2f} TFLOPS')
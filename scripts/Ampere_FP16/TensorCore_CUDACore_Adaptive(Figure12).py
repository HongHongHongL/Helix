import math
import os

from utils.get_benchmark_shape_list import get_gemm_op_MNKList

root_path = os.getcwd()

def cost_model(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        if k[5] == 1:
            continue
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2])
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def cost_model_without_gemv(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        if k[5] == 1 or k[0] == 0:
            continue
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2])
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def get_Helix_result(prof_dict, M, N, K):
    best_config = cost_model(prof_dict, M, N, K)
    if best_config[0] == 0:
        cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
    else:
        cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

def get_gemv_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp16/gemv {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

def get_gemm_result(prof_dict, M, N, K):
    best_config = cost_model_without_gemv(prof_dict, M, N, K)
    cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = [(m, n, 1024) for n in [1024, 2048, 4096] for m in [1, 2, 4, 6, 8, 10, 12, 14, 16]]
    for M, N, K in MNKList:
        helix_tflops = get_Helix_result(prof_dict, M, N, K)
        gemv_tflops = get_gemv_result(M, N, K)
        gemm_tflops = get_gemm_result(prof_dict, M, N, K)
        print(f'{M}x{N}x{K}: Helix {helix_tflops:.2f} TFLOPS, Gemv {gemv_tflops:.2f} TFLOPS, GEMM {gemm_tflops:.2f} TFLOPS')        
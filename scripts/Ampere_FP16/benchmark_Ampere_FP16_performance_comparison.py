import math
import os

from utils.get_benchmark_shape_list import get_gemm_op_MNKList

root_path = os.getcwd()

def cost_model(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2]) * (k[5] + 1)
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def cost_model_static1(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        if k[1] != k[3] or k[2] != k[4]:
            continue
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2]) * (k[5] + 1)
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def cost_model_static2(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        cost = v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def get_Helix_Default_result(prof_dict, M, N, K):
    best_config = cost_model(prof_dict, M, N, K)
    if best_config[0] == 0:
        cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
    else:
        cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

def get_Helix_Oracle_result(prof_dict, M, N, K):
    min_cost = float("inf")
    for k, _ in prof_dict.items():
        if k[0] == 0:
            cmd = f'{root_path}/build/bin_fp16/{"gemv" if k[5] == 0 else "gemv_splitK"} {M} {N} {K}'
        else:
            cmd = f'{root_path}/build/bin_fp16/{"gemm" if k[5] == 0 else "gemm_splitK"}_{k[0]}_{k[1]}_{k[2]}_{k[3]}_{k[4]} {math.ceil(M / k[1]) * k[1]} {math.ceil(N / k[2]) * k[2]} {K}'
        result = os.popen(cmd)
        cost = float(result.read().split()[-8])
        if min_cost > cost:
            min_cost = cost
    tflops = 2 * M * N * K / 1e12 / min_cost

    return tflops

def get_Helix_Static1_result(prof_dict, M, N, K):
    best_config = cost_model_static1(prof_dict, M, N, K)
    if best_config[0] == 0:
        cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
    else:
        cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

def get_Helix_Static2_result(prof_dict, M, N, K):
    best_config = cost_model_static2(prof_dict, M, N, K)
    if best_config[0] == 0:
        cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
    else:
        cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return tflops

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = get_gemm_op_MNKList()
    for M, N, K in MNKList:
        helix_oracle_flops = get_Helix_Oracle_result(prof_dict, M, N, K)
        helix_default_flops = get_Helix_Default_result(prof_dict, M, N, K)
        helix_static1_flops = get_Helix_Static1_result(prof_dict, M, N, K)
        helix_static2_flops = get_Helix_Static2_result(prof_dict, M, N, K)
        print(f'{M}x{N}x{K}: Helix_Oracle: {helix_oracle_flops:.2f} TFlops, Helix_Default: {helix_default_flops:.2f} TFlops, Helix_Static1: {helix_static1_flops:.2f} TFlops, Helix_Static2: {helix_static2_flops:.2f} TFlops')
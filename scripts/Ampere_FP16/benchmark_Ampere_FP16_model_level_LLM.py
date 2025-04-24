import math
import os

from utils.get_benchmark_shape_list import get_llm_opset_MNKList

root_path = os.getcwd()

def cost_model(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2])
        if k[5] == 1:
            num_block *= 2
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

def get_cublas_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp16/cublas_f16 {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])

    return cost

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    helix_Bert_cost, cublas_Bert_cost = 0, 0
    for M, N, K in Bert_MNKList:
        helix_Bert_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_Bert_cost += get_cublas_result(M, N, K)
    print(f'Bert: Helix: {helix_Bert_cost:.2f} ms, cublas: {cublas_Bert_cost:.2f} ms')

    helix_LLAMA2_cost, cublas_LLAMA2_cost = 0, 0
    for M, N, K in LLAMA2_MNKList:
        helix_LLAMA2_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_LLAMA2_cost += get_cublas_result(M, N, K)
    print(f'LLAMA2: Helix: {helix_LLAMA2_cost:.2f} ms, cublas: {cublas_LLAMA2_cost:.2f} ms')

    helix_GPT2_cost, cublas_GPT2_cost = 0, 0
    for M, N, K in GPT2_MNKList:
        helix_GPT2_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_GPT2_cost += get_cublas_result(M, N, K)
    print(f'GPT2: Helix: {helix_GPT2_cost:.2f} ms, cublas: {cublas_GPT2_cost:.2f} ms')
import math
import os

from utils.get_benchmark_shape_list import get_llm_opset_MNKList

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

    return cost

def get_cublas_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp32/cublas_f32 {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])

    return cost

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    helix_Bert_cost, onnxruntime_Bert_cost, mkl_Bert_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        helix_Bert_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_Bert_cost += get_cublas_result(M, N, K)
    print(f'Bert: Helix: {helix_Bert_cost:.2f} ms, cublas: {cublas_Bert_cost:.2f} ms')

    helix_LLAMA2_cost, onnxruntime_LLAMA2_cost, mkl_LLAMA2_cost = 0, 0, 0
    for M, N, K in LLAMA2_MNKList:
        helix_LLAMA2_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_LLAMA2_cost += get_cublas_result(M, N, K)
    print(f'LLAMA2: Helix: {helix_LLAMA2_cost:.2f} ms, cublas: {cublas_LLAMA2_cost:.2f} ms')

    helix_GPT2_cost, onnxruntime_GPT2_cost, mkl_GPT2_cost = 0, 0, 0
    for M, N, K in GPT2_MNKList:
        helix_GPT2_cost += get_Helix_result(prof_dict, M, N, K)
        cublas_GPT2_cost += get_cublas_result(M, N, K)
    print(f'GPT2: Helix: {helix_GPT2_cost:.2f} ms, cublas: {cublas_GPT2_cost:.2f} ms')
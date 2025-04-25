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

def Ampere_FP32_Helix_model_level_LLM_benchmark():
    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    helix_Bert_cost, helix_LLAMA2_cost, helix_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        helix_Bert_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in LLAMA2_MNKList:
        helix_LLAMA2_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in GPT2_MNKList:
        helix_GPT2_cost += get_Helix_result(prof_dict, M, N, K)

    print(f'Bert: {helix_Bert_cost:.2f} ms, LLAMA2: {helix_LLAMA2_cost:.2f} ms, GPT2: {helix_GPT2_cost:.2f} ms')

def Ampere_FP32_cublas_model_level_LLM_benchmark():
    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    cublas_Bert_cost, cublas_LLAMA2_cost, cublas_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        cublas_Bert_cost += get_cublas_result(M, N, K)
    for M, N, K in LLAMA2_MNKList:
        cublas_LLAMA2_cost += get_cublas_result(M, N, K)
    for M, N, K in GPT2_MNKList:
        cublas_GPT2_cost += get_cublas_result(M, N, K)

    print(f'Bert: {cublas_Bert_cost:.2f} ms, LLAMA2: {cublas_LLAMA2_cost:.2f} ms, GPT2: {cublas_GPT2_cost:.2f} ms')

if __name__ == "__main__":

    Ampere_FP32_Helix_model_level_LLM_benchmark()
    Ampere_FP32_cublas_model_level_LLM_benchmark()
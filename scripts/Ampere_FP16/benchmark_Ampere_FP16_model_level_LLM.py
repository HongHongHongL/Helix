import math
import os

from utils.get_benchmark_shape_list import get_llm_opset_MNKList
from Ampere_FP16.Ampere_FP16_gemm import get_Ampere_FP16_gemm_Helix_result, get_Ampere_FP16_gemm_cublas_result

root_path = os.getcwd()

def Ampere_FP16_Helix_model_level_LLM_benchmark(backend = "cuda"):
    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    helix_Bert_cost, helix_LLAMA2_cost, helix_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        cost, _ = get_Ampere_FP16_gemm_Helix_result(prof_dict, M, N, K, backend)
        helix_Bert_cost += cost
    for M, N, K in LLAMA2_MNKList:
        cost, _ = get_Ampere_FP16_gemm_Helix_result(prof_dict, M, N, K, backend)
        helix_LLAMA2_cost += cost
    for M, N, K in GPT2_MNKList:
        cost, _ = get_Ampere_FP16_gemm_Helix_result(prof_dict, M, N, K, backend)
        helix_GPT2_cost += cost

    print(f'Bert: {helix_Bert_cost:.2f} ms, LLAMA2: {helix_LLAMA2_cost:.2f} ms, GPT2: {helix_GPT2_cost:.2f} ms')

def Ampere_FP16_cublas_model_level_LLM_benchmark():
    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    cublas_Bert_cost, cublas_LLAMA2_cost, cublas_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        cost, _ = get_Ampere_FP16_gemm_cublas_result(M, N, K)
        cublas_Bert_cost += cost
    for M, N, K in LLAMA2_MNKList:
        cost, _ = get_Ampere_FP16_gemm_cublas_result(M, N, K)
        cublas_LLAMA2_cost += cost
    for M, N, K in GPT2_MNKList:
        cost, _ = get_Ampere_FP16_gemm_cublas_result(M, N, K)
        cublas_GPT2_cost += cost

    print(f'Bert: {cublas_Bert_cost:.2f} ms, LLAMA2: {cublas_LLAMA2_cost:.2f} ms, GPT2: {cublas_GPT2_cost:.2f} ms')

if __name__ == "__main__":

    Ampere_FP16_Helix_model_level_LLM_benchmark(backend="cuda")
    Ampere_FP16_Helix_model_level_LLM_benchmark(backend="tvm")
    Ampere_FP16_cublas_model_level_LLM_benchmark()
import math
import os

from utils.get_benchmark_shape_list import get_cnn_opset_MNKList

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

def get_Helix_result(prof_dict, M, N, K):
    best_config = cost_model(prof_dict, M, N, K)
    if best_config[0] == 0:
        cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
    else:
        cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])
    tflops = 2 * M * N * K / 1e12 / cost

    return cost

def get_cublas_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp16/cublas_f16 {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])

    return cost

def Ampere_FP16_Helix_model_level_CNN_benchmark():
    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    AlexNet_MNKList, ResNet_MNKList, GoogleNet_MNKList = get_cnn_opset_MNKList()

    helix_AlexNet_cost, helix_ResNet_cost, helix_GoogleNet_cost = 0, 0, 0
    for M, N, K in AlexNet_MNKList:
        helix_AlexNet_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in ResNet_MNKList:
        helix_ResNet_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in GoogleNet_MNKList:
        helix_GoogleNet_cost += get_Helix_result(prof_dict, M, N, K)

    print(f'AlexNet: {helix_AlexNet_cost:.2f} ms, ResNet: {helix_ResNet_cost:.2f} ms, GoogleNet: {helix_GoogleNet_cost:.2f} ms')

def Ampere_FP16_cublas_model_level_CNN_benchmark():
    AlexNet_MNKList, ResNet_MNKList, GoogleNet_MNKList = get_cnn_opset_MNKList()

    cublas_AlexNet_cost, cublas_ResNet_cost, cublas_GoogleNet_cost = 0, 0, 0
    for M, N, K in AlexNet_MNKList:
        cublas_AlexNet_cost += get_cublas_result(M, N, K)
    for M, N, K in ResNet_MNKList:
        cublas_ResNet_cost += get_cublas_result(M, N, K)
    for M, N, K in GoogleNet_MNKList:
        cublas_GoogleNet_cost += get_cublas_result(M, N, K)

    print(f'AlexNet: {cublas_AlexNet_cost:.2f} ms, ResNet: {cublas_ResNet_cost:.2f} ms, GoogleNet: {cublas_GoogleNet_cost:.2f} ms')

if __name__ == "__main__":

    Ampere_FP16_Helix_model_level_CNN_benchmark()
    Ampere_FP16_cublas_model_level_CNN_benchmark()
import time
import math
import os

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

def get_Helix_runtime_cost(prof_dict, M, N, K):
    T1 = time.perf_counter()
    for i in range(100):
        best_config = cost_model(prof_dict, M, N, K)
    T2 = time.perf_counter()
    cmd = f'{root_path}/build/bin_fp32/sgemm_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]}_{best_config[5]}_{best_config[6]}_{best_config[7]} {math.ceil(M / best_config[0]) * best_config[0]} {math.ceil(N / best_config[1]) * best_config[1]} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-8])

    return (T2 - T1) / 100 / 1000, cost

def Ampere_FP32_runtime_cost():

    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    for M, N, K in MNKList:
        runtime_cost, op_cost = get_Helix_runtime_cost(prof_dict, M, N, K)
        print(f'The runtime cost of {M}x{N}x{K} is {runtime_cost} ms, and the op cost is {op_cost} ms.')

if __name__ == '__main__':

    Ampere_FP32_runtime_cost()
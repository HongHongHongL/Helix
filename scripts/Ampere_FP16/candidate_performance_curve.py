import math
import os

root_path = os.getcwd()

def get_all_candidate_performance(prof_dict, M, N, K):
    min_cost = float("inf")
    min_config = (1, 1, 1, 1, 1)
    id, min_id = 0, 0
    for k, v in prof_dict.items():
        id += 1
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2])
        if k[5] == 1:
            num_block *= 2
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            min_config = k
            min_id = id
    print(f'Best candidate id: {min_id}, config = {min_config}')
    id = 0
    for k, v in prof_dict.items():
        if k[0] == 0:
            cmd = f'{root_path}/build/bin_fp16/{"gemv" if k[5] == 0 else "gemv_splitK"} {M} {N} {K}'
        else:
            cmd = f'{root_path}/build/bin_fp16/{"gemm" if k[5] == 0 else "gemm_splitK"}_{k[0]}_{k[1]}_{k[2]}_{k[3]}_{k[4]} {math.ceil(M / k[1]) * k[1]} {math.ceil(N / k[2]) * k[2]} {K}'
        result = os.popen(cmd)
        cost = float(result.read().split()[-8])
        tflops = 2 * M * N * K / 1e12 / cost
        print(f'candidate id: {min_id}, flops = {tflops} Tflops')

if __name__ == "__main__":

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    MNKList = [(476, 1024, 1024), (128, 512, 1000), (5124, 700, 2048), (32, 50515, 100)]
    for M, N, K in MNKList:
        get_all_candidate_performance(prof_dict, M, N, K)
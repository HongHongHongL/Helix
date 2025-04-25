import math
import os

from utils.get_benchmark_shape_list import get_conv_op_shape_list

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

    return tflops

def get_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad):
    cmd = f'{root_path}/build/bin_fp16/cudnn_fp16 {batch} {input_channel} {H} {W} {output_channel} {kH} {kW} {stride} {pad} 1'
    result = os.popen(cmd)
    tflops = float(result.read().splitlines()[0])

    return tflops

def Ampere_FP16_Helix_op_level_Conv_benchmark():
    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    shape_list = get_conv_op_shape_list()
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in shape_list:
        M = batch * ((H + 2 * pad - kH) // stride + 1) * ((W + 2 * pad - kW) // stride + 1)
        N = output_channel
        K = input_channel * kH * kW
        helix_tflops = get_Helix_result(prof_dict, M, N, K)
        print(f'Helix: {helix_tflops:.2f} TFLOPS')

def Ampere_FP16_cudnn_op_level_Conv_benchmark():
    shape_list = get_conv_op_shape_list()
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in shape_list:
        cudnn_tflops = get_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
        print(f'cudnn: {cudnn_tflops:.2f} TFLOPS')

if __name__ == "__main__":

    Ampere_FP16_Helix_op_level_Conv_benchmark()
    Ampere_FP16_cudnn_op_level_Conv_benchmark()
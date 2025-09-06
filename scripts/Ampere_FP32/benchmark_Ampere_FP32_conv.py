import math
import os

from utils.get_benchmark_shape_list import get_conv_op_shape_list
from Ampere_FP32.Ampere_FP32_conv import get_Ampere_FP32_conv_Helix_result, get_Ampere_FP32_conv_cudnn_result

root_path = os.getcwd()

def Ampere_FP32_Helix_op_level_Conv_benchmark(backend = "cuda"):
    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    shape_list = get_conv_op_shape_list()
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in shape_list:
        _, helix_tflops = get_Ampere_FP32_conv_Helix_result(prof_dict, batch, input_channel, H, W, output_channel, kH, kW, stride, pad, backend)
        print(f'Helix: {helix_tflops:.2f} TFLOPS')

def Ampere_FP32_cudnn_op_level_Conv_benchmark():
    shape_list = get_conv_op_shape_list()
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in shape_list:
        _, cudnn_tflops = get_Ampere_FP32_conv_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
        print(f'cudnn: {cudnn_tflops:.2f} TFLOPS')

if __name__ == "__main__":

    Ampere_FP32_Helix_op_level_Conv_benchmark(backend="cuda")
    Ampere_FP32_Helix_op_level_Conv_benchmark(backend="tvm")
    Ampere_FP32_cudnn_op_level_Conv_benchmark()
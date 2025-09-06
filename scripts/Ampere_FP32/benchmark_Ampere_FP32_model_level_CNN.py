import math
import os

from utils.get_benchmark_shape_list import get_cnn_opset_shape_List
from Ampere_FP32.Ampere_FP32_conv import get_Ampere_FP32_conv_Helix_result, get_Ampere_FP32_conv_cudnn_result

root_path = os.getcwd()

def Ampere_FP32_Helix_model_level_CNN_benchmark(backend = "cuda"):
    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    AlexNet_shape_list, ResNet_shape_list, GoogleNet_shape_list = get_cnn_opset_shape_List()

    helix_AlexNet_cost, helix_ResNet_cost, helix_GoogleNet_cost = 0, 0, 0
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in AlexNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_Helix_result(prof_dict, batch, input_channel, H, W, output_channel, kH, kW, stride, pad, backend)
        helix_AlexNet_cost += cost
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in ResNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_Helix_result(prof_dict, batch, input_channel, H, W, output_channel, kH, kW, stride, pad, backend)
        helix_ResNet_cost += cost
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in GoogleNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_Helix_result(prof_dict, batch, input_channel, H, W, output_channel, kH, kW, stride, pad, backend)
        helix_GoogleNet_cost += cost

def Ampere_FP32_cudnn_model_level_CNN_benchmark():
    AlexNet_shape_list, ResNet_shape_list, GoogleNet_shape_list = get_cnn_opset_shape_List()

    cublas_AlexNet_cost, cublas_ResNet_cost, cublas_GoogleNet_cost = 0, 0, 0
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in AlexNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
        cublas_AlexNet_cost += cost
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in ResNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
        cublas_ResNet_cost += cost
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in GoogleNet_shape_list:
        cost, _ = get_Ampere_FP32_conv_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
        cublas_GoogleNet_cost += cost

    print(f'AlexNet: {cublas_AlexNet_cost:.2f} ms, ResNet: {cublas_ResNet_cost:.2f} ms, GoogleNet: {cublas_GoogleNet_cost:.2f} ms')

if __name__ == "__main__":

    Ampere_FP32_Helix_model_level_CNN_benchmark(backend="cuda")
    Ampere_FP32_Helix_model_level_CNN_benchmark(backend="tvm")
    Ampere_FP32_cudnn_model_level_CNN_benchmark()
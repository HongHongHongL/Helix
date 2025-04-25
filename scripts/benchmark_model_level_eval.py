import argparse

from Ampere_FP16.benchmark_Ampere_FP16_model_level_LLM import Ampere_FP16_Helix_model_level_LLM_benchmark, Ampere_FP16_cublas_model_level_LLM_benchmark
from Ampere_FP16.benchmark_Ampere_FP16_model_level_CNN import Ampere_FP16_Helix_model_level_CNN_benchmark, Ampere_FP16_cublas_model_level_CNN_benchmark
from Ampere_FP32.benchmark_Ampere_FP32_model_level_LLM import Ampere_FP32_Helix_model_level_LLM_benchmark, Ampere_FP32_cublas_model_level_LLM_benchmark
from Ampere_FP32.benchmark_Ampere_FP32_model_level_CNN import Ampere_FP32_Helix_model_level_CNN_benchmark, Ampere_FP32_cublas_model_level_CNN_benchmark
from x86_CPU.benchmark_x86_CPU_model_level_LLM import x86_CPU_Helix_model_level_LLM_benchmark, x86_CPU_onnxruntime_model_level_LLM_benchmark, x86_CPU_MKL_model_level_LLM_benchmark
from x86_CPU.benchmark_x86_CPU_model_level_CNN import x86_CPU_Helix_model_level_CNN_benchmark, x86_CPU_onnxruntime_model_level_CNN_benchmark, x86_CPU_MKL_model_level_CNN_benchmark
from ARM_CPU.benchmark_ARM_CPU_model_level_LLM import ARM_CPU_Helix_model_level_LLM_benchmark, ARM_CPU_onnxruntime_model_level_LLM_benchmark, ARM_CPU_ACL_model_level_LLM_benchmark
from ARM_CPU.benchmark_ARM_CPU_model_level_CNN import ARM_CPU_Helix_model_level_CNN_benchmark, ARM_CPU_onnxruntime_model_level_CNN_benchmark, ARM_CPU_ACL_model_level_CNN_benchmark

func_dict = {
    ('LLM', 'Ampere_FP16', 'Helix'): Ampere_FP16_Helix_model_level_LLM_benchmark,
    ('LLM', 'Ampere_FP16', 'cublas'): Ampere_FP16_cublas_model_level_LLM_benchmark,
    ('CNN', 'Ampere_FP16', 'Helix'): Ampere_FP16_Helix_model_level_CNN_benchmark,
    ('CNN', 'Ampere_FP16', 'cublas'): Ampere_FP16_cublas_model_level_CNN_benchmark,
    ('LLM', 'Ampere_FP32', 'Helix'): Ampere_FP32_Helix_model_level_LLM_benchmark,
    ('LLM', 'Ampere_FP32', 'cublas'): Ampere_FP32_cublas_model_level_LLM_benchmark,
    ('CNN', 'Ampere_FP32', 'Helix'): Ampere_FP32_Helix_model_level_CNN_benchmark,
    ('CNN', 'Ampere_FP32', 'cublas'): Ampere_FP32_cublas_model_level_CNN_benchmark,
    ('LLM', 'x86_CPU', 'Helix'): x86_CPU_Helix_model_level_LLM_benchmark,
    ('LLM', 'x86_CPU', 'onnxruntime'): x86_CPU_onnxruntime_model_level_LLM_benchmark,
    ('LLM', 'x86_CPU', 'MKL'): x86_CPU_MKL_model_level_LLM_benchmark,
    ('CNN', 'x86_CPU', 'Helix'): x86_CPU_Helix_model_level_CNN_benchmark,
    ('CNN', 'x86_CPU', 'onnxruntime'): x86_CPU_onnxruntime_model_level_CNN_benchmark,
    ('CNN', 'x86_CPU', 'MKL'): x86_CPU_MKL_model_level_CNN_benchmark,
    ('LLM', 'ARM_CPU', 'Helix'): ARM_CPU_Helix_model_level_LLM_benchmark,
    ('LLM', 'ARM_CPU', 'onnxruntime'): ARM_CPU_onnxruntime_model_level_LLM_benchmark,
    ('LLM', 'ARM_CPU', 'ACL'): ARM_CPU_ACL_model_level_LLM_benchmark,
    ('CNN', 'ARM_CPU', 'Helix'): ARM_CPU_Helix_model_level_CNN_benchmark,
    ('CNN', 'ARM_CPU', 'onnxruntime'): ARM_CPU_onnxruntime_model_level_CNN_benchmark,
    ('CNN', 'ARM_CPU', 'ACL'): ARM_CPU_ACL_model_level_CNN_benchmark
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Model level performance")
    parser.add_argument("-m", "--model", type=str, choices=['LLM', 'CNN'], default='LLM', help="model name")
    parser.add_argument("-b", "--backend", type=str, choices=['Ampere_FP16', 'Ampere_FP32', 'x86_CPU', 'ARM_CPU'], default='Ampere_FP16', help="backend")
    parser.add_argument("-s", "--system", type=str, choices=['Helix', 'cublas', 'cudnn', 'cutlass', 'DietCode', 'onnxruntime', 'MKL', 'acl'], default='Helix', help="system")
    args = parser.parse_args()

    if (args.model, args.backend, args.system)  in func_dict.keys():
        func_dict[(args.model, args.backend, args.system)]()
    else:
        print(f'{args.model} {args.backend} {args.system} is not supported')
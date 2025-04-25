import argparse

from Ampere_FP16.benchmark_Ampere_FP16_gemm import Ampere_FP16_Helix_op_level_GEMM_benchmark, Ampere_FP16_cublas_op_level_GEMM_benchmark
from Ampere_FP16.benchmark_Ampere_FP16_conv import Ampere_FP16_Helix_op_level_Conv_benchmark, Ampere_FP16_cudnn_op_level_Conv_benchmark
from Ampere_FP32.benchmark_Ampere_FP32_gemm import Ampere_FP32_Helix_op_level_GEMM_benchmark, Ampere_FP32_cublas_op_level_GEMM_benchmark
from Ampere_FP32.benchmark_Ampere_FP32_conv import Ampere_FP32_Helix_op_level_Conv_benchmark, Ampere_FP32_cudnn_op_level_Conv_benchmark
from x86_CPU.benchmark_x86_CPU_gemm import x86_CPU_Helix_op_level_GEMM_benchmark, x86_CPU_onnxruntime_op_level_GEMM_benchmark, x86_CPU_MKL_op_level_GEMM_benchmark
from x86_CPU.benchmark_x86_CPU_conv import x86_CPU_Helix_op_level_Conv_benchmark, x86_CPU_onnxruntime_op_level_Conv_benchmark, x86_CPU_oneDNN_op_level_Conv_benchmark
from ARM_CPU.benchmark_ARM_CPU_gemm import ARM_CPU_Helix_op_level_GEMM_benchmark, ARM_CPU_onnxruntime_op_level_GEMM_benchmark, ARM_CPU_ACL_op_level_GEMM_benchmark
from ARM_CPU.benchmark_ARM_CPU_conv import ARM_CPU_Helix_op_level_Conv_benchmark, ARM_CPU_onnxruntime_op_level_Conv_benchmark, ARM_CPU_ACL_op_level_Conv_benchmark

func_dict = {
    ('GEMM', 'Ampere_FP16', 'Helix'): Ampere_FP16_Helix_op_level_GEMM_benchmark,
    ('GEMM', 'Ampere_FP16', 'cublas'): Ampere_FP16_cublas_op_level_GEMM_benchmark,
    ('Conv', 'Ampere_FP16', 'Helix'): Ampere_FP16_Helix_op_level_Conv_benchmark,
    ('Conv', 'Ampere_FP16', 'cudnn'): Ampere_FP16_cudnn_op_level_Conv_benchmark,
    ('GEMM', 'Ampere_FP32', 'Helix'): Ampere_FP32_Helix_op_level_GEMM_benchmark,
    ('GEMM', 'Ampere_FP32', 'cublas'): Ampere_FP32_cublas_op_level_GEMM_benchmark,
    ('Conv', 'Ampere_FP32', 'Helix'): Ampere_FP32_Helix_op_level_Conv_benchmark,
    ('Conv', 'Ampere_FP32', 'cudnn'): Ampere_FP32_cudnn_op_level_Conv_benchmark,
    ('GEMM', 'x86_CPU', 'Helix'): x86_CPU_Helix_op_level_GEMM_benchmark,
    ('GEMM', 'x86_CPU', 'onnxruntime'): x86_CPU_onnxruntime_op_level_GEMM_benchmark,
    ('GEMM', 'x86_CPU', 'MKL'): x86_CPU_MKL_op_level_GEMM_benchmark,
    ('Conv', 'x86_CPU', 'Helix'): x86_CPU_Helix_op_level_Conv_benchmark,
    ('Conv', 'x86_CPU', 'onnxruntime'): x86_CPU_onnxruntime_op_level_Conv_benchmark,
    ('Conv', 'x86_CPU', 'oneDNN'): x86_CPU_oneDNN_op_level_Conv_benchmark,
    ('GEMM', 'ARM_CPU', 'Helix'): ARM_CPU_Helix_op_level_GEMM_benchmark,
    ('GEMM', 'ARM_CPU', 'onnxruntime'): ARM_CPU_onnxruntime_op_level_GEMM_benchmark,
    ('GEMM', 'ARM_CPU', 'ACL'): ARM_CPU_ACL_op_level_GEMM_benchmark,
    ('Conv', 'ARM_CPU', 'Helix'): ARM_CPU_Helix_op_level_Conv_benchmark,
    ('Conv', 'ARM_CPU', 'onnxruntime'): ARM_CPU_onnxruntime_op_level_Conv_benchmark,
    ('Conv', 'ARM_CPU', 'ACL'): ARM_CPU_ACL_op_level_Conv_benchmark
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OP level performance")
    parser.add_argument("-o", "--op", type=str, choices=['GEMM', 'Conv'], default='GEMM', help="op type")
    parser.add_argument("-b", "--backend", type=str, choices=['Ampere_FP16', 'Ampere_FP32', 'x86_CPU', 'ARM_CPU'], default='Ampere_FP16', help="backend")
    parser.add_argument("-s", "--system", type=str, choices=['Helix', 'cublas', 'cudnn', 'cutlass', 'DietCode', 'onnxruntime', 'MKL', 'oneDNN', 'ACL'], default='Helix', help="system")
    args = parser.parse_args()

    if (args.op, args.backend, args.system)  in func_dict.keys():
        func_dict[(args.op, args.backend, args.system)]()
    else:
        print(f'{args.op} {args.backend} {args.system} is not supported')
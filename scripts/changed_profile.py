import argparse
import time

from Ampere_FP16.changed_profile_Ampere_FP16 import profile_Helix_Ampere_FP16_gemm_kernel
from Ampere_FP32.changed_profile_Ampere_FP32 import profile_Helix_Ampere_FP32_gemm_kernel
from x86_CPU.changed_profile_x86_CPU import profile_Helix_x86_CPU_gemm_kernel
from ARM_CPU.changed_profile_ARM_CPU import profile_Helix_ARM_CPU_gemm_kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get runtime cost of an operation")
    parser.add_argument("-b", "--backend", type=str, choices=['Ampere_FP16', 'Ampere_FP32', 'x86_CPU', 'ARM_CPU'], default='Ampere_FP16', help="backend")
    args = parser.parse_args()

    T1 = time.perf_counter()
    if args.backend == 'Ampere_FP16':
        profile_Helix_Ampere_FP16_gemm_kernel()
    elif args.backend == 'Ampere_FP32':
        profile_Helix_Ampere_FP32_gemm_kernel()
    elif args.backend == 'x86_CPU':
        profile_Helix_x86_CPU_gemm_kernel()
    elif args.backend == 'ARM_CPU':
        profile_Helix_ARM_CPU_gemm_kernel()
    T2 = time.perf_counter()
    print(f'Profiling time: {T2-T1} seconds')
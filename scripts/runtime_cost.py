import argparse

from Ampere_FP16.Ampere_FP16_runtime_cost import Ampere_FP16_runtime_cost
from Ampere_FP32.Ampere_FP32_runtime_cost import Ampere_FP32_runtime_cost
from x86_CPU.x86_CPU_runtime_cost import x86_CPU_runtime_cost
from ARM_CPU.ARM_CPU_runtime_cost import ARM_CPU_runtime_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get runtime cost of an operation")
    parser.add_argument("-b", "--backend", type=str, choices=['Ampere_FP16', 'Ampere_FP32', 'x86_CPU', 'ARM_CPU'], default='Ampere_FP16', help="backend")
    args = parser.parse_args()

    if args.backend == 'Ampere_FP16':
        Ampere_FP16_runtime_cost()
    elif args.backend == 'Ampere_FP32':
        Ampere_FP32_runtime_cost()
    elif args.backend == 'x86_CPU':
        x86_CPU_runtime_cost()
    elif args.backend == 'ARM_CPU':
        ARM_CPU_runtime_cost()
    else:
        raise ValueError("Invalid backend")
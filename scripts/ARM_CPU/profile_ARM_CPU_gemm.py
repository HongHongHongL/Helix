import os
import numpy as np
from tqdm import tqdm

import tvm
from tvm import tir
from tvm.script import tir as T

root_path = os.getcwd()

target = "llvm"
dtype = "float32"
dev = tvm.device(target, 0)

peak_flops = 48

@T.prim_func
def gemm(a: T.handle, b: T.handle, c: T.handle) -> None:

    T.func_attr({"global_symbol": "main", "tir.noalias": True})

    m = T.var("int32")
    n = T.var("int32")
    k = T.var("int32")

    A = T.match_buffer(a, (m, k), "float32")
    B = T.match_buffer(b, (k, n), "float32")
    C = T.match_buffer(c, (m, n), "float32")

    for m_i in T.serial(m):
        for n_i in T.serial(n):
            for k_i in T.serial(k):
                with T.block("gemm"):
                    vm, vn, vk = T.axis.remap("SSR", [m_i, n_i, k_i])
                    with T.init():
                        C[vm, vn] = T.float32(0)
                    C[vm, vn] = C[vm, vn] + A[vm, vk] * B[vk, vn]

def sch_gemm(m, n, k, m1, m2, n1, n2, k1):

    data, weight, _ = gemm.params
    sch = tir.Schedule(gemm.specialize(
        {
            data: tvm.tir.decl_buffer((m, k)), weight: tvm.tir.decl_buffer((k, n)),
        }
    ))
    gemm_block = sch.get_block("gemm")
    vm, vn, vk = sch.get_loops(gemm_block)
    sch.split(vm, factors=[None, m2, m1])
    sch.split(vn, factors=[None, n2, n1])
    sch.split(vk, factors=[None, k1])
    vm3, vm2, vm1, vn3, vn2, vn1, vk2, vk1 = sch.get_loops(gemm_block)
    sch.reorder(vm3, vn3, vk2, vm2, vn2, vk1, vm1, vn1)
    sch.vectorize(vn1)
    sch.unroll(vm1)

    sch.decompose_reduction(gemm_block, vk2)

    return sch

def profile_Helix_ARM_CPU_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/prof_dict"):
        os.makedirs(f"{root_path}/build/prof_dict")

    register_block_dict = {}

    print("Start profiling ARM CPU GEMM kernel ...")
    with tqdm(total=9*4*4, desc="profiling") as pbar:
        for M1 in range(1, 10):
            for M2 in range(1, 2):
                for N1 in range(1, 5):
                    for N2 in range(1, 2):
                        for K1 in range(1, 5):
                            if N1 == 3:
                                N1 = 1
                                N2 = 3
                            M, N, K = M1 * M2 * 20, N1 * N2 * 4 * 4, K1 * 16
                            sch = sch_gemm(M, N, K, M1, M2, N1 * 4, N2, K1)
                            func = tvm.build(sch.mod, target=target)

                            evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
                            d = np.random.rand(M, K).astype(dtype)
                            w = np.random.rand(K, N).astype(dtype)
                            Data = tvm.nd.array(d, dev)
                            Weight = tvm.nd.array(w, dev)
                            O = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
                            cost = evaluator(Data, Weight, O).mean
                            flops = (2 * M * N * K * 1e-9) / cost
                            if flops > peak_flops * 0.6:
                                register_block_dict[(M1 * M2, N1 * N2 * 4, K1)] = (cost, flops)
                            pbar.update(1)

    cache_block_dict = {}

    for key, value in register_block_dict.items():
        M1, N1, K1 = key
        M2 = 1
        N2 = 1
        if (N1 == 12):
            N1 = 4
            N2 = 3

        for Mx in range(1, 256 // M1 + 1):
            for Nx in range(1, 256 // (N1 * N2) + 1):
                M = Mx * M1 * M2
                N = Nx * N1 * N2
                K = 16 * K1

                cost = value[0] / (20 * 4 * 4) * (Mx * Nx)
                flops = (2 * M * N * K * 1e-9) / cost

                if flops > peak_flops * 0.6:
                    if (M, N, K) in cache_block_dict.keys():
                        if cache_block_dict[(M, N, K)][1] > cost:
                            cache_block_dict[(M, N, K)] = (key, cost, flops)
                    else:
                        cache_block_dict[(M, N, K)] = (key, cost, flops)

    with open(f'{root_path}/build/prof_dict/ARM_CPU_cost_model.dict', 'w') as f:
        f.write(str(cache_block_dict))
    print("Profile finished.")

if __name__ == "__main__":

    profile_Helix_ARM_CPU_gemm_kernel()
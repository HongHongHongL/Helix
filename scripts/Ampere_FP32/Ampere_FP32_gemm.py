import math
import os

import numpy as np

import tvm, tvm.testing
from tvm.script import tir as T
from tvm import tir

target = "nvidia/nvidia-a100"
dtype = "float32"
dev = tvm.device("cuda", 0)

root_path = os.getcwd()

@T.prim_func
def gemm(a: T.handle, b: T.handle, c: T.handle, m1: T.int32, m2: T.int32, m3: T.int32, m4: T.int32, m5: T.int32, n1: T.int32, n2: T.int32, n3: T.int32, n4: T.int32, n5: T.int32, k1: T.int32, k2: T.int32, k3: T.int32):
    T.func_attr({"global_symbol": "matmul", "tir.noalias": T.bool(True)})
    m = T.var("int32")
    n = T.var("int32")
    k = T.var("int32")
    A = T.match_buffer(a, (m, k), "float32")
    B = T.match_buffer(b, (k, n), "float32")
    C = T.match_buffer(c, (m, n), "float32")
    
    for i, j, k in T.grid(m1 * m2 * m3 * m4 * m5, n1 * n2 * n3 * n4 * n5, k1 * k2 * k3):
        with T.block("C"):
            vi = T.axis.spatial(m1 * m2 * m3 * m4 * m5, i)
            vj = T.axis.spatial(n1 * n2 * n3 * n4 * n5, j)
            vk = T.axis.reduce(k1 * k2 * k3, k)
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
            

def apply_trace(sch: tvm.tir.Schedule, m1, m2, m3, m4, m5, n1, n2, n3, n4, n5, k1, k2, k3):
    block_C = sch.get_block("C")
    block_shared_B = sch.cache_read(block=block_C, read_buffer_index=1, storage_scope='shared')
    block_shared_A = sch.cache_read(block=block_C, read_buffer_index=0, storage_scope='shared')
    block_local_C = sch.cache_write(block=block_C, write_buffer_index=0, storage_scope='local')
    
    (i, j, k) = sch.get_loops(block=block_C)
    bxm, vxm, txm, im2, im1 = sch.split(loop=i, factors=[None, m4, m3, m2, m1])
    bxn, vxn, txn, jn2, jn1 = sch.split(loop=j, factors=[None, n4, n3, n2, n1])
    kk3, kk2, kk1 = sch.split(loop=k, factors=[None, k2, k1])
    sch.reorder(bxm, bxn, vxm, vxn, txm, txn, kk3, kk2, im2, jn2, kk1, im1, jn1)
    bx = sch.fuse(bxm, bxn)
    vx = sch.fuse(vxm, vxn)
    tx = sch.fuse(txm, txn)
    
    sch.bind(loop=bx, thread_axis='blockIdx.x')
    sch.bind(loop=vx, thread_axis='vthread.x')
    sch.bind(loop=tx, thread_axis='threadIdx.x')
    # sch.annotate(block_or_loop=bx, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    # sch.annotate(block_or_loop=bx, ann_key="pragma_unroll_explicit", ann_val=1)
    
    sch.compute_at(block=block_shared_A, loop=kk3)
    sch.compute_at(block=block_shared_B, loop=kk3)
    sch.reverse_compute_at(block=block_local_C, loop=tx)
    
    asi, asj = sch.get_loops(block=block_shared_A)[-2 : ]
    asi, txm,  = sch.split(loop=asi, factors=[None, m3])
    asj, txn, asjvec = sch.split(loop=asj, factors=[None, n3, 4])
    sch.reorder(asi, asj, txm, txn, asjvec)
    _ = sch.fuse(asi, asj)
    tx = sch.fuse(txm, txn)
    sch.bind(loop=tx, thread_axis='threadIdx.x')
    sch.vectorize(loop=asjvec)
    
    bsi, bsj = sch.get_loops(block=block_shared_B)[-2 : ]
    bsi, txm = sch.split(loop=bsi, factors=[None, m3])
    bsj, txn, bsjvec = sch.split(loop=bsj, factors=[None, n3, 4])
    sch.reorder(bsi, bsj, txm, txn, bsjvec)
    _ = sch.fuse(bsi, bsj)
    tx = sch.fuse(txm, txn)
    sch.bind(loop=tx, thread_axis='threadIdx.x')
    sch.vectorize(loop=bsjvec)
    
    sch.decompose_reduction(block=block_C, loop=kk3)
    
    # 3 pipelined (double) buffer
    sch.annotate(block_or_loop=kk3, ann_key='software_pipeline_stage', ann_val=[0, 0, 4])
    sch.annotate(block_or_loop=kk3, ann_key='software_pipeline_order', ann_val=[0, 1, 2])
    
    # # cp.async
    sch.annotate(block_or_loop=kk3, ann_key="software_pipeline_async_stages", ann_val=[0])


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

def get_Ampere_FP32_gemm_Helix_result(prof_dict, M, N, K, backend):
    best_config = cost_model(prof_dict, M, N, K)
    if backend == "cuda":
        cmd = f'{root_path}/build/bin_fp32/sgemm_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]}_{best_config[5]}_{best_config[6]}_{best_config[7]} {math.ceil(M / best_config[0]) * best_config[0]} {math.ceil(N / best_config[1]) * best_config[1]} {K}'
        result = os.popen(cmd)
        cost = float(result.read().split()[-8])
        tflops = 2 * M * N * K / 1e12 / cost
    else:
        M4, M3, M2, M1 = 16, 8, 4, 2
        M5 = M // (M4 * M3 * M2 * M1)
        N4, N3, N2, N1 = 16, 8, 4, 2
        N5 = N // (N4 * N3 * N2 * N1)
        K2, K1 = 1, 32
        K3 = K // (K2 * K1)

        data, weight, _, m1, m2, m3, m4, m5, n1, n2, n3, n4, n5, k1, k2, k3 = gemm.params

        sch = tir.Schedule(gemm.specialize(
            {
                data: tvm.tir.decl_buffer((M, K)), weight: tvm.tir.decl_buffer((K, N)),
                m1: M1, m2: M2, m3: M3, m4: M4, m5: M5, n1: N1, n2: N2, n3: N3, n4: N4, n5: N5, k1: K1, k2: K2, k3: K3
            }
        ))
        
        apply_trace(sch, M1, M2, M3, M4, M5, N1, N2, N3, N4, N5, K1, K2, K3)

        func = tvm.build(sch.mod, target=target)

        evaluator = func.time_evaluator(func.entry_name, dev, number=100)
        d = np.random.rand(M, K).astype(dtype)
        w = np.random.rand(K, N).astype(dtype)
        Data = tvm.nd.array(d, dev)
        Weight = tvm.nd.array(w, dev)
        O = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
        cost = evaluator(Data, Weight, O).mean
        tflops = 2 * M * N * K * 1e-9 / cost

    return cost, tflops

def get_Ampere_FP32_gemm_cublas_result(M, N, K):
    cmd = f'{root_path}/build/bin_fp32/cublas_f32 {M} {N} {K}'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])
    tflops = 2 * M * N * K / 1e9 / cost

    return cost, tflops
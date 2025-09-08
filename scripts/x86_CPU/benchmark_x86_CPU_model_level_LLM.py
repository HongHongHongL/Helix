import os
import time
import math
import numpy as np

import tvm
from tvm import tir
from tvm.script import tir as T

import onnx
from onnx import helper, TensorProto
import onnxruntime

from utils.get_benchmark_shape_list import get_llm_opset_MNKList

root_path = os.getcwd()

target = "llvm -mcpu=skylake-avx512"
dtype = "float32"
dev = tvm.device(target, 0)

num_thread = 48

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

def get_tiling(prof_dict, M, N, K):

    min_cost = float("inf")
    min_cache = (1, 1, 1)
    min_reg = (1, 1, 1)

    for k, v in prof_dict.items():
        thread = math.ceil(N / k[1])
        cost = v[1] * math.ceil(M / k[0]) * math.ceil(thread / num_thread)

        if min_cost > cost:
            min_cost = cost
            min_cache = k
            min_reg = v[0]

    M1, M2, M3 = min_reg[0], 1, min_cache[0] // min_reg[0]
    N1, N2, N3 = min_reg[1], 1, min_cache[1] // min_reg[1]
    if N1 == 48 or N1 == 80:
        N2 = N1 // 16
        N1 = 16
    K1 = min_reg[2]

    return M1, M2, M3, N1, N2, N3, K1


def sch_gemm(prof_dict, m, n, k):

    m1, m2, m3, n1, n2, n3, k1 = get_tiling(prof_dict, m, n, k)

    data, weight, _ = gemm.params
    sch = tir.Schedule(gemm.specialize(
        {
            data: tvm.tir.decl_buffer((math.ceil(m / (m1 * m2 * m3)) * (m1 * m2 * m3), math.ceil(k / k1) * k1)), weight: tvm.tir.decl_buffer((math.ceil(k / k1) * k1, math.ceil(n / (n1 * n2 * n3)) * (n1 * n2 * n3))),
        }
    ))
    gemm_block = sch.get_block("gemm")
    vm, vn, vk = sch.get_loops(gemm_block)
    sch.split(vm, factors=[None, m3, m2, m1])
    sch.split(vn, factors=[None, n3, n2, n1])
    sch.split(vk, factors=[None, k1])
    vm4, vm3, vm2, vm1, vn4, vn3, vn2, vn1, vk2, vk1 = sch.get_loops(
        gemm_block)
    sch.reorder(vn4, vm4, vm3, vn3, vk2, vm2, vn2, vk1, vm1, vn1)
    sch.vectorize(vn1)
    sch.unroll(vk1)
    sch.unroll(vm1)
    sch.parallel(vn4)

    pack_block = sch.reindex_cache_read(gemm_block, 1, "local", lambda vm, vn, vk: (
        vn // (n1 * n2 * n3), (vn // (n1 * n2)) % n3, vk // k1, (vn // n1) % n2, vk % k1, vn % n1))
    sch.compute_at(block=pack_block, loop=vn4, preserve_unit_loops=True)
    _, vpn, vpk = sch.get_loops(pack_block)
    sch.split(vpn, factors=[None, n2, n1])
    sch.split(vpk, factors=[None, k1])
    _, vpn3, vpn2, vpn1, vpk2, vpk1 = sch.get_loops(pack_block)
    sch.reorder(vpn3, vpk2, vpn2, vpk1, vpn1)
    sch.vectorize(vpn1)

    cache_block = sch.cache_write(
        block=gemm_block, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=cache_block, loop=vn4,
                           preserve_unit_loops=True)
    _, vcm, vcn= sch.get_loops(cache_block)
    sch.split(vcm, factors=[None, m2, m1])
    sch.split(vcn, factors=[None, n2, n1])
    _, vcm3, vcm2, vcm1, vcn3, vcn2, vcn1 = sch.get_loops(cache_block)
    sch.reorder(vcm3, vcn3, vcm2, vcn2, vcm1, vcn1)
    sch.vectorize(vcn1)

    sch.decompose_reduction(gemm_block, vm3)

    return sch, math.ceil(m / (m1 * m2 * m3)) * (m1 * m2 * m3), math.ceil(n / (n1 * n2 * n3)) * (n1 * n2 * n3), math.ceil(k / k1) * k1

def get_Helix_result(prof_dict, M, N, K):
    os.environ['TVM_NUM_THREADS'] = str(num_thread)
    os.environ['OMP_NUM_THREADS'] = str(num_thread)

    sch, M, N, K = sch_gemm(prof_dict, M, N, K)
    func = tvm.build(sch.mod, target=target)

    evaluator = func.time_evaluator(
        func.entry_name, dev, number=100)
    d = np.random.rand(M, K).astype(dtype)
    w = np.random.rand(K, N).astype(dtype)
    Data = tvm.nd.array(d, dev)
    Weight = tvm.nd.array(w, dev)
    O = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
    cost = evaluator(Data, Weight, O).mean

    return cost

def get_onnxruntime_result(M, N, K):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [M, K])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [K, N])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [M, N])

    node = helper.make_node(
        "Gemm",
        inputs=["x", "w"],
        outputs=["y"],
    )
    graph_def = helper.make_graph(
        [node],
        'test_gemm_model',
        [x, w],
        [y],
    )
    model = onnx.helper.make_model(graph_def, producer_name='onnx-example', opset_imports=[helper.make_opsetid('', 13)])

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = num_thread
    opts.inter_op_num_threads = num_thread
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])

    ort_inputs = {}
    ort_inputs['x'] = np.random.rand(M, K).astype(np.float32)
    ort_inputs['w'] = np.random.rand(K, N).astype(np.float32)

    outputs = [x.name for x in ort_session.get_outputs()]

    T1 = time.perf_counter()
    for _ in range(100):
        ort_outs = ort_session.run(outputs, ort_inputs)
    T2 =time.perf_counter()
    return (T2 - T1) / 100

def get_MKL_result(M, N, K):
    if os.path.exists(f'{root_path}/build/bin_mkl/run_benchmark'):
        result = os.popen(f'{root_path}/build/bin_mkl/run_benchmark {M} {N} {K}')
        context = result.read()
        flops = float(context.splitlines()[0])
        return 2 * M * N * K * 1e-9 / flops
    else:
        return 0

def x86_CPU_Helix_model_level_LLM_benchmark():
    with open(f'{root_path}/build/prof_dict/x86_CPU_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    helix_Bert_cost, helix_LLAMA2_cost, helix_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        helix_Bert_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in LLAMA2_MNKList:
        helix_LLAMA2_cost += get_Helix_result(prof_dict, M, N, K)
    for M, N, K in GPT2_MNKList:
        helix_GPT2_cost += get_Helix_result(prof_dict, M, N, K)

    print(f'Bert: {helix_Bert_cost:.2f} ms, LLAMA2: {helix_LLAMA2_cost:.2f} ms, GPT2: {helix_GPT2_cost:.2f} ms')

def x86_CPU_onnxruntime_model_level_LLM_benchmark():
    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    onnxruntime_Bert_cost, onnxruntime_LLAMA2_cost, onnxruntime_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        onnxruntime_Bert_cost += get_onnxruntime_result(M, N, K)
    for M, N, K in LLAMA2_MNKList:
        onnxruntime_LLAMA2_cost += get_onnxruntime_result(M, N, K)
    for M, N, K in GPT2_MNKList:
        onnxruntime_GPT2_cost += get_onnxruntime_result(M, N, K)

    print(f'Bert: {onnxruntime_Bert_cost:.2f} ms, LLAMA2: {onnxruntime_LLAMA2_cost:.2f} ms, GPT2: {onnxruntime_GPT2_cost:.2f} ms')

def x86_CPU_MKL_model_level_LLM_benchmark():
    Bert_MNKList, LLAMA2_MNKList, GPT2_MNKList = get_llm_opset_MNKList()

    mkl_Bert_cost, mkl_LLAMA2_cost, mkl_GPT2_cost = 0, 0, 0
    for M, N, K in Bert_MNKList:
        mkl_Bert_cost += get_MKL_result(M, N, K)
    for M, N, K in LLAMA2_MNKList:
        mkl_LLAMA2_cost += get_MKL_result(M, N, K)
    for M, N, K in GPT2_MNKList:
        mkl_GPT2_cost += get_MKL_result(M, N, K)

    print(f'Bert: {mkl_Bert_cost:.2f} ms, LLAMA2: {mkl_LLAMA2_cost:.2f} ms, GPT2: {mkl_GPT2_cost:.2f} ms')

if __name__ == "__main__":

    x86_CPU_Helix_model_level_LLM_benchmark()
    x86_CPU_onnxruntime_model_level_LLM_benchmark()
    x86_CPU_MKL_model_level_LLM_benchmark()
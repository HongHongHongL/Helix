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

from utils.get_benchmark_shape_list import get_cnn_opset_shape_List

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

def get_onnxruntime_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad):
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [batch, input_channel, H, W])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [output_channel, input_channel, kH, kW])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [batch, output_channel, (H + pad * 2 - kH) // stride + 1, (W + pad * 2 - kW) // stride + 1])

    node_with_padding = helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["y"],
        kernel_shape=[kH, kW],
        strides=[stride, stride],
        dilations=[1, 1],
        group=1,
        pads=[pad, pad, pad, pad],
    )

    graph_def = helper.make_graph(
        [node_with_padding],
        'test_conv_model',
        [x, w],
        [y],
    )

    model = onnx.helper.make_model(graph_def, producer_name='onnx-example')

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = num_thread
    opts.inter_op_num_threads = num_thread
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])

    ort_inputs = {}
    data = []
    ort_inputs['x'] = np.random.rand(batch, input_channel, H, W).astype(np.float32)

    outputs = [x.name for x in ort_session.get_outputs()]

    T1 = time.perf_counter()
    for _ in range(100):
        ort_outs = ort_session.run(outputs, ort_inputs)
    T2 =time.perf_counter()
    return (T2 - T1) / 100

def get_oneDNN_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad):
    if os.path.exists(f'{root_path}/build/bin_onednn/benchdnn'):
        with open(f'{root_path}/build/bin_onednn/shape', 'w') as f:
            f.write(f'mb{batch}ic{input_channel}ih{H}iw{W}oc{output_channel}oh{(H + 2 * pad - kH) // stride + 1}ow{(W + 2 * pad - kW) // stride + 1}kh{kH}kw{kW}sh{stride}sw{stride}ph{pad}pw{pad}')
        result = os.popen(f'{root_path}/build/bin_onednn/benchdnn --conv --dt=f32 --dir=FWD_B --batch={root_path}/build/bin_onednn/shape')
        context = result.read()
        cost = float(context.splitlines()[0])
        return cost
    else:
        return 0

def x86_CPU_Helix_model_level_CNN_benchmark():
    with open(f'{root_path}/build/prof_dict/x86_CPU_cost_model.dict', 'r') as f:
        lines = f.readlines()
        prof_dict = eval(lines[0])

    AlexNet_shape_list, ResNet_shape_list, GoogleNet_shape_list = get_cnn_opset_shape_List()

    helix_AlexNet_cost, helix_ResNet_cost, helix_GoogleNet_cost = 0, 0, 0
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in AlexNet_shape_list:
        M = batch * ((H + 2 * pad - kH) // stride + 1) * ((W + 2 * pad - kW) // stride + 1)
        N = output_channel
        K = input_channel * kH * kW
        helix_AlexNet_cost += get_Helix_result(prof_dict, M, N, K)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in ResNet_shape_list:
        M = batch * ((H + 2 * pad - kH) // stride + 1) * ((W + 2 * pad - kW) // stride + 1)
        N = output_channel
        K = input_channel * kH * kW
        helix_ResNet_cost += get_Helix_result(prof_dict, M, N, K)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in GoogleNet_shape_list:
        M = batch * ((H + 2 * pad - kH) // stride + 1) * ((W + 2 * pad - kW) // stride + 1)
        N = output_channel
        K = input_channel * kH * kW
        helix_GoogleNet_cost += get_Helix_result(prof_dict, M, N, K)

    print(f'AlexNet: {helix_AlexNet_cost:.2f} ms, ResNet: {helix_ResNet_cost:.2f} ms, GoogleNet: {helix_GoogleNet_cost:.2f} ms')

def x86_CPU_onnxruntime_model_level_CNN_benchmark():
    AlexNet_shape_list, ResNet_shape_list, GoogleNet_shape_list = get_cnn_opset_shape_List()

    onnxruntime_AlexNet_cost, onnxruntime_ResNet_cost, onnxruntime_GoogleNet_cost = 0, 0, 0
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in AlexNet_shape_list:
        onnxruntime_AlexNet_cost += get_onnxruntime_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in ResNet_shape_list:
        onnxruntime_ResNet_cost += get_onnxruntime_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in GoogleNet_shape_list:
        onnxruntime_GoogleNet_cost += get_onnxruntime_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)

    print(f'AlexNet: {onnxruntime_AlexNet_cost:.2f} ms, ResNet: {onnxruntime_ResNet_cost:.2f} ms, GoogleNet: {onnxruntime_GoogleNet_cost:.2f} ms')

def x86_CPU_oneDNN_model_level_CNN_benchmark():
    AlexNet_shape_list, ResNet_shape_list, GoogleNet_shape_list = get_cnn_opset_shape_List()

    oneDNN_AlexNet_cost, oneDNN_ResNet_cost, oneDNN_GoogleNet_cost = 0, 0, 0
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in AlexNet_shape_list:
        oneDNN_AlexNet_cost += get_oneDNN_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in ResNet_shape_list:
        oneDNN_ResNet_cost += get_oneDNN_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)
    for batch, input_channel, H, W, output_channel, kH, kW, stride, pad in GoogleNet_shape_list:
        oneDNN_GoogleNet_cost += get_oneDNN_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad)

    print(f'AlexNet: {oneDNN_AlexNet_cost:.2f} ms, ResNet: {oneDNN_ResNet_cost:.2f} ms, GoogleNet: {oneDNN_GoogleNet_cost:.2f} ms')

if __name__ == "__main__":

    x86_CPU_Helix_model_level_CNN_benchmark()
    x86_CPU_onnxruntime_model_level_CNN_benchmark()
    x86_CPU_oneDNN_model_level_CNN_benchmark()
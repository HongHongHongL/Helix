import math
import os

import tvm
from tvm.script import tir as T
from tvm import tir
import numpy as np
from tvm.tir import tensor_intrin

target = "nvidia/nvidia-a100"
dtype = "float16"
dev = tvm.device("cuda", 0)

root_path = os.getcwd()

@T.prim_func
def gemm(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "matmul", "tir.noalias": T.bool(True)})
    M = T.var("int32")
    N = T.var("int32")
    K = T.var("int32")
    A = T.match_buffer(a, (M, K), dtype)
    B = T.match_buffer(b, (N, K), dtype)
    C = T.match_buffer(c, (M, N), dtype)
    
    for i, j, k in T.grid(M, N, K):
        with T.block("C"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            vk = T.axis.reduce(K, k)
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
            

def apply_trace(sch: tvm.tir.Schedule,
        K_STAGE: T.int32, BLOCK_ROWS: T.int32, 
        BLOCK_COLS: T.int32, BLOCK_K: T.int32,
        WARP_ROWS: T.int32, WARP_COLS: T.int32, 
        BLOCK_STRIDE: T.int32, 
        MMA_M: T.int32, MMA_N: T.int32, MMA_K: T.int32, 
        WARP_SIZE: T.int32, 
        THREAD_COPY_BYTES: T.int32,
        PERMUTED_OFFSET: T.int32, PERMUTED_COLS: T.int32,
    
        BLOCK_ROW_WARPS : T.int32,
        BLOCK_COL_WARPS : T.int32,
        BLOCK_ROW_TILES : T.int32,
        BLOCK_COL_TILES : T.int32,
        WARP_ROW_TILES : T.int32,
        WARP_COL_TILES : T.int32,
        WARPS_PER_BLOCK : T.int32,
        THREADS_PER_BLOCK : T.int32,
        CHUNK_K : T.int32,
        CHUNK_LINE_BYTES : T.int32,
        CHUNK_COPY_LINES_PER_WARP : T.int32, 
        CHUNK_COPY_LINE_LANES : T.int32,
        AB_SMEM_STRIDE : T.int32,
        C_SMEM_STRIDE : T.int32,
        C_SMEM_OFFSET : T.int32,
        SMEM_BANK_ROWS : T.int32,
        SMEM_WARP_OFFSET : T.int32,
        ARRAY_OFFSET_INSMEM : T.int32):
    block_C = sch.get_block("C")
    i, j, k = sch.get_loops(block=block_C)
    
    # Cache read and write
    block_shared_B = sch.cache_read(block=block_C, read_buffer_index=1, storage_scope="shared.dyn")
    block_shared_A = sch.cache_read(block=block_C, read_buffer_index=0, storage_scope="shared.dyn")
    block_local_B = sch.cache_read(block=block_C, read_buffer_index=1, storage_scope="warp")
    block_local_A = sch.cache_read(block=block_C, read_buffer_index=0, storage_scope="warp")
    block_shared_C = sch.cache_write(block=block_C, write_buffer_index=0, storage_scope="shared.dyn")
    block_local_C = sch.cache_write(block=block_C, write_buffer_index=0, storage_scope="warp")
    
    # Split M = by * ty_i * iter_i * mma_i
    by, i = sch.split(loop=i, factors=[None, BLOCK_ROWS])
    ty_i, i = sch.split(loop=i, factors=[None, WARP_ROWS])
    iter_i, mma_i = sch.split(loop=i, factors=[None, MMA_M])
    
    # Split N = bz * ty_j * iter_j * (mma_j * 2)
    #TODO: 2 * m16n8k16 = m16n16k16
    bz, j = sch.split(loop=j, factors=[None, BLOCK_COLS])
    ty_j, j = sch.split(loop=j, factors=[None, WARP_COLS])
    iter_j, mma_j = sch.split(loop=j, factors=[None, MMA_N * 2 if MMA_N == 8 else MMA_N])
    
    # Split K = k_smem_stage * k_reg_stage * mma_k
    k_smem_stage, k = sch.split(loop=k, factors=[None, BLOCK_K])
    k_reg_stage, mma_k = sch.split(loop=k, factors=[None, MMA_K])
    
    # Reorder and Fuse
    sch.reorder(by, bz, ty_i, ty_j, k_smem_stage, k_reg_stage, iter_i, iter_j, mma_i, mma_j, mma_k)
    ty = sch.fuse(ty_i, ty_j)
    sch.reorder(by, bz, ty, k_smem_stage, k_reg_stage, iter_i, iter_j, mma_i, mma_j, mma_k)
    
    # Bind threadIdx.x = 32
    sch.bind(loop=by, thread_axis="blockIdx.y")
    sch.bind(loop=bz, thread_axis="blockIdx.z")
    sch.bind(loop=ty, thread_axis="threadIdx.y")
    
    # Compute at
    sch.compute_at(block=block_local_A, loop=k_reg_stage)
    sch.compute_at(block=block_local_B, loop=k_reg_stage)
    sch.compute_at(block=block_shared_A, loop=k_smem_stage)
    sch.compute_at(block=block_shared_B, loop=k_smem_stage)
    sch.reverse_compute_at(block=block_local_C, loop=ty)
    sch.reverse_compute_at(block=block_shared_C, loop=bz)
    
    # Decompose Init
    block_C_init = sch.decompose_reduction(block=block_C, loop=k_smem_stage)
    
    # Co fetch
    @T.prim_func
    def A_g2s_desc(A_global: T.handle, A_smem: T.handle) -> None :
        Ag = T.match_buffer(param=A_global, shape=(BLOCK_ROWS, BLOCK_K), dtype="float16", scope="global")
        As = T.match_buffer(param=A_smem, shape=(BLOCK_ROWS, BLOCK_K), dtype="float16", scope="shared.dyn")
        with T.block("root"):
            T.reads(Ag[0:BLOCK_ROWS, 0:BLOCK_K])
            T.writes(As[0:BLOCK_ROWS, 0:BLOCK_K])
            for i, k in T.grid(BLOCK_ROWS, BLOCK_K):
                with T.block("update"):
                    vi, vk = T.axis.remap("SS", [i, k])
                    As[vi, vk] = Ag[vi, vk]
    @T.prim_func
    def A_g2s_intrin(A_global: T.handle, A_smem: T.handle) -> None :
        Ag = T.match_buffer(param=A_global, shape=(BLOCK_ROWS, BLOCK_K), dtype="float16", scope="global", offset_factor=1)
        As = T.match_buffer(param=A_smem, shape=(BLOCK_ROWS, BLOCK_K), dtype="float16", scope="shared.dyn", offset_factor=1)
        with T.block("root"):
            T.reads(Ag[0:BLOCK_ROWS, 0:BLOCK_K])
            T.writes(As[0:BLOCK_ROWS, 0:BLOCK_K])
            for ty in T.thread_binding((BLOCK_ROWS // WARP_ROWS) * (BLOCK_COLS // WARP_COLS), thread="threadIdx.y"):
                for tx in T.thread_binding(WARP_SIZE, thread="threadIdx.x"):
                    for as_iter in T.serial(BLOCK_ROWS // (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK)):
                        for as_vec in T.vectorized(THREAD_COPY_BYTES // 2):
                            with T.block("A_shared"):
                                gsi = T.axis.spatial(BLOCK_ROWS, BLOCK_ROWS // WARPS_PER_BLOCK * ty + as_iter * CHUNK_COPY_LINES_PER_WARP + tx // CHUNK_COPY_LINE_LANES)
                                gj = T.axis.spatial(BLOCK_K, tx % CHUNK_COPY_LINE_LANES * (THREAD_COPY_BYTES // 2) + as_vec)
                                sj = T.axis.spatial(BLOCK_K, (tx % CHUNK_COPY_LINE_LANES + tx // CHUNK_COPY_LINE_LANES % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS) // SMEM_BANK_ROWS)% CHUNK_COPY_LINE_LANES * (THREAD_COPY_BYTES // 2) + as_vec)
                                As[gsi, sj] = Ag[gsi, gj]
                                # As[gsi, gj] = Ag[gsi, gj] # NOTE without Permute
    tir.TensorIntrin.register("Ag2s", A_g2s_desc, A_g2s_intrin)
    i_blockm, j_blockk = sch.get_loops(block=block_shared_A)[-2 : ]
    sch.tensorize(block_or_loop=i_blockm, tensor_intrin="Ag2s")
    
    @T.prim_func
    def B_g2s_desc(B_global: T.handle, B_smem: T.handle) -> None :
        Bg = T.match_buffer(param=B_global, shape=(BLOCK_COLS, BLOCK_K), dtype="float16", scope="global")
        Bs = T.match_buffer(param=B_smem, shape=(BLOCK_COLS, BLOCK_K), dtype="float16", scope="shared.dyn")
        with T.block("root"):
            T.reads(Bg[0:BLOCK_COLS, 0:BLOCK_K])
            T.writes(Bs[0:BLOCK_COLS, 0:BLOCK_K])
            for j, k in T.grid(BLOCK_COLS, BLOCK_K):
                with T.block("update"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    Bs[vj, vk] = Bg[vj, vk]
    @T.prim_func
    def B_g2s_intrin(B_global: T.handle, B_smem: T.handle) -> None :
        Bg = T.match_buffer(param=B_global, shape=(BLOCK_COLS, BLOCK_K), dtype="float16", scope="global", offset_factor=1)
        Bs = T.match_buffer(param=B_smem, shape=(BLOCK_COLS, BLOCK_K), dtype="float16", scope="shared.dyn", offset_factor=1)
        with T.block("root"):
            T.reads(Bg[0:BLOCK_COLS, 0:BLOCK_K])
            T.writes(Bs[0:BLOCK_COLS, 0:BLOCK_K])
            for ty in T.thread_binding((BLOCK_ROWS // WARP_ROWS) * (BLOCK_COLS // WARP_COLS), thread="threadIdx.y"):
                for tx in T.thread_binding(WARP_SIZE, thread="threadIdx.x"):
                    for bs_iter in T.serial(BLOCK_COLS // (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK)):
                        for bs_vec in T.vectorized(THREAD_COPY_BYTES // 2):
                            with T.block("B_shared"):
                                gsi = T.axis.spatial(BLOCK_COLS, BLOCK_COLS // WARPS_PER_BLOCK * ty + bs_iter * CHUNK_COPY_LINES_PER_WARP + tx // CHUNK_COPY_LINE_LANES)
                                gj = T.axis.spatial(BLOCK_K, tx % CHUNK_COPY_LINE_LANES * (THREAD_COPY_BYTES // 2) + bs_vec)
                                sj = T.axis.spatial(BLOCK_K, (tx % CHUNK_COPY_LINE_LANES + tx // CHUNK_COPY_LINE_LANES % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS) // SMEM_BANK_ROWS) % CHUNK_COPY_LINE_LANES * (THREAD_COPY_BYTES // 2) + bs_vec)
                                Bs[gsi, sj] = Bg[gsi, gj]
                                # Bs[gsi, gj] = Bg[gsi, gj] # NOTE without Permute
    tir.TensorIntrin.register("Bg2s", B_g2s_desc, B_g2s_intrin)
    i_blockn, j_blockk = sch.get_loops(block=block_shared_B)[-2 : ]
    sch.tensorize(block_or_loop=i_blockn, tensor_intrin="Bg2s")
    
    
    @T.prim_func
    def C_s2g_desc(C_smem: T.handle, C_global: T.handle) -> None :
        Cs = T.match_buffer(param=C_smem, shape=(BLOCK_ROWS, BLOCK_COLS), dtype="float16", scope="shared.dyn")
        Cg = T.match_buffer(param=C_global, shape=(BLOCK_ROWS, BLOCK_COLS), dtype="float16", scope="global")
        with T.block("root"):
            T.reads(Cs[0:BLOCK_ROWS, 0:BLOCK_COLS])
            T.writes(Cg[0:BLOCK_ROWS, 0:BLOCK_COLS])
            for i, j in T.grid(BLOCK_ROWS, BLOCK_COLS):
                with T.block("update"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Cg[vi, vj] = Cs[vi, vj]

    write_back_line_thread = BLOCK_COLS * 2 // THREAD_COPY_BYTES
    write_per_warp_line = WARP_SIZE // write_back_line_thread
    write_warp_lines = BLOCK_ROWS // WARPS_PER_BLOCK
    @T.prim_func
    def C_s2g_intrin(C_smem: T.handle, C_global: T.handle) -> None:
        Cs = T.match_buffer(param=C_smem, shape=(BLOCK_ROWS, BLOCK_COLS), dtype="float16", scope="shared.dyn", offset_factor=1)
        Cg = T.match_buffer(param=C_global, shape=(BLOCK_ROWS, BLOCK_COLS), dtype="float16", scope="global", offset_factor=1)
        with T.block("root"):
            T.reads(Cs[0:BLOCK_ROWS, 0:BLOCK_COLS])
            T.writes(Cg[0:BLOCK_ROWS, 0:BLOCK_COLS])
            for ty in T.thread_binding((BLOCK_ROWS // WARP_ROWS) * (BLOCK_COLS // WARP_COLS), thread="threadIdx.y"):
                for tx in T.thread_binding(WARP_SIZE, thread="threadIdx.x"):
                    for cs_iter in T.serial((BLOCK_ROWS // WARPS_PER_BLOCK) // (WARP_SIZE // (BLOCK_COLS * 2 // THREAD_COPY_BYTES))):
                        for cs_vec in T.vectorized(THREAD_COPY_BYTES // 2):
                            with T.block("C_shared"):
                                gsi = T.axis.spatial(BLOCK_ROWS, ty * write_warp_lines + cs_iter * write_per_warp_line + tx // write_back_line_thread)
                                gj = T.axis.spatial(BLOCK_COLS, (tx % write_back_line_thread) * (THREAD_COPY_BYTES // 2) + cs_vec)
                                sj = T.axis.spatial(BLOCK_COLS, ((tx % write_back_line_thread + (cs_iter * write_per_warp_line + tx // write_back_line_thread) % 8) % (C_SMEM_STRIDE * 2 // THREAD_COPY_BYTES)) * (THREAD_COPY_BYTES // 2) + cs_vec)
                                Cg[gsi, gj] = Cs[gsi, sj]
                                # Cg[gsi, gj] = Cs[gsi, gj] # NOTE no Permute
    i_blockm, j_blockn = sch.get_loops(block=block_shared_C)[-2 : ]
    tir.TensorIntrin.register("Cs2g", C_s2g_desc, C_s2g_intrin)
    sch.tensorize(block_or_loop=i_blockm, tensor_intrin="Cs2g")
    
    
    # Local bind
    def shared_16x16_to_ldmatrix_32x8_layout(i, j):
        thread_id = 4 * (i % 8) + (j % 8) // 2
        return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)
    
    def ldmatrix_32x8_to_shared_16x16_layout(thread_id, local_id):
        row = 8 * (local_id % 4 // 2) + (thread_id // 4)
        col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
        return row, col
    
    @T.prim_func
    def A_s2r_intrin(A_reg: T.handle, A_smem: T.handle) -> None :
        Ar = T.match_buffer(param=A_reg, shape=(32, 8), dtype="float16", scope="warp", offset_factor=16)
        As = T.match_buffer(param=A_smem, shape=(16, 16), dtype="float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
        with T.block("root"):
            T.reads(As[0:16, 0:16])
            T.writes(Ar[0:32, 0:8])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("A_reg"):
                    si = T.axis.spatial(16, tx % 16)
                    sj = T.axis.spatial(16, (As.elem_offset % BLOCK_K
                                             + tx // 16 * 8 
                                             + tx % 16 % ARRAY_OFFSET_INSMEM // SMEM_BANK_ROWS * PERMUTED_OFFSET) % AB_SMEM_STRIDE)
                    # sj = T.axis.spatial(16, (As.elem_offset % BLOCK_K + tx // 16 * 8) % AB_SMEM_STRIDE)
                    vtx = T.axis.spatial(32, tx)
                    T.reads(As[0:16, 0:16])
                    T.writes(Ar[0:32, 0:8])
                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", Ar.data, Ar.elem_offset + 8 * vtx, T.tvm_access_ptr(T.type_annotation("float16"), As.data, As.elem_offset // BLOCK_K * BLOCK_K, As.strides[0] * 16, 1), As.strides[0] * si + sj)
                    
    i_warpm, j_mmak = sch.get_loops(block=block_local_A)[-2 : ]
    al_iter, i_mmam = sch.split(loop=i_warpm, factors=[None, MMA_M])
    index_map_A = lambda i, j : (i // 16, j // 16, *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16))
    sch.transform_layout(block=block_local_A, buffer=("write", 0), index_map=index_map_A)
    tir.TensorIntrin.register("As2r", tensor_intrin.cuda.get_ldmatrix_intrin(16, "float16", "A", False, "shared.dyn")[0], A_s2r_intrin)
    sch.tensorize(block_or_loop=i_mmam, tensor_intrin="As2r")
    
    @T.prim_func
    def B_s2r_intrin(B_reg: T.handle, B_smem: T.handle) -> None :
        Br = T.match_buffer(param=B_reg, shape=(32, 8), dtype="float16", scope="warp", offset_factor=16)
        Bs = T.match_buffer(param=B_smem, shape=(16, 16), dtype="float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
        with T.block("root"):
            T.reads(Bs[0:16, 0:16])
            T.writes(Br[0:32, 0:8])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("A_reg"):
                    si = T.axis.spatial(16, 8 * (tx // 16) + (tx % 8))
                    sj = T.axis.spatial(16, (Bs.elem_offset % BLOCK_K 
                                             + (tx % 16 // 8) * 8
                                             + tx % 16 % ARRAY_OFFSET_INSMEM // SMEM_BANK_ROWS * PERMUTED_OFFSET) % AB_SMEM_STRIDE)
                    vtx = T.axis.spatial(32, tx)
                    T.reads(Bs[0:16, 0:16])
                    T.writes(Br[0:32, 0:8])
                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", Br.data, Br.elem_offset + 8 * vtx, T.tvm_access_ptr(T.type_annotation("float16"), Bs.data, Bs.elem_offset // BLOCK_K * BLOCK_K, Bs.strides[0] * 16, 1), Bs.strides[0] * si + sj)
    
    @T.prim_func
    def B_s2r_intrin_x2(B_reg: T.handle, B_smem: T.handle) -> None :
        Br = T.match_buffer(param=B_reg, shape=(32, 8), dtype="float16", scope="warp", offset_factor=16)
        Bs = T.match_buffer(param=B_smem, shape=(16, 16), dtype="float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
        with T.block("root"):
            T.reads(Bs[0:16, 0:16])
            T.writes(Br[0:32, 0:8])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("A_reg"):
                    si_1 = T.axis.spatial(16, tx % 8)
                    si_2 = T.axis.spatial(16, MMA_N + tx % 8)
                    sj = T.axis.spatial(16, (Bs.elem_offset % BLOCK_K 
                                             + (tx // 8) % 2 * 8
                                             + tx % 16 % ARRAY_OFFSET_INSMEM // SMEM_BANK_ROWS * PERMUTED_OFFSET) % AB_SMEM_STRIDE)
                    vtx = T.axis.spatial(32, tx)
                    T.reads(Bs[0:16, 0:16])
                    T.writes(Br[0:32, 0:8])
                    T.ptx_ldmatrix("float16", T.bool(False), 2, ".b16", Br.data, Br.elem_offset + 8 * vtx, T.tvm_access_ptr(T.type_annotation("float16"), Bs.data, Bs.elem_offset // BLOCK_K * BLOCK_K, Bs.strides[0] * 16, 1), Bs.strides[0] * si_1 + sj)
                    T.ptx_ldmatrix("float16", T.bool(False), 2, ".b16", Br.data, Br.elem_offset + 8 * vtx + 4, T.tvm_access_ptr(T.type_annotation("float16"), Bs.data, Bs.elem_offset // BLOCK_K * BLOCK_K, Bs.strides[0] * 16, 1), Bs.strides[0] * si_2 + sj)
                    
                    
    
    @T.prim_func
    def B_s2r_intrin_woPermute(B_reg: T.handle, B_smem: T.handle) -> None :
        Br = T.match_buffer(param=B_reg, shape=(32, 8), dtype="float16", scope="warp", offset_factor=16)
        Bs = T.match_buffer(param=B_smem, shape=(16, 16), dtype="float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
        with T.block("root"):
            T.reads(Bs[0:16, 0:16])
            T.writes(Br[0:32, 0:8])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("A_reg"):
                    si = T.axis.spatial(16, 8 * (tx // 16) + (tx % 8))
                    sj = T.axis.spatial(16, (Bs.elem_offset % BLOCK_K 
                                             + (tx % 16 // 8) * 8) % AB_SMEM_STRIDE)
                    vtx = T.axis.spatial(32, tx)
                    T.reads(Bs[0:16, 0:16])
                    T.writes(Br[0:32, 0:8])
                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", Br.data, Br.elem_offset + 8 * vtx, T.tvm_access_ptr(T.type_annotation("float16"), Bs.data, Bs.elem_offset // BLOCK_K * BLOCK_K, Bs.strides[0] * 16, 1), Bs.strides[0] * si + sj)
    
    @T.prim_func
    def B_s2r_intrin_navie(B_reg: T.handle, B_smem: T.handle) -> None :
        Br = T.match_buffer(param=B_reg, shape=(32, 8), dtype="float16", scope="warp", offset_factor=16)
        Bs = T.match_buffer(param=B_smem, shape=(16, 16), dtype="float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
        with T.block("root"):
            T.reads(Bs[0:16, 0:16])
            T.writes(Br[0:32, 0:8])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("A_reg"):
                    si = T.axis.spatial(16, 8 * (tx // 16) + (tx % 8))
                    sj = T.axis.spatial(16, ((tx % 16 // 8) * 8) % AB_SMEM_STRIDE)
                    vtx = T.axis.spatial(32, tx)
                    T.reads(Bs[0:16, 0:16])
                    T.writes(Br[0:32, 0:8])
                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", Br.data, Br.elem_offset + 8 * vtx, T.tvm_access_ptr(T.type_annotation("float16"), Bs.data, Bs.elem_offset, Bs.strides[0] * 16, 1), Bs.strides[0] * si + sj)
    
    # 2 * m16n8k16 = m16n16k16
    i_warpn, j_mmak = sch.get_loops(block=block_local_B)[-2 : ]
    bl_iter, i_mman = sch.split(loop=i_warpn, factors=[None, MMA_N * 2 if MMA_N == 8 else MMA_N])
    index_map_B = lambda i, j : (i // 16, j // 16, *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16))
    sch.transform_layout(block=block_local_B, buffer=("write", 0), index_map=index_map_B)
    tir.TensorIntrin.register("Bs2r", tensor_intrin.cuda.get_ldmatrix_intrin(16, "float16", "B", True, "shared.dyn")[0], B_s2r_intrin_x2)
    sch.tensorize(block_or_loop=i_mman, tensor_intrin="Bs2r")

    @T.prim_func
    def C_r2s_intrin(C_reg: T.handle, C_smem: T.handle):
        Cr = T.match_buffer(C_reg, (32, 8), "float16", scope="warp", offset_factor=1)
        Cs = T.match_buffer(C_smem, (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
        with T.block("root"):
            T.reads(Cr[0:32, 0:8])
            T.writes(Cs[0:16, 0:16])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                for local_id in T.serial(8):
                    with T.block("C_shared"):
                        si = T.axis.spatial(16, 8 * (local_id % 4 // 2) + tx // 4)
                        sj = T.axis.spatial(16, 0 - Cs.elem_offset % Cs.strides[0]
                                                + Cs.elem_offset % Cs.strides[0] // BLOCK_COLS * BLOCK_COLS
                                                + (Cs.elem_offset % Cs.strides[0] % BLOCK_COLS 
                                                + 8 * (local_id // 4) + tx % 4 * 2 + local_id % 2
                                                + ((tx // 4) % 8) * PERMUTED_OFFSET) % C_SMEM_STRIDE)
                        vtx = T.axis.spatial(32, tx)
                        vlocal_id = T.axis.spatial(8, local_id)
                        Cs[si, sj] = Cr[vtx, vlocal_id]
    
    @T.prim_func
    def C_r2s_intrin_woPermute(C_reg: T.handle, C_smem: T.handle):
        Cr = T.match_buffer(C_reg, (32, 8), "float16", scope="warp", offset_factor=1)
        Cs = T.match_buffer(C_smem, (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
        with T.block("root"):
            T.reads(Cr[0:32, 0:8])
            T.writes(Cs[0:16, 0:16])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                for local_id in T.serial(8):
                    with T.block("C_shared"):
                        si = T.axis.spatial(16, 8 * (local_id % 4 // 2) + tx // 4)
                        sj = T.axis.spatial(16, 0 - Cs.elem_offset % Cs.strides[0]
                                                + Cs.elem_offset % Cs.strides[0] // BLOCK_COLS * BLOCK_COLS
                                                + (Cs.elem_offset % Cs.strides[0] % BLOCK_COLS 
                                                + 8 * (local_id // 4) + tx % 4 * 2 + local_id % 2
                                                ) % C_SMEM_STRIDE)
                        vtx = T.axis.spatial(32, tx)
                        vlocal_id = T.axis.spatial(8, local_id)
                        Cs[si, sj] = Cr[vtx, vlocal_id]
    @T.prim_func
    def C_r2s_intrin_failvectorize(C_reg: T.handle, C_smem: T.handle):
        Cr = T.match_buffer(C_reg, (32, 8), "float16", scope="warp", offset_factor=1)
        Cs = T.match_buffer(C_smem, (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
        with T.block("root"):
            T.reads(Cr[0:32, 0:8])
            T.writes(Cs[0:16, 0:16])
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                for local_id_div2 in T.serial(8 // 2):
                    for local_id_vec in T.vectorized(2):
                        with T.block("C_shared"):
                            si = T.axis.spatial(16, 8 * (local_id_div2 % 2) + tx // 4)
                            sj = T.axis.spatial(16, 0 - Cs.elem_offset % Cs.strides[0]
                                                    + Cs.elem_offset % Cs.strides[0] // BLOCK_COLS * BLOCK_COLS
                                                    + (Cs.elem_offset % Cs.strides[0] % BLOCK_COLS 
                                                    + 8 * (local_id_div2 // 2) + tx % 4 * 2 + local_id_vec
                                                    + ((tx // 4) % 8) * PERMUTED_OFFSET) % C_SMEM_STRIDE)
                            vtx = T.axis.spatial(32, tx)
                            vlocal_id = T.axis.spatial(8, local_id_div2 * 2 + local_id_vec)
                            Cs[si, sj] = Cr[vtx, vlocal_id]

    i_warpm, i_warpn = sch.get_loops(block=block_local_C)[-2 : ]
    cli_iter, i_mmam = sch.split(loop=i_warpm, factors=[None, MMA_M])
    clj_iter, j_mman = sch.split(loop=i_warpn, factors=[None, MMA_N * 2 if MMA_N == 8 else MMA_M])
    sch.reorder(cli_iter, clj_iter, i_mmam, j_mman)
    index_map_C = lambda i, j : (i // 16, j // 16, *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16))
    sch.transform_layout(block=block_local_C, buffer=("read", 0), index_map=index_map_C)
    tir.TensorIntrin.register("Cr2s", tensor_intrin.cuda.get_mma_store_intrin("float16", 8, "shared.dyn", False)[0], C_r2s_intrin)
    sch.tensorize(block_or_loop=i_mmam, tensor_intrin="Cr2s")
    
    # MMA compute
    mmai, mmaj, mmak = sch.get_loops(block=block_C)[-3 : ] 
    block_mma_inner = sch.blockize(target=mmai)
    sch.tensorize(block_or_loop=block_mma_inner, tensor_intrin="mma_f16f16f16_trans_b")
    
    # Init tensorize
    i_init, _ = sch.get_loops(block=block_C_init)[-2 : ]
    sch.tensorize(block_or_loop=i_init, tensor_intrin="mma_fill_16x16_f16")

    # Multi Stage
    sch.annotate(block_or_loop=k_smem_stage, ann_key="software_pipeline_stage", ann_val=[0, 0, K_STAGE - 2, K_STAGE - 1, K_STAGE - 1])
    sch.annotate(block_or_loop=k_smem_stage, ann_key="software_pipeline_order", ann_val=[2, 3, 0, 1, 4])
    
    sch.annotate(block_or_loop=k_smem_stage, ann_key="software_pipeline_async_stages", ann_val=[0])
    
    sch.annotate(block_or_loop=k_reg_stage, ann_key="software_pipeline_stage", ann_val=[0, 0, 0])
    sch.annotate(block_or_loop=k_reg_stage, ann_key="software_pipeline_order", ann_val=[0, 1, 2])


def cost_model(prof_dict, M, N, K):
    min_cost = float("inf")
    best_config = (1, 1, 1, 1, 1)
    for k, v in prof_dict.items():
        num_block = math.ceil(M / k[1]) * math.ceil(N / k[2]) * (k[5] + 1)
        w = math.ceil(num_block / 108)
        cost = w * v
        if min_cost > cost:
            min_cost = cost
            best_config = k
    return best_config

def get_Ampere_FP16_conv_Helix_result(prof_dict, batch, input_channel, H, W, output_channel, kH, kW, stride, pad, backend):
    M = batch * ((H + 2 * pad - kH) // stride + 1) * ((W + 2 * pad - kW) // stride + 1)
    N = output_channel
    K = input_channel * kH * kW
    best_config = cost_model(prof_dict, M, N, K)
    if backend == "cuda":
        cmd_im2col = f'{root_path}/build/bin_fp16/im2col {batch} {input_channel} {H} {W} {output_channel} {kH} {kW} {stride} {pad}'
        if best_config[0] == 0:
            cmd = f'{root_path}/build/bin_fp16/{"gemv" if best_config[5] == 0 else "gemv_splitK"} {M} {N} {K}'
        else:
            cmd = f'{root_path}/build/bin_fp16/{"gemm" if best_config[5] == 0 else "gemm_splitK"}_{best_config[0]}_{best_config[1]}_{best_config[2]}_{best_config[3]}_{best_config[4]} {math.ceil(M / best_config[1]) * best_config[1]} {math.ceil(N / best_config[2]) * best_config[2]} {K}'
        result_im2col = os.popen(cmd_im2col)
        result = os.popen(cmd)
        print(result_im2col.read(), result.read())
        cost = float(result_im2col.read().split()[0]) + float(result.read().split()[-8]) * 1e3
        tflops = 2 * M * N * K / 1e9 / cost
    else:
        #define K_STAGE 3 // 2 - 5
        K_STAGE = best_config[0]
        #define BLOCK_ROWS 256 //256  32
        BLOCK_ROWS = best_config[1]
        #define BLOCK_COLS 128 //128  32
        BLOCK_COLS = best_config[2]
        #define BLOCK_K    32 
        BLOCK_K = 32
        #define WARP_ROWS 64 // 128 64 32 16 
        WARP_ROWS = best_config[3]
        #define WARP_COLS 64 // 128 64 32 16
        WARP_COLS = best_config[4]
        #define BLOCK_STRIDE 1 // < M / BLOCK_ROWS
        BLOCK_STRIDE = 1
        #define MMA_M 16
        MMA_M = 16
        #define MMA_N 8
        MMA_N = 8
        #define MMA_K 16
        MMA_K = 16
        #define WARP_SIZE 32
        WARP_SIZE = 32
        #define THREAD_COPY_BYTES 16
        THREAD_COPY_BYTES = 16
        #define BLOCK_ROW_WARPS (BLOCK_COLS / WARP_COLS) // BLOCK_COLS / WARP_COLS 2
        BLOCK_ROW_WARPS = (BLOCK_COLS // WARP_COLS)
        #define BLOCK_COL_WARPS (BLOCK_ROWS / WARP_ROWS)  // BLOCK_ROWS / WARP_ROWS 4
        BLOCK_COL_WARPS = (BLOCK_ROWS // WARP_ROWS)
        #define BLOCK_ROW_TILES (BLOCK_COLS / MMA_N)  // BLOCK_COLS / MMA_N 16
        BLOCK_ROW_TILES = (BLOCK_COLS // MMA_N)
        #define BLOCK_COL_TILES (BLOCK_ROWS / MMA_M)  // BLOCK_ROWS / MMA_M 16
        BLOCK_COL_TILES = (BLOCK_ROWS // MMA_M)
        #define WARP_ROW_TILES (WARP_COLS / MMA_N)  // WARP_COLS / MMA_N 8
        WARP_ROW_TILES = (WARP_COLS // MMA_N)
        #define WARP_COL_TILES (WARP_ROWS / MMA_M)  // WARP_ROWS / MMA_M 4
        WARP_COL_TILES = (WARP_ROWS // MMA_M)
        #define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS 8
        WARPS_PER_BLOCK = (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
        #define THREADS_PER_BLOCK WARP_SIZE * WARPS_PER_BLOCK  // WARP_SIZE * WARPS_PER_BLOCK 256
        THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK
        #define CHUNK_K (BLOCK_K / MMA_K)  // 32 / MMA_K 2
        CHUNK_K = (BLOCK_K // MMA_K)
        #define CHUNK_LINE_BYTES (CHUNK_K * MMA_K * sizeof(half))          // CHUNK_K * MMA_K * sizeof(half) 64
        CHUNK_LINE_BYTES = (CHUNK_K * MMA_K * 2)
        #define CHUNK_COPY_LINES_PER_WARP (WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES)  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES 8
        CHUNK_COPY_LINES_PER_WARP = (WARP_SIZE * THREAD_COPY_BYTES // CHUNK_LINE_BYTES)
        #define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP 4
        CHUNK_COPY_LINE_LANES = (WARP_SIZE // CHUNK_COPY_LINES_PER_WARP)
        #define AB_SMEM_STRIDE (CHUNK_K * MMA_K)  // CHUNK_K * MMA_K 32
        AB_SMEM_STRIDE = (CHUNK_K * MMA_K)
        #define C_SMEM_STRIDE BLOCK_COLS  // BLOCK_COLS 128
        C_SMEM_STRIDE = BLOCK_COLS
        #define C_SMEM_OFFSET WARP_COLS   // WARP_COLS 64
        C_SMEM_OFFSET = WARP_COLS
        #define SMEM_BANK_ROWS (128 / (AB_SMEM_STRIDE * sizeof(half)))  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))
        SMEM_BANK_ROWS = (128 // (AB_SMEM_STRIDE * 2))
        #define SMEM_WARP_OFFSET (BLOCK_ROWS + BLOCK_COLS)
        SMEM_WARP_OFFSET = (BLOCK_ROWS + BLOCK_COLS)
        #define PERMUTED_OFFSET 8
        PERMUTED_OFFSET = 8
        #define PERMUTED_COLS 4
        PERMUTED_COLS = 4
        #define ARRAY_OFFSET_INSMEM (PERMUTED_COLS * SMEM_BANK_ROWS) 
        ARRAY_OFFSET_INSMEM = (PERMUTED_COLS * SMEM_BANK_ROWS)

        
        data, weight, _ = gemm.params
        sch = tir.Schedule(gemm.specialize(
        {
            data: tvm.tir.decl_buffer((M, K), dtype), weight: tvm.tir.decl_buffer((N, K), dtype),
        }))
        
        apply_trace(sch, K_STAGE=K_STAGE, BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS, BLOCK_K=BLOCK_K, WARP_ROWS=WARP_ROWS, WARP_COLS=WARP_COLS,
                    BLOCK_STRIDE=BLOCK_STRIDE, MMA_M=MMA_M, MMA_N=MMA_N, MMA_K=MMA_K, WARP_SIZE=WARP_SIZE, THREAD_COPY_BYTES=THREAD_COPY_BYTES, 
                    PERMUTED_OFFSET=PERMUTED_OFFSET, PERMUTED_COLS=PERMUTED_COLS, BLOCK_ROW_WARPS=BLOCK_ROW_WARPS, BLOCK_COL_WARPS=BLOCK_COL_WARPS,
                    BLOCK_ROW_TILES=BLOCK_ROW_TILES, BLOCK_COL_TILES=BLOCK_COL_TILES, WARP_ROW_TILES=WARP_ROW_TILES, WARP_COL_TILES=WARP_COL_TILES, 
                    WARPS_PER_BLOCK=WARPS_PER_BLOCK, THREADS_PER_BLOCK=THREADS_PER_BLOCK, CHUNK_K=CHUNK_K, CHUNK_LINE_BYTES=CHUNK_LINE_BYTES, 
                    CHUNK_COPY_LINES_PER_WARP=CHUNK_COPY_LINES_PER_WARP, CHUNK_COPY_LINE_LANES=CHUNK_COPY_LINE_LANES, AB_SMEM_STRIDE=AB_SMEM_STRIDE, 
                    C_SMEM_STRIDE=C_SMEM_STRIDE, C_SMEM_OFFSET=C_SMEM_OFFSET, SMEM_BANK_ROWS=SMEM_BANK_ROWS, SMEM_WARP_OFFSET=SMEM_WARP_OFFSET,
                    ARRAY_OFFSET_INSMEM= ARRAY_OFFSET_INSMEM)

        seq = tvm.transform.Sequential(
            [
                tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
                tvm.tir.transform.ConvertBlocksToOpaque(),
                tvm.tir.transform.UnifyThreadBinding(),
                tvm.tir.transform.LowerMatchBuffer(),
                tvm.tir.transform.CompactBufferAllocation(),
                tvm.tir.transform.InjectSoftwarePipeline(),
            ]
        )
        mod = seq(sch.mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            func = tvm.build(mod, target=target)
        evaluator = func.time_evaluator(func.entry_name, dev, number=100)
        d = np.random.rand(M, K).astype(dtype)
        w = np.random.rand(N, K).astype(dtype)
        Data = tvm.nd.array(d, dev)
        Weight = tvm.nd.array(w, dev)
        O = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
        cost = evaluator(Data, Weight, O).mean
        tflops = 2 * M * N * K / 1e12 / cost

    return cost, tflops

def get_Ampere_FP16_conv_cudnn_result(batch, input_channel, H, W, output_channel, kH, kW, stride, pad):
    cmd = f'{root_path}/build/bin_fp16/cudnn_fp16 {batch} {input_channel} {H} {W} {output_channel} {kH} {kW} {stride} {pad} 1'
    result = os.popen(cmd)
    cost = float(result.read().split()[-4])
    tflops = float(result.read().split()[-2])

    return cost, tflops
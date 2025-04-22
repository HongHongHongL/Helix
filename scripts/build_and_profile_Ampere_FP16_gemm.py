import os
from tqdm import tqdm

root_path = os.getcwd()
COMPILE_OPTIONS = "-O3 -std=c++17 -arch sm_80 -lcublas -w"

def compile_kernel(kernel_name, k_stage=None, block_rows=None, block_cols=None, warp_rows=None, warp_cols=None):
    output_name = kernel_name
    compile_cmd = f"nvcc {root_path}/gemm_template/Ampere_FP16/{kernel_name}.cu {COMPILE_OPTIONS}"
    
    if k_stage is not None:
        compile_cmd += f" -DK_STAGE={k_stage}"
        output_name += f"_{k_stage}"
    if block_rows is not None:
        compile_cmd += f" -DBLOCK_ROWS={block_rows}"
        output_name += f"_{block_rows}"
    if block_cols is not None:
        compile_cmd += f" -DBLOCK_COLS={block_cols}"
        output_name += f"_{block_cols}"
    if warp_rows is not None:
        compile_cmd += f" -DWARP_ROWS={warp_rows}"
        output_name += f"_{warp_rows}"
    if warp_cols is not None:
        compile_cmd += f" -DWARP_COLS={warp_cols}"
        output_name += f"_{warp_cols}"
    
    compile_cmd += f" -o {root_path}/build/bin_fp16/{output_name}"
    os.system(compile_cmd)

def is_valid_config(block_rows_scale, block_cols_scale, block_rows, block_cols, warp_rows, warp_cols):
    if block_rows_scale == 4 and block_cols_scale == 4:
        return False
    if block_rows > 256 or block_cols > 256:
        return False
    if warp_cols == warp_rows * 8 or warp_rows == warp_cols * 8:
        return False
    if block_cols == block_rows * 16 or block_rows == block_cols * 16:
        return False
    return True

def build_Helix_Ampere_FP16_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/bin_fp16"):
        os.makedirs(f"{root_path}/build/bin_fp16")

    print("Start building Ampere FP16 GEMM kernel ...")
    with tqdm(total=(4*4*4*3*3+1)*2, desc="building") as pbar:
        for K_STAGE in range(2, 6):
            for WARP_ROWS in [16, 32, 64, 128]:
                for WARP_COLS in [16, 32, 64, 128]:
                    for BLOCK_ROWS_SCALE in [1, 2, 4]:
                        for BLOCK_COLS_SCALE in [1, 2, 4]:
                            BLOCK_ROWS = BLOCK_ROWS_SCALE * WARP_ROWS
                            BLOCK_COLS = BLOCK_COLS_SCALE * WARP_COLS
                            
                            if not is_valid_config(BLOCK_ROWS_SCALE, BLOCK_COLS_SCALE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS):
                                pbar.update(2)
                                continue
                            
                            compile_kernel("gemm", K_STAGE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS)
                            pbar.update(1)
                            compile_kernel("gemm_splitK", K_STAGE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS)
                            pbar.update(1)

        compile_kernel("gemv")
        pbar.update(1)
        compile_kernel("gemv_splitK")
        pbar.update(1)
    print("Build finished.")

def profile_Helix_Ampere_FP16_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/prof_dict"):
        os.makedirs(f"{root_path}/build/prof_dict")

    prof_dict = {}
    print("Start profiling Ampere FP16 GEMM kernel ...")
    with tqdm(total=(4*4*4*3*3)*2, desc="profiling") as pbar:
        for K_STAGE in range(2, 6):
            for WARP_ROWS in [16, 32, 64, 128]:
                for WARP_COLS in [16, 32, 64, 128]:
                    for BLOCK_ROWS_SCALE in [1, 2, 4]:
                        for BLOCK_COLS_SCALE in [1, 2, 4]:
                            BLOCK_ROWS = BLOCK_ROWS_SCALE * WARP_ROWS
                            BLOCK_COLS = BLOCK_COLS_SCALE * WARP_COLS

                            if not is_valid_config(BLOCK_ROWS_SCALE, BLOCK_COLS_SCALE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS):
                                pbar.update(2)
                                continue

                            cmd = f"{root_path}/build/bin_fp16/gemm_{K_STAGE}_{BLOCK_ROWS}_{BLOCK_COLS}_{WARP_ROWS}_{WARP_COLS} {BLOCK_ROWS} {BLOCK_COLS} 4096"
                            with os.popen(cmd) as result:
                                cost = result.read().split()[-8]
                                prof_dict[(K_STAGE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS, 0)] = float(cost)
                                pbar.update(1)
                            
                            cmd_splitK = f"{root_path}/build/bin_fp16/gemm_splitK_{K_STAGE}_{BLOCK_ROWS}_{BLOCK_COLS}_{WARP_ROWS}_{WARP_COLS} {BLOCK_ROWS} {BLOCK_COLS} 4096"
                            with os.popen(cmd_splitK) as result_splitK:
                                cost_splitK = result_splitK.read().split()[-8]
                                prof_dict[(K_STAGE, BLOCK_ROWS, BLOCK_COLS, WARP_ROWS, WARP_COLS, 1)] = float(cost_splitK)
                                pbar.update(1)

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'w') as f:
        f.write(str(prof_dict))
    print("Profile finished.")

def build_cublas_FP16_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/bin_fp16"):
        os.makedirs(f"{root_path}/build/bin_fp16")

    compile_kernel("cublas_f16")

if __name__ == "__main__":

    build_Helix_Ampere_FP16_gemm_kernel()
    profile_Helix_Ampere_FP16_gemm_kernel()
    build_cublas_FP16_gemm_kernel()
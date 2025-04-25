import os
from tqdm import tqdm

root_path = os.getcwd()

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

def profile_Helix_Ampere_FP16_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/prof_dict"):
        os.makedirs(f"{root_path}/build/prof_dict")

    prof_dict = {}
    print("Start profiling Ampere FP16 GEMM kernel ...")
    with tqdm(total=(4*4*4*3*3+1)*2, desc="profiling") as pbar:
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

        cmd_gemv = f"{root_path}/build/bin_fp16/gemv 1 128 1024"
        with os.popen(cmd_gemv) as result_gemv:
            cost_gemv = result_gemv.read().split()[-8]
            prof_dict[(0, 1, 128, 0, 0, 0)] = float(cost_gemv)
            pbar.update(1)

        cmd_gemv_splitK = f"{root_path}/build/bin_fp16/gemv_splitK 1 128 1024"
        with os.popen(cmd_gemv_splitK) as result_gemv_splitK:
            cost_gemv_splitK = result_gemv_splitK.read().split()[-8]
            prof_dict[(0, 1, 128, 0, 0, 1)] = float(cost_gemv_splitK)
            pbar.update(1)

    with open(f'{root_path}/build/prof_dict/Ampere_FP16_cost_model.dict', 'w') as f:
        f.write(str(prof_dict))
    print("Profile finished.")

if __name__ == "__main__":

    profile_Helix_Ampere_FP16_gemm_kernel()

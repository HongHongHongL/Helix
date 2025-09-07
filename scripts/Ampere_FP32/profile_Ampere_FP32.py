import os
from tqdm import tqdm

root_path = os.getcwd()

def is_valid_config(BMt, BNt, BM, BN):
    if BMt * BNt * 32 >= 512:
        return False
    if BM > 128 or BN > 128 or BM < 4 or BN < 4:
        return False
    return True

def get_copy_value(base_value, threshold=4):
    return base_value if base_value < threshold else threshold

def profile_Helix_Ampere_FP32_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/prof_dict"):
        os.makedirs(f"{root_path}/build/prof_dict")

    prof_dict = {}
    print("Start profiling Ampere FP32 GEMM kernel ...")
    with tqdm(total=3*4*5*3*8*8, desc="profiling") as pbar:
        for WARP_ROW_THREAD in [1, 2, 4]:
            for TM in [1, 2, 4, 8]:
                for TN in [1, 2, 4, 8, 16]:
                    for BK in [8, 16, 32]:
                        for BMt in range(1, 9):
                            for BNt in range(1, 9):
                                BM = BMt * WARP_ROW_THREAD * TM
                                BN = BNt * (32 // WARP_ROW_THREAD) * TN

                                if not is_valid_config(BMt, BNt, BM, BN):
                                    pbar.update(1)
                                    continue

                                COPY_A = get_copy_value(TM)
                                COPY_B = get_copy_value(TN)

                                cmd = f"{root_path}/build/bin_fp32/sgemm_{BM}_{BN}_{BK}_{TM}_{TN}_{WARP_ROW_THREAD}_{COPY_A}_{COPY_B} {BM} {BN} 1024"
                                with os.popen(cmd) as result:
                                    try:
                                        cost = result.read().split()[-8]
                                    except:
                                        print(f"Can't use config {cmd}. Error: {cmd} {result.read()}")
                                        cost = float('inf')
                                    prof_dict[(BM, BN, BK, TM, TN, WARP_ROW_THREAD, COPY_A, COPY_B)] = float(cost)
                                    pbar.update(1)

    with open(f'{root_path}/build/prof_dict/Ampere_FP32_cost_model.dict', 'w') as f:
        f.write(str(prof_dict))
    print("Profile finished.")

if __name__ == "__main__":

    profile_Helix_Ampere_FP32_gemm_kernel()

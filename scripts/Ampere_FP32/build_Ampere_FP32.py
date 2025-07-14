import os
from tqdm import tqdm

root_path = os.getcwd()
COMPILE_OPTIONS = "-O3 -std=c++17 -arch sm_80 -lcublas -w"

def is_valid_config(BMt, BNt, BM, BN):
    if BMt * BNt * 32 >= 512:
        return False
    if BM > 128 or BN > 128 or BM < 4 or BN < 4:
        return False
    return True

def get_copy_value(base_value, threshold=4):
    return base_value if base_value < threshold else threshold

def compile_sgemm(BM, BN, BK, TM, TN, WARP_ROW_THREAD, COPY_A, COPY_B):
    output_name = f"sgemm_{BM}_{BN}_{BK}_{TM}_{TN}_{WARP_ROW_THREAD}_{COPY_A}_{COPY_B}"
    compile_cmd = (
        f"nvcc {root_path}/src_template/Ampere_FP32/sgemm.cu {COMPILE_OPTIONS} "
        f"-DBM={BM} -DBN={BN} -DBK={BK} "
        f"-DTM={TM} -DTN={TN} "
        f"-DWARP_ROW_THREAD={WARP_ROW_THREAD} "
        f"-DCOPY_A_SHM_REG_FLOAT={COPY_A} "
        f"-DCOPY_B_SHM_REG_FLOAT={COPY_B} "
        f"-o {root_path}/build/bin_fp32/{output_name}"
    )
    os.system(compile_cmd)

def compile_kernel(kernel_name):
    output_name = kernel_name
    compile_cmd = f"nvcc {root_path}/src_template/Ampere_FP32/{kernel_name}.cu {COMPILE_OPTIONS}"
    
    compile_cmd += f" -o {root_path}/build/bin_fp32/{output_name}"
    os.system(compile_cmd)

def build_Helix_Ampere_FP32_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/bin_fp32"):
        os.makedirs(f"{root_path}/build/bin_fp32")

    print("Start building Ampere FP32 GEMM kernel ...")
    with tqdm(total=3*4*5*3*8*8+1, desc="building") as pbar:
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

                                compile_sgemm(BM, BN, BK, TM, TN, WARP_ROW_THREAD, COPY_A, COPY_B)
                                pbar.update(1)
        compile_kernel("im2col")
        pbar.update(1)
    print("Build finished.")

def build_cublas_FP32_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/bin_fp32"):
        os.makedirs(f"{root_path}/build/bin_fp32")

    compile_kernel("cublas_f32")

def build_cudnn_FP32_gemm_kernel():
    if not os.path.exists(f"{root_path}/build/bin_fp32"):
        os.makedirs(f"{root_path}/build/bin_fp32")

    compile_kernel("cudnn_fp32")

if __name__ == "__main__":

    build_Helix_Ampere_FP32_gemm_kernel()
    build_cublas_FP32_gemm_kernel()
    build_cudnn_FP32_gemm_kernel()
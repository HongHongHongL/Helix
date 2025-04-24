# Helix

## How to build

### Git clone Helix repo

```
git clone https://github.com/HongHongHongL/Helix.git
cd Helix
export PYTHONPATH=$PYTHONPATH:path_to_Helix/scripts
```

### Build and profile kernel

#### Ampere FP16

```
python3 scripts/Ampere_FP16/build_Ampere_FP16_gemm.py
python3 scripts/Ampere_FP16/profile_Ampere_FP16_gemm.py
```

#### Ampere FP32

```
python3 scripts/Ampere_FP32/build_Ampere_FP32_gemm.py
python3 scripts/Ampere_FP32/profile_Ampere_FP32_gemm.py
```

#### x86 CPU

```
python3 scripts/x86_CPU/profile_x86_CPU_gemm.py
```

#### ARM CPU

```
python3 scripts/ARM_CPU/profile_ARM_CPU_gemm.py
```

### Test experiment

#### Ampere FP16

```
python3 scripts/Ampere_FP16/benchmark_Ampere_FP16_gemm.py
python3 scripts/Ampere_FP16/benchmark_Ampere_FP16_conv.py
python3 scripts/Ampere_FP16/benchmark_Ampere_FP16_model_level_LLM.py
python3 scripts/Ampere_FP16/benchmark_Ampere_FP16_model_level_CNN.py
```

#### Ampere FP32

```
python3 scripts/Ampere_FP32/benchmark_Ampere_FP32_gemm.py
python3 scripts/Ampere_FP32/benchmark_Ampere_FP32_conv.py
python3 scripts/Ampere_FP32/benchmark_Ampere_FP32_model_level_LLM.py
python3 scripts/Ampere_FP32/benchmark_Ampere_FP32_model_level_CNN.py
```

#### x86 CPU

```
python3 scripts/x86_CPU/benchmark_x86_CPU_gemm.py
python3 scripts/x86_CPU/benchmark_x86_CPU_conv.py
python3 scripts/x86_CPU/benchmark_x86_CPU_model_level_LLM.py
python3 scripts/x86_CPU/benchmark_x86_CPU_model_level_CNN.py
```

#### ARM CPU

```
python3 scripts/ARM_CPU/benchmark_ARM_CPU_gemm.py
python3 scripts/ARM_CPU/benchmark_ARM_CPU_conv.py
python3 scripts/ARM_CPU/benchmark_ARM_CPU_model_level_LLM.py
python3 scripts/ARM_CPU/benchmark_ARM_CPU_model_level_CNN.py
```

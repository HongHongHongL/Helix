# Helix

## How to build

### Git clone Helix repo

```
git clone https://github.com/HongHongHongL/Helix.git
cd Helix
export PYTHONPATH=$PYTHONPATH:path_to_Helix/scripts
```

### Build kernel

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

### Test experiment

#### Ampere FP16

```
python3 scripts/Ampere_FP16/benchmark_Ampere_FP16_gemm.py
```

#### Ampere FP32

```
python3 scripts/Ampere_FP32/benchmark_Ampere_FP32_gemm.py
```

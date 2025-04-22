# Helix

## How to build

### Git clone Helix repo

```
git clone https://github.com/HongHongHongL/Helix.git
cd Helix
```

### Build kernel

```
python3 scripts/build_and_profile_Ampere_FP16_gemm.py
```

### Test experiment

```
python3 scripts/benchmark_Ampere_FP16_gemm.py
```

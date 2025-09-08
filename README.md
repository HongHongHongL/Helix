# Helix

## How to build

### Git clone Helix repo

```
git clone https://github.com/HongHongHongL/Helix.git
cd Helix
export PYTHONPATH=$PYTHONPATH:path_to_Helix/scripts
```

### Build the docker image

```
docker build -f dockerfile.cpu -t helix_cpu:latest .
docker build -f dockerfile.gpu -t helix_gpu:latest .
```

### Build and profile kernel

#### Ampere FP16

```
python3 scripts/Ampere_FP16/build_Ampere_FP16.py
python3 scripts/Ampere_FP16/profile_Ampere_FP16.py
```

#### Ampere FP32

```
python3 scripts/Ampere_FP32/build_Ampere_FP32.py
python3 scripts/Ampere_FP32/profile_Ampere_FP32.py
```

#### x86 CPU

```
python3 scripts/x86_CPU/profile_x86_CPU.py
```

##### oneDNN

https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/driver_conv.md

#### ARM CPU

```
python3 scripts/ARM_CPU/profile_ARM_CPU.py
```

##### ACL

https://github.com/ARM-software/ComputeLibrary/blob/main/docs/user_guide/how_to_build_and_run_examples.dox

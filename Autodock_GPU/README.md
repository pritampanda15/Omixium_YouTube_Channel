# AutoDock-GPU Build Instructions
# ------------------------------------------------------
# By Dr. Pritam Kumar Panda, Stanford University
# YouTube: https://www.youtube.com/@Omixium_ai 
# ------------------------------------------------------

## Environment Setup

Before building, set the CUDA environment variables:

```bash
export GPU_INCLUDE_PATH=/usr/local/cuda-12.4/include
export GPU_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
```

---

## CUDA Compute Capability

In CUDA, **compute capability** defines the features and capabilities of a GPU. It is specified as a number (e.g., `5.0`, `8.0`). In Makefiles, the dot is often omitted:

* `50` → `5.0`
* `80` → `8.0`

| Compute Capability | GPU Architecture (Examples)   | Notes                                                               |
| ------------------ | ----------------------------- | ------------------------------------------------------------------- |
| 5.0 (`50`)         | Maxwell (e.g., GTX 750, 980)  | Older GPUs, no Tensor cores.                                        |
| 8.0 (`80`)         | Ampere (e.g., RTX 3090, A100) | Modern GPUs, supports Tensor cores, faster FP16, sparse operations. |

### Why the Makefile sets `MIN_COMPUTE`:

* `MIN_COMPUTE=50` → Allows older GPUs to compile and run.
* `MIN_COMPUTE=80` → Ensures newer GPUs with **Tensor cores** are used when `TENSOR=ON`. Tensor cores accelerate mixed-precision matrix math for deep learning.

**Key Takeaway:**

* Older GPUs (Maxwell, Pascal) → `MIN_COMPUTE=50`
* Tensor core acceleration → GPU must support compute capability ≥ `8.0` (`80`)

---

## Makefile

This is the primary Makefile for AutoDock-GPU compilation:

```makefile
# AutoDock-GPU Makefile

# ------------------------------------------------------
# Note: Environment variables must be defined before compiling
# DEVICE? (CPU, GPU, CUDA, OCLGPU)
# ------------------------------------------------------
OVERLAP = ON

ifeq ($(DEVICE), $(filter $(DEVICE),GPU CUDA))
MIN_COMPUTE:=50
ifeq ($(TENSOR), ON)
MIN_COMPUTE:=80
endif
TARGETS_SUPPORTED := $(shell ./test_cuda.sh nvcc "$(GPU_INCLUDE_PATH)" "$(GPU_LIBRARY_PATH)" "$(TARGETS)" "$(MIN_COMPUTE)")
# if user specifies DEVICE=GPU the test result determines wether CUDA will be used or not
ifeq ($(TARGETS_SUPPORTED),)
ifeq ($(DEVICE),CUDA)
$(error Cuda verification failed)
else
$(info Cuda is not available, using OpenCL)
$(info )
override DEVICE:=GPU
export
endif
else
#override TARGETS:=$(TARGETS_SUPPORTED)
export
override DEVICE:=CUDA
endif
endif
ifeq ($(DEVICE),CUDA)
override DEVICE:=GPU
export
include Makefile.Cuda
else
ifeq ($(DEVICE),$(filter $(DEVICE),OCLGPU OPENCL))
override DEVICE:=GPU
export
$(info Using OpenCL)
$(info )
endif
$(info Please make sure to set environment variables)
$(info GPU_INCLUDE_PATH and GPU_LIBRARY_PATH)
$(info )
include Makefile.OpenCL
endif
```

---

## Makefile.Cuda

This file handles CUDA kernel compilation and host binary building:

```makefile
# AutoDock-GPU CUDA Makefile

NVCC = nvcc
CPP = g++
UNAME := $(shell uname)

# List of supported compute capabilities
TARGETS = 50 52 53 60 61 62 70 72 75 80 86 87 89 90

ifeq ($(TENSOR), ON)
    NVTENSOR=-DUSE_NVTENSOR
    NVTENSOR+=-I./wmma_extension/include
    TARGETS = 80
endif

# Correct syntax for CUDA_TARGETS
CUDA_TARGETS=$(foreach target,$(TARGETS),-gencode arch=compute_$(target),code=sm_$(target))

ifeq ($(DEVICE), CPU)
    DEV =-DCPU_DEVICE
else ifeq ($(DEVICE), GPU)
    DEV =-DGPU_DEVICE
endif

# ------------------------------------------------------
# Project directories
# opencl_lvs: wrapper for OpenCL APIs
COMMON_DIR=./common
HOST_INC_DIR=./host/inc
HOST_SRC_DIR=./host/src
KRNL_DIR=./cuda
KCMN_DIR=$(COMMON_DIR)
BIN_DIR=./bin
LIB_CUDA = kernels.o -lcurand -lcudart -DUSE_CUDA

TARGET := autodock
TOOL_TARGET := adgpu_analysis

IFLAGS=-I$(COMMON_DIR) -I$(HOST_INC_DIR) -I$(GPU_INCLUDE_PATH) -I$(KRNL_DIR)
LFLAGS=-L$(GPU_LIBRARY_PATH) -Wl,-rpath=$(GPU_LIBRARY_PATH):$(CPU_LIBRARY_PATH) -lstdc++fs
CFLAGS=-std=c++17 $(IFLAGS) $(LFLAGS)
TOOL_CFLAGS=-std=c++17 -I$(COMMON_DIR) -I$(HOST_INC_DIR) $(LFLAGS) 
CFLAGS=-std=c++17 $(IFLAGS) $(LFLAGS)

ifeq ($(DEVICE), CPU)
	TARGET:=$(TARGET)_cpu
else ifeq ($(DEVICE), GPU)
	NWI=-DN64WI
	TARGET:=$(TARGET)_gpu
endif

ifeq ($(OVERLAP), ON)
	PIPELINE=-DUSE_PIPELINE -fopenmp
endif

BIN := $(wildcard $(TARGET)*)

# ------------------------------------------------------
# Number of work-items (wi)
# Valid values: 32, 64, 128, 256
NUMWI=

ifeq ($(NUMWI), 32)
ifeq ($(TENSOR), ON)
$(error NUMWI needs to be at least 64 with TENSOR=ON)
endif
	NWI=-DN32WI
	TARGET:=$(TARGET)_32wi
else ifeq ($(NUMWI), 64)
	NWI=-DN64WI
	TARGET:=$(TARGET)_64wi
else ifeq ($(NUMWI), 128)
	NWI=-DN128WI
	TARGET:=$(TARGET)_128wi
else ifeq ($(NUMWI), 256)
		NWI=-DN256WI
		TARGET:=$(TARGET)_256wi
else
	ifeq ($(DEVICE), CPU)
		NWI=-DN16WI
		TARGET:=$(TARGET)_16wi
	else ifeq ($(DEVICE), GPU)
		NWI=-DN128WI
		TARGET:=$(TARGET)_128wi
	endif
endif

# ------------------------------------------------------
# Configuration
# FDEBUG (full) : enables debugging on both host + device
# LDEBUG (light): enables debugging on host
# RELEASE
CONFIG=RELEASE
#CONFIG=FDEBUG

ifeq ($(CONFIG),FDEBUG)
	OPT =-O0 -g3 -Wall -DDOCK_DEBUG
	CUDA_FLAGS = -G -use_fast_math --ptxas-options="-v" $(CUDA_TARGETS) -std=c++11
else ifeq ($(CONFIG),LDEBUG)
	OPT =-O0 -g3 -Wall
	CUDA_FLAGS = -use_fast_math --ptxas-options="-v" $(CUDA_TARGETS) -std=c++11
else ifeq ($(CONFIG),RELEASE)
	OPT =-O3
	CUDA_FLAGS = -use_fast_math --ptxas-options="-v" $(CUDA_TARGETS) -std=c++11
else
	OPT =
	CUDA_FLAGS = -use_fast_math --ptxas-options="-v" $(CUDA_TARGETS) -std=c++11
endif

# ------------------------------------------------------
# Reproduce results (remove randomness)
REPRO=NO

ifeq ($(REPRO),YES)
	REP =-DREPRO
else
	REP =
endif
# ------------------------------------------------------

all: otool odock

check-env-dev:
	@if test -z "$$DEVICE"; then \
		echo "Please set DEVICE to either CPU, GPU, CUDA, or OCLGPU to build docking software."; \
		exit 1; \
	else \
		if [ "$$DEVICE" = "CPU" ]; then \
			echo "DEVICE is set to $$DEVICE which is not a valid Cuda device."; \
			exit 1; \
		else \
			if [ "$$DEVICE" = "GPU" ]; then \
				echo "DEVICE is set to $$DEVICE"; \
			else \
				echo "DEVICE value is invalid. Please set DEVICE to either CPU, GPU, CUDA, or OCLGPU"; \
				exit 1; \
			fi; \
		fi; \
	fi; \
	echo " "

check-env-cpu:
	@if test -z "$$CPU_INCLUDE_PATH"; then \
		echo "CPU_INCLUDE_PATH is undefined"; \
	else \
		echo "CPU_INCLUDE_PATH is set to $$CPU_INCLUDE_PATH"; \
	fi; \
	if test -z "$$CPU_LIBRARY_PATH"; then \
		echo "CPU_LIBRARY_PATH is undefined"; \
	else \
		echo "CPU_LIBRARY_PATH is set to $$CPU_LIBRARY_PATH"; \
	fi; \
	echo " "

check-env-gpu:
	@if test -z "$$GPU_INCLUDE_PATH"; then \
		echo "GPU_INCLUDE_PATH is undefined"; \
	else \
		echo "GPU_INCLUDE_PATH is set to $$GPU_INCLUDE_PATH"; \
	fi; \
	if test -z "$$GPU_LIBRARY_PATH"; then \
		echo "GPU_LIBRARY_PATH is undefined"; \
	else \
		echo "GPU_LIBRARY_PATH is set to $$GPU_LIBRARY_PATH"; \
	fi; \
	echo " "

check-env-all: check-env-dev check-env-cpu check-env-gpu

# ------------------------------------------------------
# Priting out its git version hash

GIT_VERSION := $(shell ./version_string.sh)

CFLAGS+=-DVERSION=\"$(GIT_VERSION)\"
TOOL_CFLAGS+=-DVERSION=\"$(GIT_VERSION)\"

# ------------------------------------------------------

kernels: $(KERNEL_SRC)
	$(NVCC) $(NWI) $(REP) $(CUDA_FLAGS) $(IFLAGS) $(CUDA_INCLUDES) $(NVTENSOR) -c $(KRNL_DIR)/kernels.cu

otool:
	@echo "Building" $(TOOL_TARGET) "..."
	$(CPP) \
	$(shell ls $(HOST_SRC_DIR)/*.cpp) \
	$(TOOL_CFLAGS) \
	-o$(BIN_DIR)/$(TOOL_TARGET) \
	$(PIPELINE) $(NVTENSOR) $(OPT) -DTOOLMODE $(REP)

odock: check-env-all kernels
	@echo "Building" $(TARGET) "..."
	$(CPP) \
	$(shell ls $(HOST_SRC_DIR)/*.cpp) \
	$(CFLAGS) \
	$(LIB_CUDA) \
	-o$(BIN_DIR)/$(TARGET) \
	$(DEV) $(NWI) $(PIPELINE) $(NVTENSOR) $(OPT) $(DD) $(REP) $(KFLAGS)

# Example
# 1ac8: for testing gradients of translation and rotation genes
# 7cpa: for testing gradients of torsion genes (15 torsions) 
# 3tmn: for testing gradients of torsion genes (1 torsion)

PDB      := 3ce3
NRUN     := 100
NGEN     := 27000
POPSIZE  := 150
TESTNAME := test
TESTLS   := ad

test: odock
	$(BIN_DIR)/$(TARGET) \
	-ffile ./input/$(PDB)/derived/$(PDB)_protein.maps.fld \
	-lfile ./input/$(PDB)/derived/$(PDB)_ligand.pdbqt \
	-nrun $(NRUN) \
	-ngen $(NGEN) \
	-psize $(POPSIZE) \
	-resnam $(TESTNAME) \
	-gfpop 0 \
	-lsmet $(TESTLS)

.PHONY: clean
```

---

## Build Instructions

```bash
# Build GPU target with different work-group sizes
make DEVICE=GPU NUMWI=256
make DEVICE=GPU NUMWI=128
make DEVICE=GPU NUMWI=64
```

**Notes:**

* `DEVICE` → `CPU`, `GPU`, `CUDA`, `OCLGPU`, `OPENCL`
* `NUMWI` → Work-group/thread block size (e.g., `32`, `64`, `128`, `256`)
* Host binaries are placed under `./bin` as:

```
autodock_<DEVICE>_<NUMWI>wi
```

* Example: `autodock_GPU_128wi`

**Hints:**

* Choose `NUMWI` based on GPU and workload.
* For modern GPUs: try `128` or `64`.
* On macOS CPU builds: use `NUMWI=1`.

---

This README now **contains everything needed to compile AutoDock-GPU**, including the Makefile and Makefile.Cuda, with clear guidance on environment setup, GPU compute capability, and build instructions.

---


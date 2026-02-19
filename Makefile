NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublasLt -lcublas -lcuda
OUT_DIR = out
ARCH ?= sm_80
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
# Skip first 1000 launches to avoid warmup overhead
NCU_COMMAND = $(NCU_PATH) --set full --import-source yes --launch-skip 1000 

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=$(ARCH) -DSM_ARCH=$(ARCH_NUM)

# Include Cutlass/Cute headers
NVCC_FLAGS += -Ithird_party/cutlass/include

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

matmul_bench: matmul.cu 
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmul_profile: matmul_bench
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

python_profile:
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-own -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-example --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-torch -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-own --no-cute-example
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-cute-example -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-own --no-cublas

python_profile_all:
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-own-1024 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 1024,1024,1024 --no-cute-example --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-torch-1024 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 1024,1024,1024 --no-cute-own --no-cute-example
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-cute-example-1024 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 1024,1024,1024 --no-cute-own --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-own-2048 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-example --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-torch-2048 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-own --no-cute-example
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-cute-example-2048 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 2048,2048,2048 --no-cute-own --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-own-4096 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 4096,4096,4096 --no-cute-example --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-torch-4096 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 4096,4096,4096 --no-cute-own --no-cute-example
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-cute-example-4096 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 4096,4096,4096 --no-cute-own --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-own-8192 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 8192,8192,8192 --no-cute-example --no-cublas
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-torch-8192 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 8192,8192,8192 --no-cute-own --no-cute-example
	$(NCU_PATH) --set full --import-source yes --launch-skip 1000 --launch-count 20 -o profiling/cute_dsl_matmul_profile-cute-example-8192 -f uv run python matmuls/cute_mma_matmul_3stage_pip_grid_swizzle_dsl.py --benchmark --mnk 8192,8192,8192 --no-cute-own --no-cublas


clean:
	rm $(OUT_DIR)/*

print_copy: copy/print_copy.cpp
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ -o $(OUT_DIR)/$@
	./out/print_copy > ./out/print_copy.tex
	pdflatex out/print_copy.tex

print_layout: copy/print_layout.cpp
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ -o $(OUT_DIR)/$@
	./out/print_layout > ./out/print_layout.tex
	pdflatex out/print_layout.tex

print_mma: matmuls/print_mma.cpp
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ -o $(OUT_DIR)/$@
	./out/print_mma > ./out/print_mma.tex
	pdflatex out/print_mma.tex

copy_bench: copy.cu
	rm -f $(OUT_DIR)/$@
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

copy_profile: copy_bench
	$(NCU_COMMAND) -o profiling/copy_bench_profile.ncu-rep -f $(OUT_DIR)/$^
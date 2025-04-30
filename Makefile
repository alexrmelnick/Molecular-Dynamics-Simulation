# GPU build: compile all CUDA & C sources together
run_gpu: 
	nvcc -arch=sm_86 -O1 main_gpu.cu -o run_gpu

# CPU-only build: compile/link all CPU sources
run_cpu: 
	gcc main_cpu.c -lrt -lm -fopenmp -mavx -O1 -o run_cpu

# NOTE: For some reason, NVCC -O1 runs about 50% faster than GCC -O1 for the same code.
# OpenMP build: compile/link all CPU sources for OpenMP tests
run_openmp:
	gcc run_openmp.c -fopenmp -lrt -lm -O1 -o run_openmp

clean:
	rm -f run_all run_cpu run_openmp

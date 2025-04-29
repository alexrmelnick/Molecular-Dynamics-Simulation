# GPU build: compile all CUDA & C sources together
run_all: 
	nvcc -arch=sm_86 -Xcompiler -fopenmp -O1 main.cu -o run_all

# CPU-only build: compile/link all CPU sources
run_cpu: 
	gcc main_cpu.c -lrt -lm -mavx -O1 -o run_cpu

# NOTE: For some reason, NVCC -O1 runs about 50% faster than GCC -O1 for the same code.

clean:
	rm -f run_all run_cpu
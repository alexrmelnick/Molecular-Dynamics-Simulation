run_all: main.cu
	nvcc -arch=sm_86 -Xcompiler -fopenmp -O1 main.cu -o run_all

clean: 
	rm run_all

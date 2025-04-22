cpu_cell: serial_cell.c
	gcc serial_cell.c -lrt -lm -o cell
	./cell < input.in

cpu_trivial: serial_trivial.c
	gcc serial_trivial.c -lm -o serial
	./serial < input.in

gpu_cell: GPU_cell.cu
	nvcc GPU_cell.cu -o gcell

gpu_trivial: GPU_trivial.cu
	nvcc GPU_trivial.cu -o gserial

run_all: main.c
	gcc main.c -lrt -lm -o run_all

all: cpu_cell cpu_trivial gpu_cell gpu_trivial run_all

clean: 
	rm serial cell gserial gcell run_all



0.055000  0.844935 -5.088893 -3.821491
 0.110000  0.429956 -4.465322 -3.820388
 0.165000  0.535316 -4.624309 -3.821336
 0.220000  0.512052 -4.589109 -3.821032
 0.275000  0.525341 -4.609158 -3.821147
 0.330000  0.527853 -4.612893 -3.821114
 0.385000  0.532143 -4.619321 -3.821106
 0.440000  0.540697 -4.632173 -3.821128
 0.495000  0.530670 -4.617102 -3.821097

 0.055000  0.844935 -5.088894 -3.821491
 0.110000  0.429956 -4.465322 -3.820388
 0.165000  0.535316 -4.624309 -3.821336
 0.220000  0.512052 -4.589109 -3.821032
 0.275000  0.525341 -4.609158 -3.821147
 0.330000  0.527853 -4.612893 -3.821114
 0.385000  0.532143 -4.619321 -3.821106
 0.440000  0.540697 -4.632173 -3.821128
 0.495000  0.530670 -4.617103 -3.821097
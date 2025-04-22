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

all: cpu_cell cpu_trivial gpu_cell gpu_trivial

clean: 
	rm serial cell gserial gcell
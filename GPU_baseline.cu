/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Baseline Implementation in CUDA!
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


int gpu_base()
{
	// int stepCount;
	for (stepCount = 1; stepCount <= STEPLIMIT; stepCount++)
	{
		SingleStepBaseCUDA();
		if (stepCount % STEPAVG == 0)
			EvalPropsBase(); // Note that this uses the CPU version of EvalPropsBase since we are only using this to verify the correctness of the GPU code.
	}
	return 0;
}

/*----------------------------------------------------------------------------*/
void ComputeAccelBaseCUDA()
{
	/*------------------------------------------------------------------------------

	Sets up and launches the kernel to compute accelerations, ra, on the GPU.

	------------------------------------------------------------------------------*/
	size_t bytes = nAtom * 3 * sizeof(double);
	double *d_r = nullptr, *d_ra = nullptr;

	cudaMalloc(&d_r, bytes);
	cudaMalloc(&d_ra, bytes);
	cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);
	cudaMemset(d_ra, 0, bytes);

	int grid = (nAtom + BLOCK_SIZE - 1) / BLOCK_SIZE;
	ComputeAccelBaseKernel<<<grid, BLOCK_SIZE>>>(
		d_r,
		d_ra,
		nAtom,
		make_double3(RegionH[0], RegionH[1], RegionH[2]),
		Duc,
		Uc);
	cudaDeviceSynchronize();

	cudaMemcpy(ra, d_ra, bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_r);
	cudaFree(d_ra);
}

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
void SingleStepBaseCUDA()
{
	/*------------------------------------------------------------------------------
		r & rv are propagated by DeltaT in time using the velocity-Verlet method.
	------------------------------------------------------------------------------*/
	int n, k;

	HalfKickBaseCUDA();	  /* First half kick to obtain v(t+Dt/2) */
	UpdatePositionCUDA(); /* Update atomic coordinates to r(t+Dt) */
	ApplyBoundaryCondBaseCUDA();
	ComputeAccelBaseCUDA(); /* Computes new accelerations, a(t+Dt) */
	HalfKickBaseCUDA();		/* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickBaseCUDA()
{
	/*------------------------------------------------------------------------------
		Sets up and launches the kernel to compute the half kick on the GPU.
	------------------------------------------------------------------------------*/
	size_t bytes = nAtom * 3 * sizeof(double);
	double *d_rv = nullptr, *d_ra = nullptr;

	cudaMalloc(&d_rv, bytes);
	cudaMalloc(&d_ra, bytes);
	cudaMemcpy(d_rv, rv, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ra, ra, bytes, cudaMemcpyHostToDevice);

	int grid = (nAtom + BLOCK_SIZE - 1) / BLOCK_SIZE;
	HalfKickBaseKernel<<<grid, BLOCK_SIZE>>>(
		d_rv,
		d_ra,
		nAtom,
		DeltaTH);
	cudaDeviceSynchronize();

	cudaMemcpy(rv, d_rv, bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_rv);
	cudaFree(d_ra);
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondBaseCUDA()
{
	/*------------------------------------------------------------------------------
		Sets up and launches the kernel to apply periodic boundary conditions on the GPU.
	------------------------------------------------------------------------------*/
	size_t bytes = nAtom * 3 * sizeof(double);
	double *d_r = nullptr;

	cudaMalloc(&d_r, bytes);
	cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);

	int grid = (nAtom + BLOCK_SIZE - 1) / BLOCK_SIZE;
	ApplyBoundaryCondKernel<<<grid, BLOCK_SIZE>>>(
		d_r,
		nAtom,
		make_double3(Region[0], Region[1], Region[2]),
		make_double3(RegionH[0], RegionH[1], RegionH[2]));
	cudaDeviceSynchronize();

	cudaMemcpy(r, d_r, bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_r);
}

void UpdatePositionCUDA()
{
    size_t bytes = nAtom * 3 * sizeof(double);
    double *d_r = nullptr, *d_rv = nullptr;

    // Allocate & copy data to device
    cudaMalloc(&d_r,  bytes);
    cudaMalloc(&d_rv, bytes);
	cudaMemcpy(d_r,  r,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rv, rv, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int grid = (nAtom + BLOCK_SIZE - 1) / BLOCK_SIZE;
    UpdatePositionKernel<<<grid, BLOCK_SIZE>>>(
        d_r,      // device positions
        d_rv,     // device velocities
        nAtom,    // atom count
        DELTAT    // timestep
    );
    cudaDeviceSynchronize();

    // Copy results back and free
    cudaMemcpy(r, d_r, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_r);
    cudaFree(d_rv);
}

// Compute accelerations (Baseline O(N^2))
__global__ void ComputeAccelBaseKernel(
	const double *r,
	double *ra,
	int nAtom,
	double3 RegionH,
	double Duc,
	double Uc)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;
	if (j1 >= nAtom)
		return;

	// Local accumulators
	double ax = 0.0, ay = 0.0, az = 0.0;

	// Iterate over all other atoms
	for (int j2 = 0; j2 < nAtom; ++j2)
	{
		if (j2 == j1)
			continue;
		// Compute displacement vector
		double dx = r[j1 * 3] - r[j2 * 3];
		double dy = r[j1 * 3 + 1] - r[j2 * 3 + 1];
		double dz = r[j1 * 3 + 2] - r[j2 * 3 + 2];

		// Periodic boundary (nearest image)
		dx = dx - signR_device(RegionH.x, dx - RegionH.x) - signR_device(RegionH.x, dx + RegionH.x);
		dy = dy - signR_device(RegionH.y, dy - RegionH.y) - signR_device(RegionH.y, dy + RegionH.y);
		dz = dz - signR_device(RegionH.z, dz - RegionH.z) - signR_device(RegionH.z, dz + RegionH.z);

		double rr = dx * dx + dy * dy + dz * dz;
		// Within cutoff
		if (rr < RCUT * RCUT)
		{
			double ri2 = 1.0 / rr;
			double ri6 = ri2 * ri2 * ri2;
			double r1 = sqrt(rr);
			double fcVal = 48.0 * ri2 * ri6 * (ri6 - 0.5) + Duc / r1;
			ax += fcVal * dx;
			ay += fcVal * dy;
			az += fcVal * dz;
		}
	}
	// Write back
	ra[j1 * 3] = ax;
	ra[j1 * 3 + 1] = ay;
	ra[j1 * 3 + 2] = az;
}

// Velocity half-kick (v += deltat/2 * a)
__global__ void HalfKickBaseKernel(
	double *rv,
	const double *ra,
	int nAtom,
	double DeltaTH)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nAtom)
		return;
	rv[idx * 3] += DeltaTH * ra[idx * 3];
	rv[idx * 3 + 1] += DeltaTH * ra[idx * 3 + 1];
	rv[idx * 3 + 2] += DeltaTH * ra[idx * 3 + 2];
}

// Position update (r += deltat * v)
__global__ void UpdatePositionKernel(
	double *r,
	const double *rv,
	int nAtom,
	double deltaT)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nAtom)
		return;
	r[idx * 3] += deltaT * rv[idx * 3];
	r[idx * 3 + 1] += deltaT * rv[idx * 3 + 1];
	r[idx * 3 + 2] += deltaT * rv[idx * 3 + 2];
}

// Apply periodic boundary conditions
__global__ void ApplyBoundaryCondKernel(
	double *r,
	int nAtom,
	double3 Region,
	double3 RegionH)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nAtom)
		return;
	// X
	double x = r[idx * 3];
	x = x - signR_device(RegionH.x, x) - signR_device(RegionH.x, x - Region.x);
	r[idx * 3] = x;
	// Y
	double y = r[idx * 3 + 1];
	y = y - signR_device(RegionH.y, y) - signR_device(RegionH.y, y - Region.y);
	r[idx * 3 + 1] = y;
	// Z
	double z = r[idx * 3 + 2];
	z = z - signR_device(RegionH.z, z) - signR_device(RegionH.z, z - Region.z);
	r[idx * 3 + 2] = z;
}

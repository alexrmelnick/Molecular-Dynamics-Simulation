/*******************************************************************************

Header file for the CUDA implementations (Single-Threaded and Multi-Threaded) of Molecular Dynamics simulation

*******************************************************************************/

/* Functions & function prototypes ********************************************/

/* CUDA Baseline O(N^2) Implementation */
int CUDA_base();
__global__ void ComputeAccelBaseKernel();
__global__ void SingleStepBaseKernel();
__global__ void HalfKickBaseKernel();
__global__ void ApplyBoundaryCondBaseKernel();
__global__ void EvalPropsBaseKernel();

/* CUDA Newton's 3rd Law O(N^2) Implementation */
int CUDA_N3L();
__global__ void ComputeAccelN3LKernel();
__global__ void SingleStepN3LKernel();
__global__ void HalfKickN3LKernel();
__global__ void ApplyBoundaryCondN3LKernel();
__global__ void EvalPropsN3LKernel();


/* CUDA Cell List O(N) Implementation */
int CUDA_Cell();
__global__ void ComputeAccelCellKernel();
__global__ void SingleStepCellKernel();
__global__ void HalfKickCellKernel();
__global__ void ApplyBoundaryCondCellKernel();
__global__ void EvalPropsCellKernel();

/* Constants ******************************************************************/

__constant__ int   d_nAtom;         // number of atoms
__constant__ float d_DELTAT;        // time step
__constant__ float d_DeltaTH;       // half time step
__constant__ float d_RCUT;          // cut-off radius
__constant__ float d_Region[3];     // box lengths
__constant__ float d_RegionH[3];    // half box lengths

/* Variables ******************************************************************/

extern float *d_r;                  // positions [nAtom*3]
extern float *d_rv;                 // velocities
extern float *d_ra;                 // accelerations
extern float *d_potEnergy;          // potential energies
extern float *d_kinEnergy;          // kinetic energies

/******************************************************************************/

/*******************************************************************************

Header file for the CUDA implementations (Single-Threaded and Multi-Threaded) of Molecular Dynamics simulation

*******************************************************************************/

#define BLOCK_SIZE 256

// Device (GPU) kernel declarations for Baseline (O(N^2)) implementation
__global__ void ComputeAccelBaseKernel(
    const double *r,       // positions: flattened nAtom x 3 array
    double *ra,            // accelerations: flattened nAtom x 3 array
    int nAtom,
    double3 RegionH,
    double Duc,
    double Uc
);
__global__ void HalfKickBaseKernel(
    double *rv,            // velocities
    const double *ra,      // accelerations
    int nAtom,
    double DeltaTH
);
__global__ void UpdatePositionKernel(
    double *r,
    const double *rv,
    int nAtom,
    double deltaT
);
__global__ void ApplyBoundaryCondKernel(
    double *r,
    int nAtom,
    double3 Region,
    double3 RegionH
);

// Device (GPU) kernel declarations for Newton's 3rd Law (O(N^2)) implementation
__global__ void ComputeAccelN3LKernel(
    const double *r,
    double *ra,
    int nAtom,
    double3 RegionH,
    double Duc,
    double Uc
);
__global__ void HalfKickN3LKernel(
    double *rv,
    const double *ra,
    int nAtom,
    double DeltaTH
);

// Device (GPU) kernel declarations for Cell-List (O(N)) implementation
__global__ void ComputeAccelCellKernel(
    const double *r,
    double *ra,
    int nAtom,
    double3 Region,
    int *head,
    int *lscl,
    int *lc,
    double *rc,
    double Duc,
    double Uc
);
__global__ void HalfKickCellKernel(
    double *rv,
    const double *ra,
    int nAtom,
    double DeltaTH
);

// Host (CPU) wrapper prototypes for three variants
// Baseline
void ComputeAccelBaseCUDA();
void SingleStepBaseCUDA();
void EvalPropsBaseCUDA();

// Newton's 3rd Law
void ComputeAccelN3LCUDA();
void SingleStepN3LCUDA();
void EvalPropsN3LCUDA();

// Cell-List
void ComputeAccelCellCUDA();
void SingleStepCellCUDA();
void EvalPropsCellCUDA();

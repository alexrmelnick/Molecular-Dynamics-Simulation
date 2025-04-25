/*******************************************************************************

Header file for the CPU implementations (Single-Threaded and Multi-Threaded) of Molecular Dynamics simulation

*******************************************************************************/
#define NMAX 100000  /* Maximum number of atoms which can be simulated */
#define RCUT 2.5     /* Potential cut-off length */
#define PI 3.141592653589793
/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0
#define NCLMAX 10000 /* Maximum number of linked-list cells */
#define EMPTY -1
#define DENSITY 0.8f     /* Number density of atoms (in reduced unit) */
#define INITTEMP 1.0f    /* Starting temperature (in reduced unit) */
#define DELTAT 0.005f     /* Size of a time step (in reduced unit) */
#define STEPLIMIT 50  //100      /* Number of time steps to be simulated */
#define STEPAVG 11       /* Reporting interval for statistical data */

/* Functions & function prototypes ********************************************/

double SignR(double v,double x) {if (x > 0) return v; else return -v;}
double Dmod(double a, double b) {
	int n;
	n = (int) (a/b);
	return (a - b*n);
}
double RandR(double *seed) {
	*seed = Dmod(*seed*DMUL,D2P31M);
	return (*seed/D2P31M);
}
void RandVec3(double *p, double *seed) {
	double x,y,s = 2.0;
	while (s > 1.0) {
		x = 2.0*RandR(seed) - 1.0; y = 2.0*RandR(seed) - 1.0; s = x*x + y*y;
	}
	p[2] = 1.0 - 2.0*s; s = 2.0*sqrt(1.0 - s); p[0] = s*x; p[1] = s*y;
}


/* Serial Baseline O(N^2) Implementation */
void ComputeAccelBase();
void SingleStepBase();
void HalfKickBase();
void ApplyBoundaryCondBase();
void EvalPropsBase();

/* Serial Newton's 3rd Law O(N^2) Implementation */
void ComputeAccelN3L();
void SingleStepN3L();
void HalfKickN3L();
void ApplyBoundaryCondN3L();
void EvalPropsN3L();

/* Baeline OpenMP */
void ComputeAccelPBase();
void SingleStepPBase();
void HalfKickPBase();
void ApplyBoundaryCondPBase();
void EvalPropsPBase();

/* N3L OpenMP */
void ComputeAccelPN3L();
void SingleStepPN3L();
void HalfKickPN3L();
void ApplyBoundaryCondPN3L();
void EvalPropsPN3L();

/* Cells O(N) Implementation */
void SortCells();
void CompressCells();
void ComputeAccelCell();
void SingleStepCell();
void HalfKickCell();
void ApplyBoundaryCondCell();
void EvalPropsCell();

/* Constants ******************************************************************/

int InitUcell[3];   /* Number of unit cells */
double Region[3];  /* MD box lengths */
double RegionH[3]; /* Half the box lengths */
double DeltaTH;    /* Half the time step */
double Uc, Duc;    /* Potential cut-off parameters */

/* Variables ******************************************************************/

int nAtom;            /* Number of atoms */
double r[NMAX][3];    /* r[i][0|1|2] is the x|y|z coordinate of atom i */
double rv[NMAX][3];   /* Atomic velocities */
double ra[NMAX][3];   /* Acceleration on atoms */
double kinEnergy;     /* Kinetic energy */
double potEnergy;     /* Potential energy */
double totEnergy;     /* Total energy */
double temperature;   /* Current temperature */
int stepCount;        /* Current time step */
int head_tail[NCLMAX][2];     /* Headers for the linked cell lists */
int lscl[NMAX];       /* Linked cell lists */
int lc[3];            /* Number of cells in the x|y|z direction */
double rc[3];         /* Length of a cell in the x|y|z direction */
double r_cell[NMAX][3];  
double rv_cell[NMAX][3];   
double ra_cell[NMAX][3];
/******************************************************************************/

/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Baseline Implementation
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

int serial_AVX() {
	//int stepCount;
	for (stepCount=1; stepCount<=STEPLIMIT; stepCount++) {
		SingleStepAVX(); 
		// if (stepCount%STEPAVG == 0) EvalPropsBase();
	}
	return 0;
}

/*----------------------------------------------------------------------------*/
void ComputeAccelAVX() {
/*------------------------------------------------------------------------------
	Acceleration, ra, are computed as a function of atomic coordinates, r,
	using the Lennard-Jones potential.  The sum of atomic potential energies,
	potEnergy, is also computed.   
------------------------------------------------------------------------------*/
	double dr[3],f,fcVal,rrCut,rr,ri2,ri6,r1;
	int j1,j2,n,k;

	rrCut = RCUT*RCUT;
	for (n=0; n<nAtom; n++) for (k=0; k<3; k++) ra[n][k] = 0.0;
	potEnergy = 0.0;

	/* Doubly-nested loop over atomic pairs */
	for (j1=0; j1<nAtom-1; j1++) {
		for (j2=j1+1; j2<nAtom; j2++) {
			/* Computes the squared atomic distance */
			for (rr=0.0, k=0; k<3; k++) {
				dr[k] = r[j1][k] - r[j2][k];
				/* Chooses the nearest image */
				dr[k] = dr[k] - SignR(RegionH[k],dr[k]-RegionH[k])
								- SignR(RegionH[k],dr[k]+RegionH[k]);
				rr = rr + dr[k]*dr[k];
			}
			/* Computes acceleration & potential within the cut-off distance */
			if (rr < rrCut) {
				ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
				fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
				for (k=0; k<3; k++) {
					f = fcVal*dr[k];
					ra[j1][k] = ra[j1][k] + f;
					ra[j2][k] = ra[j2][k] - f;
				}
				potEnergy = potEnergy + 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-RCUT);
			} 
		} 
	}
}

/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
void SingleStepAVX() {
/*------------------------------------------------------------------------------
	r & rv are propagated by DeltaT in time using the velocity-Verlet method.
------------------------------------------------------------------------------*/
	int n,k;

	HalfKickAVX(); /* First half kick to obtain v(t+Dt/2) */
	/*for (n=0; n<nAtom; n++) // Update atomic coordinates to r(t+Dt) 
		for (k=0; k<3; k++) r[n][k] = r[n][k] + DELTAT*rv[n][k];*/

	__m256d   m1;
	__m256d* pos = (__m256d*)r;
	__m256d* vel = (__m256d*)rv;
	__m256d dt = _mm256_set1_pd((double)DELTAT);

	int nloop = 3*nAtom / 4;

	for(n=0; n<nloop; n++) {
		m1 = _mm256_mul_pd(*vel,dt);
		*pos = _mm256_add_pd(*pos,m1);
		pos++;
		vel++;
	}

	ApplyBoundaryCondAVX();
	ComputeAccelAVX(); /* Computes new accelerations, a(t+Dt) */
	HalfKickAVX(); /* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickAVX() {
/*------------------------------------------------------------------------------
	Accelerates atomic velocities, rv, by half the time step.
------------------------------------------------------------------------------*/
	int n,k;
	__m256d   m1;
	__m256d* vel = (__m256d*)rv;
  	__m256d* acc = (__m256d*)ra;
	__m256d dth = _mm256_set1_pd((double)DeltaTH);

	int nloop = 3*nAtom / 4;

	for(n=0; n<nloop; n++) {
		m1 = _mm256_mul_pd(*acc,dth);
		*vel = _mm256_add_pd(*vel,m1);
		acc++;
		vel++;
	}

	/*
	for (n=0; n<nAtom; n++)
		for (k=0; k<3; k++) rv[n][k] = rv[n][k] + DeltaTH*ra[n][k]; */
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondAVX() {
/*------------------------------------------------------------------------------
	Applies periodic boundary conditions to atomic coordinates.
------------------------------------------------------------------------------*/
	// SignR(double v,double x) {if (x > 0) return v; else return -v;}

	int n,k;
	__m256d m1, m2, m3, m4;
	__m256d* pos = (__m256d*)r;
	__m256d regionH = _mm256_set1_pd((double)RegionH[0]);
	__m256d region = _mm256_set1_pd((double)Region[0]);
	uint64_t signbit[] = {0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000};
	__m256d negative = _mm256_loadu_pd(&signbit);

	int nloop = 3*nAtom / 4;

	for(n=0; n<nloop; n++) {
		m1 = _mm256_and_pd(*pos,negative);
		m2 = _mm256_xor_pd(regionH,m1);
		m3 = _mm256_sub_pd(*pos,region);
		m1 = _mm256_and_pd(m3,negative);
		m4 = _mm256_xor_pd(regionH,m1);
		m3 = _mm256_add_pd(m2,m4);
		*pos = _mm256_sub_pd(*pos,m3);
		pos++;
	}

	/*
	for (n=0; n<nAtom; n++) 
		for (k=0; k<3; k++) 
			r[n][k] = r[n][k] - SignR(RegionH[k],r[n][k])
			                  - SignR(RegionH[k],r[n][k]-Region[k]);*/
}

/*----------------------------------------------------------------------------*/
void EvalPropsAVX() {
/*------------------------------------------------------------------------------
	Evaluates physical properties: kinetic, potential & total energies.
------------------------------------------------------------------------------*/
	double vv;
	int n,k;


	kinEnergy = 0.0;
	for (n=0; n<nAtom; n++) {
		vv = 0.0;
		for (k=0; k<3; k++)
			vv = vv + rv[n][k]*rv[n][k]; 
		kinEnergy = kinEnergy + vv;
	}
	kinEnergy *= (0.5/nAtom);
	potEnergy /= nAtom;
	totEnergy = kinEnergy + potEnergy;
	temperature = kinEnergy*2.0/3.0;

	/* Print the computed properties */
	printf("%9.6f %9.6f %9.6f %9.6f\n",
	stepCount*DELTAT,temperature,potEnergy,totEnergy);
}
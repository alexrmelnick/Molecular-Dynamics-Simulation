/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Leverage cell lists - use preallocated arrays as opposed to linked-lists
*******************************************************************************/
#include <stdio.h>
#include <math.h>

int serial_cell() {
	for (stepCount=1; stepCount<=STEPLIMIT; stepCount++) {
		SingleStep(); 
		if (stepCount%STEPAVG == 0) EvalPropsCell();
	}
	return 0;
}

/*----------------------------------------------------------------------------*/
void ComputeAccelCell() {
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
void SingleStepCell() {
/*------------------------------------------------------------------------------
	r & rv are propagated by DeltaT in time using the velocity-Verlet method.
------------------------------------------------------------------------------*/
	int n,k;

	HalfKickCell(); /* First half kick to obtain v(t+Dt/2) */
	for (n=0; n<nAtom; n++) /* Update atomic coordinates to r(t+Dt) */
		for (k=0; k<3; k++) r[n][k] = r[n][k] + DELTAT*rv[n][k];
	ApplyBoundaryCondCell();
	ComputeAccelCell(); /* Computes new accelerations, a(t+Dt) */
	HalfKickCell(); /* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickCell() {
/*------------------------------------------------------------------------------
	Accelerates atomic velocities, rv, by half the time step.
------------------------------------------------------------------------------*/
	int n,k;
	for (n=0; n<nAtom; n++)
		for (k=0; k<3; k++) rv[n][k] = rv[n][k] + DeltaTH*ra[n][k];
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondCell() {
/*------------------------------------------------------------------------------
	Applies periodic boundary conditions to atomic coordinates.
------------------------------------------------------------------------------*/
	int n,k;
	for (n=0; n<nAtom; n++) 
		for (k=0; k<3; k++) 
			r[n][k] = r[n][k] - SignR(RegionH[k],r[n][k])
			                  - SignR(RegionH[k],r[n][k]-Region[k]);
}

/*----------------------------------------------------------------------------*/
void EvalPropsCell() {
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
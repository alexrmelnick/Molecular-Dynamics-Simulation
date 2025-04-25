/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Leverage cell lists - use preallocated arrays as opposed to linked-lists
*******************************************************************************/
#include <stdio.h>
#include <math.h>

int serial_cell() {
	int sortstep = 5;
	// Make cell arrays
	SortCells();
	for (stepCount=1; stepCount<=STEPLIMIT; stepCount++) {
		SingleStepCell(); 
		if (stepCount%STEPAVG == 0) EvalPropsCell();
		if (stepCount%sortstep == 0) SortCells();
	}
	return 0;
}

/*----------------------------------------------------------------------------*/
void SortCells() {
/*------------------------------------------------------------------------------
	Fill in any gaps
------------------------------------------------------------------------------*/
	int x, y, z, head_ind, tail_ind;

	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];

		while(tail_ind > head_ind) {
			if(r[head_ind][0] == EMPTY) {
				// move tail-1 to head
				// set tail-1 to zeros
				// decrement tail
			}
			head_ind++;
		}

		head_tail[c][1] = tail_ind;
	}

}

/*----------------------------------------------------------------------------*/
void ComputeAccelCell() {
/*------------------------------------------------------------------------------
	Acceleration, ra, are computed as a function of atomic coordinates, r,
	using the Lennard-Jones potential.  The sum of atomic potential energies,
	potEnergy, is also computed.   
------------------------------------------------------------------------------*/
	int i,j,a,lcyz,lcxyz,mc[3],c,mc1[3],c1;
	double dr[3],rr,ri2,ri6,r1,rrCut,fcVal,f,rshift[3];
	int inner_cell_start, inner_cell_end, neighbor_cell_start, neighbor_cell_end;

	/* Reset the potential & forces */
	for (i=0; i<nAtom; i++) for (a=0; a<3; a++) ra[i][a] = 0.0;
	potEnergy = 0.0;

	/* Make a linked-cell list, lscl--------------------------------------------*/

	lcyz = lc[1]*lc[2];
	lcxyz = lc[0]*lcyz;

	/* Reset the headers, head */
	for (c=0; c<lcxyz; c++) head[c] = EMPTY;

	/* Scan atoms to construct headers, head, & linked lists, lscl */

	for (i=0; i<nAtom; i++) {
		for (a=0; a<3; a++) mc[a] = r[i][a]/rc[a];

		/* Translate the vector cell index, mc, to a scalar cell index */
		c = mc[0]*lcyz+mc[1]*lc[2]+mc[2];

		/* Link to the previous occupant (or EMPTY if you're the 1st) */
		lscl[i] = head[c];

		/* The last one goes to the header */
		head[c] = i;
	} /* Endfor atom i */

	/* Calculate pair interaction-----------------------------------------------*/

	rrCut = RCUT*RCUT;

	/* Scan inner cells */
	for (mc[0]=0; mc[0]<lc[0]; (mc[0])++)
	for (mc[1]=0; mc[1]<lc[1]; (mc[1])++)
	for (mc[2]=0; mc[2]<lc[2]; (mc[2])++) {

		/* Calculate a scalar cell index */
		c = mc[0]*lcyz+mc[1]*lc[2]+mc[2];
		/* Skip this cell if empty */
		if (head_tail[c][0] == head_tail[c][1]) continue;
			
		inner_cell_start = head_tail[c][0];
		inner_cell_end = head_tail[c][1];

		/* Scan the neighbor cells (including itself) of cell c */
		for (mc1[0]=mc[0]-1; mc1[0]<=mc[0]+1; (mc1[0])++)
		for (mc1[1]=mc[1]-1; mc1[1]<=mc[1]+1; (mc1[1])++)
		for (mc1[2]=mc[2]-1; mc1[2]<=mc[2]+1; (mc1[2])++) {
			/* Periodic boundary condition by shifting coordinates */
			for (a=0; a<3; a++) {
				if (mc1[a] < 0)
					rshift[a] = -Region[a];
				else if (mc1[a]>=lc[a])
					rshift[a] = Region[a];
				else
					rshift[a] = 0.0;
			}
			/* Calculate the scalar cell index of the neighbor cell */
			c1 = ((mc1[0]+lc[0])%lc[0])*lcyz
				+((mc1[1]+lc[1])%lc[1])*lc[2]
				+((mc1[2]+lc[2])%lc[2]);
			/* Skip this neighbor cell if empty */
			if (head_tail[c1][0] == head_tail[c1][1]) continue;

			neighbor_cell_start = head_tail[c1][0];
			neighbor_cell_end = head_tail[c1][1];

			/* Scan atom i in cell c */
			for(i = inner_cell_start; i < inner_cell_end; i++) {
				// if empty slot, continue to the next
				if(r[i][0] == EMPTY) continue;

				for(j = neighbor_cell_start; i < neighbor_cell_end; i++) {
					// if empty slot, continue to the next
					if(r[j][0] == EMPTY) continue;
					if(i < j) {
						/* Pair vector dr = r[i]-r[j] */
						for (rr=0.0, a=0; a<3; a++) {
							dr[a] = r[i][a]-(r[j][a]+rshift[a]);
							rr += dr[a]*dr[a];
						}

						/* Calculate potential & forces if rij<RCUT */
						if (rr < rrCut) {
							ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
							fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
							for (a=0; a<3; a++) {
								f = fcVal*dr[a];
								ra[i][a] += f;
								ra[j][a] -= f;
							}
							potEnergy += 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-RCUT);
						}
					}
				}
			}
		} /* Endfor neighbor cells, c1 */
	} /* Endfor central cell, c */
}

/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
void SingleStepCell() {
/*------------------------------------------------------------------------------
	r & rv are propagated by DeltaT in time using the velocity-Verlet method.
------------------------------------------------------------------------------*/
	int n,k;
	//printf("single step");
	HalfKickCell(); /* First half kick to obtain v(t+Dt/2) */
	//printf("half kick");
	for (n=0; n<nAtom; n++) /* Update atomic coordinates to r(t+Dt) */
		for (k=0; k<3; k++) r[n][k] = r[n][k] + DELTAT*rv[n][k];
	//printf("update");
	ApplyBoundaryCondCell();
	//printf("boundary");
	ComputeAccelCell(); /* Computes new accelerations, a(t+Dt) */
	//printf("computeaccel");
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
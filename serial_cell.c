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
		if (stepCount%sortstep == 0) CompressCells();
	}
	return 0;
}


/*----------------------------------------------------------------------------*/
void SortCells() {
/*------------------------------------------------------------------------------
	Sort atoms into cells
------------------------------------------------------------------------------*/
	int c, i, a, tail, mc[3];

	// define bounds
	int lcyz = lc[1]*lc[2];
	int total_cells = lc[0] * lcyz;
	int buffer_len = nAtom / total_cells * 2;
	printf("SIZE OF BUFFER %d for %d number of cells\n", buffer_len, total_cells);

	// initialize head_tail
	for (c = 0; c < total_cells; c++) {
		head_tail[c][0] = c*buffer_len;
		head_tail[c][1] = c*buffer_len;
		//printf("head_tail[%d] = %d, %d\n",c,head_tail[c][0],head_tail[c][1]);
	}

	// sort to cell arrays
	for (i = 0; i < nAtom; i++) {
		for (a=0; a<3; a++) mc[a] = r[i][a]/rc[a];
		c = mc[0]*lcyz+mc[1]*lc[2]+mc[2];

		tail = head_tail[c][1];
		for(a=0; a<3; a++) {
			r_cell[tail][a] = r[i][a];
		}
		for(a=0; a<3; a++) {
			rv_cell[tail][a] = rv[i][a];
		}
		for(a=0; a<3; a++) {
			ra_cell[tail][a] = ra[i][a];
		}
		head_tail[c][1]++;
	}
}


/*----------------------------------------------------------------------------*/
void CompressCells() {
/*------------------------------------------------------------------------------
	Fill in any gaps
------------------------------------------------------------------------------*/
	int c, i, head_ind, tail_ind;

	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];

		while(tail_ind > head_ind) {
			if(r_cell[head_ind][0] == EMPTY) {
				// move [tail-1] to head and set [tail-1] to zero
				for(i=0; i<3; i++) {
					r_cell[head_ind][i] = r_cell[tail_ind-1][i];
					r_cell[tail_ind-1][i] = EMPTY;
				}
				for(i=0; i<3; i++) {
					rv_cell[head_ind][i] = rv_cell[tail_ind-1][i];
					rv_cell[tail_ind-1][i] = EMPTY;
				}
				for(i=0; i<3; i++) {
					ra_cell[head_ind][i] = ra_cell[tail_ind-1][i];
					ra_cell[tail_ind-1][i] = EMPTY;
				}
				tail_ind--;
			}
			head_ind++;
		}
		// update tail pointer
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
	int i,j,a,lcyz,total_cells,mc[3],c,mc1[3],c1,head_ind,tail_ind,c_new,tail_new;
	double dr[3],rr,ri2,ri6,r1,rrCut,fcVal,f,rshift[3];
	int inner_cell_start, inner_cell_end, neighbor_cell_start, neighbor_cell_end;

	/* Reset the potential & forces */
	potEnergy = 0.0;

	/* Move any atoms if necessary and set accel to zero ---------------------*/
	lcyz = lc[1]*lc[2];
	total_cells = lc[0] * lcyz;
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];
		for(i=head_ind; i<tail_ind; i++) {
			if (r_cell[i][0] == EMPTY) continue;
			for (a=0; a<3; a++) mc[a] = r_cell[i][a]/rc[a];
			for (a=0; a<3; a++) ra_cell[i][a] = 0.0;
			/* Translate the vector cell index, mc, to a scalar cell index */
			c_new = mc[0]*lcyz+mc[1]*lc[2]+mc[2];

			//if different, move
			if(c_new != c) {
				//printf("MOVING");
				tail_new = head_tail[c_new][1];
				for(a=0; a<3; a++) {
					r_cell[tail_new][a] = r_cell[i][a];
					r_cell[i][a] = EMPTY;
				}
				for(a=0; a<3; a++) {
					rv_cell[tail_new][a] = rv_cell[i][a];
					rv_cell[i][a] = EMPTY;
				}
				for(a=0; a<3; a++) {
					ra_cell[tail_new][a] = ra_cell[i][a];
					ra_cell[i][a] = EMPTY;
				}
				head_tail[c_new][1]++;
			}
		}
	}


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

			//printf("HERE!");

			/* Scan atom i in cell c */
			for(i = inner_cell_start; i < inner_cell_end; i++) {
				// if empty slot, continue to the next
				if(r_cell[i][0] == EMPTY) continue;

				for(j = neighbor_cell_start; j < neighbor_cell_end; j++) {
					// if empty slot, continue to the next
					if(r_cell[j][0] == EMPTY) continue;
					if(i < j) {
						/* Pair vector dr = r[i]-r[j] */
						for (rr=0.0, a=0; a<3; a++) {
							dr[a] = r_cell[i][a]-(r_cell[j][a]+rshift[a]);
							rr += dr[a]*dr[a];
						}

						/* Calculate potential & forces if rij<RCUT */
						if (rr < rrCut) {
							ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
							fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
							for (a=0; a<3; a++) {
								f = fcVal*dr[a];
								ra_cell[i][a] += f;
								ra_cell[j][a] -= f;
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
	int c,k,i;
	int head_ind, tail_ind;
	//printf("single step");
	HalfKickCell(); /* First half kick to obtain v(t+Dt/2) */
	//printf("half kick");
	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];
		for(i=head_ind; i<tail_ind; i++) {
			if (r_cell[i][0] == EMPTY) continue;
			for (k=0; k<3; k++) r_cell[i][k] = r_cell[i][k] + DELTAT*rv_cell[i][k];
		}
	}
	//printf("update");
	ApplyBoundaryCondCell();
	//printf("boundary");
	ComputeAccelCell(); /* Computes new accelerations, a(t+Dt) */
	//printf("computeaccel\n");
	HalfKickCell(); /* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickCell() {
/*------------------------------------------------------------------------------
	Accelerates atomic velocities, rv, by half the time step.
------------------------------------------------------------------------------*/
	int c,k,i;
	int head_ind, tail_ind;

	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];
		for(i=head_ind; i<tail_ind; i++) {
			if (r_cell[i][0] == EMPTY) continue;
			for (k=0; k<3; k++) rv_cell[i][k] = rv_cell[i][k] + DeltaTH*ra_cell[i][k];
		}
	}
		
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondCell() {
/*------------------------------------------------------------------------------
	Applies periodic boundary conditions to atomic coordinates.
------------------------------------------------------------------------------*/
	int c,i,k;
	int head_ind, tail_ind;

	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];
		for(i=head_ind; i<tail_ind; i++) {
			if (r_cell[i][0] == EMPTY) continue;
			for (k=0; k<3; k++) 
				r_cell[i][k] = r_cell[i][k] - SignR(RegionH[k],r_cell[i][k])
								- SignR(RegionH[k],r_cell[i][k]-Region[k]);
		}
	}
}

/*----------------------------------------------------------------------------*/
void EvalPropsCell() {
/*------------------------------------------------------------------------------
	Evaluates physical properties: kinetic, potential & total energies.
------------------------------------------------------------------------------*/
	double vv;
	int c,k,i;
	int head_ind, tail_ind;

	int total_cells = lc[0] * lc[1] * lc[2];
	for (c = 0; c < total_cells; c++) {
		head_ind = head_tail[c][0];
		tail_ind = head_tail[c][1];
		for(i=head_ind; i<tail_ind; i++) {
			if (r_cell[i][0] == EMPTY) continue;
			vv = 0.0;
			for (k=0; k<3; k++) vv = vv + rv_cell[i][k]*rv_cell[i][k];
			kinEnergy = kinEnergy + vv;
		}
	}
	
	kinEnergy *= (0.5/nAtom);
	potEnergy /= nAtom;
	totEnergy = kinEnergy + potEnergy;
	temperature = kinEnergy*2.0/3.0;

	/* Print the computed properties */
	printf("%9.6f %9.6f %9.6f %9.6f\n",
	stepCount*DELTAT,temperature,potEnergy,totEnergy);
}
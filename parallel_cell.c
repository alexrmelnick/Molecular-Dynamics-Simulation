/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Leverage cell lists - use preallocated arrays as opposed to linked-lists
Now with OpenMP parallelization
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <omp.h>

// #include <cpu.h> // This should be commented out but it is needed to get rid of all of the squiggles while editing

int parallel_cell()
{
    int sortstep = 5;
    // Make cell arrays
    SortPCells();
    for (stepCount = 1; stepCount <= STEPLIMIT; stepCount++)
    {
        SingleStepPCell();
        if (stepCount % STEPAVG == 0)
            EvalPropsPCell(); // Only necessary for verification
        if (stepCount % sortstep == 0)
            CompressPCells();
    }
    return 0;
}

/*----------------------------------------------------------------------------*/
void SortPCells()
{
    /*------------------------------------------------------------------------------
        Sort atoms into cells
    ------------------------------------------------------------------------------*/
    int c, i, a, tail, mc[3];

    // define bounds
    int lcyz = lc[1] * lc[2];
    int total_cells = lc[0] * lcyz;
    int buffer_len = (2 * nAtom + total_cells - 1) / total_cells; // 2*nAtom/total_cells rounded up
    printf("SIZE OF BUFFER %d for %d number of cells\n", buffer_len, total_cells);

// #pragma omp parallel
//     {

        // initialize head_tail
// #pragma omp for
        for (c = 0; c < total_cells; c++)
        {
            head_tail[c][0] = c * buffer_len;
            head_tail[c][1] = c * buffer_len;
            // printf("head_tail[%d] = %d, %d\n",c,head_tail[c][0],head_tail[c][1]);
        }

        // sort to cell arrays
// #pragma omp for private(i, a, c, mc, tail)
        for (i = 0; i < nAtom; i++)
        {
            for (a = 0; a < 3; a++)
                mc[a] = r[i][a] / rc[a];
            c = mc[0] * lcyz + mc[1] * lc[2] + mc[2];

// #pragma omp atomic capture // Atomic operation required unless we totally change the way we do this
            tail = head_tail[c][1]++;

            for (a = 0; a < 3; a++)
            {
                r_cell[tail][a] = r[i][a];
            }
            for (a = 0; a < 3; a++)
            {
                rv_cell[tail][a] = rv[i][a];
            }
            for (a = 0; a < 3; a++)
            {
                ra_cell[tail][a] = ra[i][a];
            }
        }
    // }
}

/*----------------------------------------------------------------------------*/
void CompressPCells()
{
    /*------------------------------------------------------------------------------
        Fill in any gaps
    ------------------------------------------------------------------------------*/
    int c, i, head_ind, tail_ind;
    int total_cells = lc[0] * lc[1] * lc[2];

// #pragma omp parallel for private(c, i, head_ind, tail_ind) schedule(static)
    for (c = 0; c < total_cells; c++)
    {
        head_ind = head_tail[c][0];
        tail_ind = head_tail[c][1];

        while (tail_ind > head_ind)
        {
            if (r_cell[head_ind][0] == EMPTY)
            {
                // move [tail-1] to head and set [tail-1] to zero
                for (i = 0; i < 3; i++)
                {
                    r_cell[head_ind][i] = r_cell[tail_ind - 1][i];
                    r_cell[tail_ind - 1][i] = EMPTY;
                }
                for (i = 0; i < 3; i++)
                {
                    rv_cell[head_ind][i] = rv_cell[tail_ind - 1][i];
                    rv_cell[tail_ind - 1][i] = EMPTY;
                }
                for (i = 0; i < 3; i++)
                {
                    ra_cell[head_ind][i] = ra_cell[tail_ind - 1][i];
                    ra_cell[tail_ind - 1][i] = EMPTY;
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
void ComputeAccelPCell()
{
    /*------------------------------------------------------------------------------
        Acceleration, ra, are computed as a function of atomic coordinates, r,
        using the Lennard-Jones potential.  The sum of atomic potential energies,
        potEnergy, is also computed.
    ------------------------------------------------------------------------------*/
    int i, j, a, lcyz, total_cells, c, c1, head_ind, tail_ind, c_new, tail_new;
    double dr[3], rr, ri2, ri6, r1, rrCut, fcVal, f, rshift[3];
    int inner_cell_start, inner_cell_end, neighbor_cell_start, neighbor_cell_end;

    /* Reset the potential & forces */
    potEnergy = 0.0;

    /* Move any atoms if necessary and set accel to zero ---------------------*/
    lcyz = lc[1] * lc[2];
    total_cells = lc[0] * lcyz;
    int mc[3];
    int buffer_len = (2 * nAtom + total_cells - 1) / total_cells; // 2*nAtom/total_cells rounded up
    int total_slots = total_cells * buffer_len;
    int maxThreads = omp_get_max_threads();

    double (*ra_private)[total_slots][3] = (double (*)[total_slots][3])calloc(maxThreads, sizeof *ra_private); // Allocate private arrays for each thread so we don't have to use super slow atomic operations

#pragma omp parallel reduction(+ : potEnergy)
    {
        int tid = omp_get_thread_num();

#pragma omp for private(c, head_ind, tail_ind, c_new, tail_new, i, a, mc)
        for (c = 0; c < total_cells; c++)
        {
            head_ind = head_tail[c][0];
            tail_ind = head_tail[c][1];

            for (i = head_ind; i < tail_ind; i++)
            {
                if (r_cell[i][0] == EMPTY)
                    continue;
                for (a = 0; a < 3; a++)
                    mc[a] = r_cell[i][a] / rc[a];
                for (a = 0; a < 3; a++)
                    ra_cell[i][a] = 0.0;
                /* Translate the vector cell index, mc, to a scalar cell index */
                c_new = mc[0] * lcyz + mc[1] * lc[2] + mc[2];

                // if different, move
                if (c_new != c)
                {
                    // printf("MOVING");
#pragma omp atomic capture // Atomic operation required
                    tail_new = head_tail[c_new][1]++;
                    for (a = 0; a < 3; a++)
                    {
                        r_cell[tail_new][a] = r_cell[i][a];
                        r_cell[i][a] = EMPTY;
                    }
                    for (a = 0; a < 3; a++)
                    {
                        rv_cell[tail_new][a] = rv_cell[i][a];
                        rv_cell[i][a] = EMPTY;
                    }
                    for (a = 0; a < 3; a++)
                    {
                        ra_cell[tail_new][a] = ra_cell[i][a];
                        ra_cell[i][a] = EMPTY;
                    }
                }
            }
        }

        /* Calculate pair interaction-----------------------------------------------*/

        rrCut = RCUT * RCUT;

/* Scan inner cells */
#pragma omp for collapse(3) schedule(dynamic) private(c, c1, inner_cell_start, inner_cell_end, neighbor_cell_start, neighbor_cell_end, i, j, a, rr, dr, ri2, ri6, r1, fcVal, f)
        for (int mc0 = 0; mc0 < lc[0]; mc0++)
            for (int mc1 = 0; mc1 < lc[1]; mc1++)
                for (int mc2 = 0; mc2 < lc[2]; mc2++)
                {

                    /* Calculate a scalar cell index */
                    int c = mc0 * lcyz + mc1 * lc[2] + mc2;

                    /* Skip this cell if empty */
                    if (head_tail[c][0] == head_tail[c][1])
                        continue;

                    inner_cell_start = head_tail[c][0];
                    inner_cell_end = head_tail[c][1];

                    /* Scan the neighbor cells (including itself) of cell c */
                    for (int nm0 = mc0 - 1; nm0 <= mc0 + 1; nm0++)
                        for (int nm1 = mc1 - 1; nm1 <= mc1 + 1; nm1++)
                            for (int nm2 = mc2 - 1; nm2 <= mc2 + 1; nm2++)
                            {
                                /* Periodic boundary condition by shifting coordinates */
                                double rshift0 = (nm0 < 0) ? -Region[0] : (nm0 >= lc[0] ? Region[0] : 0.0);
                                double rshift1 = (nm1 < 0) ? -Region[1] : (nm1 >= lc[1] ? Region[1] : 0.0);
                                double rshift2 = (nm2 < 0) ? -Region[2] : (nm2 >= lc[2] ? Region[2] : 0.0);

                                /* Calculate the scalar cell index of the neighbor cell */
                                c1 = ((nm0 + lc[0]) % lc[0]) * lcyz + ((nm1 + lc[1]) % lc[1]) * lc[2] + ((nm2 + lc[2]) % lc[2]);
                                /* Skip this neighbor cell if empty */

                                if (head_tail[c1][0] == head_tail[c1][1])
                                    continue;

                                neighbor_cell_start = head_tail[c1][0];
                                neighbor_cell_end = head_tail[c1][1];

                                // printf("HERE!");

                                /* Scan atom i in cell c */
                                for (i = inner_cell_start; i < inner_cell_end; i++)
                                {

                                    // if empty slot, continue to the next
                                    if (r_cell[i][0] == EMPTY)
                                        continue;

                                    for (j = neighbor_cell_start; j < neighbor_cell_end; j++)
                                    {

                                        // if empty slot, continue to the next
                                        if (r_cell[j][0] == EMPTY)
                                            continue;
                                        if (i < j)
                                        {
                                            rshift[0] = rshift0;
                                            rshift[1] = rshift1;
                                            rshift[2] = rshift2;

                                            /* Pair vector dr = r[i]-r[j] */
                                            for (rr = 0.0, a = 0; a < 3; a++)
                                            {
                                                dr[a] = r_cell[i][a] - (r_cell[j][a] + rshift[a]);
                                                rr += dr[a] * dr[a];
                                            }

                                            /* Calculate potential & forces if rij<RCUT */
                                            if (rr < rrCut)
                                            {
                                                ri2 = 1.0 / rr;
                                                ri6 = ri2 * ri2 * ri2;
                                                r1 = sqrt(rr);
                                                fcVal = 48.0 * ri2 * ri6 * (ri6 - 0.5) + Duc / r1;
                                                for (a = 0; a < 3; a++)
                                                {
                                                    f = fcVal * dr[a];
                                                    ra_private[tid][i][a] += f; // thread-array update
                                                    ra_private[tid][j][a] -= f;
                                                }
                                                potEnergy += 4.0 * ri6 * (ri6 - 1.0) - Uc - Duc * (r1 - RCUT);
                                            }
                                        }
                                    }
                                }
                            } /* Endfor neighbor cells, c1 */
                } /* Endfor central cell, c */

        // Synchronize and merge thread-arrays into the global array
#pragma omp barrier
#pragma omp for
        for (int slot = 0; slot < total_slots; slot++)
        {
            for (int a = 0; a < 3; a++)
            {
                double sum = 0.0;
                for (int t = 0; t < maxThreads; t++)
                    sum += ra_private[t][slot][a];
                ra_cell[slot][a] = sum;
            }
        }
    } // end of parallel region

    free(ra_private); // Free the private arrays
}
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
void SingleStepPCell()
{
    /*------------------------------------------------------------------------------
        r & rv are propagated by DeltaT in time using the velocity-Verlet method.
    ------------------------------------------------------------------------------*/
    int c, k, i;
    int head_ind, tail_ind;
    // printf("single step");
    HalfKickPCell(); /* First half kick to obtain v(t+Dt/2) */
    // printf("half kick");
    int total_cells = lc[0] * lc[1] * lc[2];
    for (c = 0; c < total_cells; c++)
    {
        head_ind = head_tail[c][0];
        tail_ind = head_tail[c][1];
        for (i = head_ind; i < tail_ind; i++)
        {
            if (r_cell[i][0] == EMPTY)
                continue;
            for (k = 0; k < 3; k++)
                r_cell[i][k] = r_cell[i][k] + DELTAT * rv_cell[i][k];
        }
    }
    // printf("update");
    ApplyBoundaryCondPCell();
    // printf("boundary");
    ComputeAccelPCell(); /* Computes new accelerations, a(t+Dt) */
    // printf("computeaccel\n");
    HalfKickPCell(); /* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickPCell()
{
    /*------------------------------------------------------------------------------
        Accelerates atomic velocities, rv, by half the time step.
    ------------------------------------------------------------------------------*/
    int c, k, i;
    int head_ind, tail_ind;

    int total_cells = lc[0] * lc[1] * lc[2];

// #pragma omp parallel for private(c, k, i, head_ind, tail_ind) schedule(static)
    for (c = 0; c < total_cells; c++)
    {
        head_ind = head_tail[c][0];
        tail_ind = head_tail[c][1];
        for (i = head_ind; i < tail_ind; i++)
        {
            if (r_cell[i][0] == EMPTY)
                continue;
            for (k = 0; k < 3; k++)
                rv_cell[i][k] = rv_cell[i][k] + DeltaTH * ra_cell[i][k];
        }
    }
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondPCell()
{
    /*------------------------------------------------------------------------------
        Applies periodic boundary conditions to atomic coordinates.
    ------------------------------------------------------------------------------*/
    int c, i, k;
    int head_ind, tail_ind;

    int total_cells = lc[0] * lc[1] * lc[2];

// #pragma omp parallel for private(c, k, i, head_ind, tail_ind) schedule(static)
    for (c = 0; c < total_cells; c++)
    {
        head_ind = head_tail[c][0];
        tail_ind = head_tail[c][1];
        for (i = head_ind; i < tail_ind; i++)
        {
            if (r_cell[i][0] == EMPTY)
                continue;
            for (k = 0; k < 3; k++)
                r_cell[i][k] = r_cell[i][k] - SignR(RegionH[k], r_cell[i][k]) - SignR(RegionH[k], r_cell[i][k] - Region[k]);
        }
    }
}

/*----------------------------------------------------------------------------*/
void EvalPropsPCell()
{
    /*------------------------------------------------------------------------------
        Evaluates physical properties: kinetic, potential & total energies.
    ------------------------------------------------------------------------------*/
    double vv;
    int c, k, i;
    int head_ind, tail_ind;

    int total_cells = lc[0] * lc[1] * lc[2];
    for (c = 0; c < total_cells; c++)
    {
        head_ind = head_tail[c][0];
        tail_ind = head_tail[c][1];
        for (i = head_ind; i < tail_ind; i++)
        {
            if (r_cell[i][0] == EMPTY)
                continue;
            vv = 0.0;
            for (k = 0; k < 3; k++)
                vv = vv + rv_cell[i][k] * rv_cell[i][k];
            kinEnergy = kinEnergy + vv;
        }
    }

    kinEnergy *= (0.5 / nAtom);
    potEnergy /= nAtom;
    totEnergy = kinEnergy + potEnergy;
    temperature = kinEnergy * 2.0 / 3.0;

    /* Print the computed properties */
    printf("%9.6f %9.6f %9.6f %9.6f\n",
           stepCount * DELTAT, temperature, potEnergy, totEnergy);
}
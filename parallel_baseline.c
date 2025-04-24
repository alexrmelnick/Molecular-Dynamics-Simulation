/*******************************************************************************
Molecular dynamics (MD) simulation with the Lennard-Jones potential.

Baseline Implementation, now with OpenMP parallelization
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <omp.h>

int parallel_base()
{
    // int stepCount;
    //  This loop cannot be parallelized because each iteration depends on the previous one
    for (stepCount = 1; stepCount <= STEPLIMIT; stepCount++)
    {
        SingleStepPBase();
        if (stepCount % STEPAVG == 0)
            EvalPropsPBase();
    }
    return 0;
}

/*----------------------------------------------------------------------------*/
void ComputeAccelPBase()
{
    /*------------------------------------------------------------------------------
        Acceleration, ra, are computed as a function of atomic coordinates, r,
        using the Lennard-Jones potential.  The sum of atomic potential energies,
        potEnergy, is also computed.
    ------------------------------------------------------------------------------*/
    double f, fcVal, ri2, ri6, r1;
    int k;
    int maxThreads = omp_get_max_threads();
    double (*ra_private)[nAtom][3] = (double (*)[nAtom][3])calloc(maxThreads, sizeof *ra_private); // Allocate private arrays for each thread so we don't have to use super slow atomic operations

    // for (n = 0; n < nAtom; n++)
    //     for (k = 0; k < 3; k++)
    //         ra[n][k] = 0.0;
    potEnergy = 0.0;

#pragma omp parallel private(f, fcVal, ri2, ri6, r1, k)
    {
        int nThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        /* Doubly-nested loop over atomic pairs */
#pragma omp for reduction(+ : potEnergy)
        for (int j1 = 0; j1 < nAtom; j1++)
        {
            double dr[3], rr = 0.0;

            for (int j2 = 0; j2 < nAtom; j2++)
            {
                /* Computes the squared atomic distance */
                if (j1 != j2)
                {
                    rr = 0.0;
                    for (k = 0; k < 3; k++)
                    {
                        dr[k] = r[j1][k] - r[j2][k];
                        /* Chooses the nearest image */
                        dr[k] = dr[k] - SignR(RegionH[k], dr[k] - RegionH[k]) - SignR(RegionH[k], dr[k] + RegionH[k]);
                        rr = rr + dr[k] * dr[k];
                    }

                    /* Computes acceleration & potential within the cut-off distance */
                    if (sqrt(rr) < RCUT)
                    {
                        ri2 = 1.0 / rr;
                        ri6 = ri2 * ri2 * ri2;
                        r1 = sqrt(rr);
                        fcVal = 48.0 * ri2 * ri6 * (ri6 - 0.5) + Duc / r1;
                        for (k = 0; k < 3; k++)
                        {
                            f = fcVal * dr[k];
                            ra_private[tid][j1][k] += f;
                        }
                        potEnergy = potEnergy + 0.5 * (4.0 * ri6 * (ri6 - 1.0) - Uc - Duc * (r1 - RCUT));
                    }
                }
            }
        }
    }

    // Once all threads have finished, we need to combine the results from each thread
#pragma omp barrier
#pragma omp for
    for (int n = 0; n < nAtom; n++)
    {
        for (k = 0; k < 3; k++)
        {
            double sum = 0;
            for (int t = 0; t < maxThreads; t++)
                sum += ra_private[t][n][k];
            ra[n][k] = sum;
        }
    } // End of OpenMP parallel region

    free(ra_private); // Free the private arrays
}

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
void SingleStepPBase()
{
    /*------------------------------------------------------------------------------
        r & rv are propagated by DeltaT in time using the velocity-Verlet method.
    ------------------------------------------------------------------------------*/
    int n, k;

    HalfKickPBase();            /* First half kick to obtain v(t+Dt/2) */
    for (n = 0; n < nAtom; n++) /* Update atomic coordinates to r(t+Dt) */
        for (k = 0; k < 3; k++)
            r[n][k] = r[n][k] + DELTAT * rv[n][k];
    ApplyBoundaryCondPBase();
    ComputeAccelPBase(); /* Computes new accelerations, a(t+Dt) */
    HalfKickPBase();     /* Second half kick to obtain v(t+Dt) */
}

/*----------------------------------------------------------------------------*/
void HalfKickPBase()
{
    /*------------------------------------------------------------------------------
        Accelerates atomic velocities, rv, by half the time step.
    ------------------------------------------------------------------------------*/
    int n, k;
    for (n = 0; n < nAtom; n++)
        for (k = 0; k < 3; k++)
            rv[n][k] = rv[n][k] + DeltaTH * ra[n][k];
}

/*----------------------------------------------------------------------------*/
void ApplyBoundaryCondPBase()
{
    /*------------------------------------------------------------------------------
        Applies periodic boundary conditions to atomic coordinates.
    ------------------------------------------------------------------------------*/
    int n, k;
    for (n = 0; n < nAtom; n++)
        for (k = 0; k < 3; k++)
            r[n][k] = r[n][k] - SignR(RegionH[k], r[n][k]) - SignR(RegionH[k], r[n][k] - Region[k]);
}

/*----------------------------------------------------------------------------*/
void EvalPropsPBase()
{
    /*------------------------------------------------------------------------------
        Evaluates physical properties: kinetic, potential & total energies.
    ------------------------------------------------------------------------------*/
    double vv;
    int n, k;

    kinEnergy = 0.0;
    for (n = 0; n < nAtom; n++)
    {
        vv = 0.0;
        for (k = 0; k < 3; k++)
            vv = vv + rv[n][k] * rv[n][k];
        kinEnergy = kinEnergy + vv;
    }
    kinEnergy *= (0.5 / nAtom);
    potEnergy /= nAtom;
    totEnergy = kinEnergy + potEnergy;
    temperature = kinEnergy * 2.0 / 3.0;

    /* Print the computed properties */
    printf("%9.6f %9.6f %9.6f %9.6f\n",
           stepCount * DELTAT, temperature, potEnergy, totEnergy);
}
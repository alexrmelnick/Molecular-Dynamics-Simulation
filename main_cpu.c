#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

#include "cpu.h"
#include "serial_baseline.c"
#include "parallel_baseline.c"

#include "serial_N3L.c"
#include "serial_N3L_AVX.c"
#include "parallel_N3L.c"

#include "parallel_cell.c"
#include "serial_cell.c"


#define OPTIONS 7        // The number of different variants of the simulation

// The number of particles in the simulation
// Number of particles = 4*(A * test_number * test_number + B * test_number + C)^3
// with A, B, C being the parameters of a quadratic function and test_number being a number in the range [0, NUM_TESTS)
#define A 0
#define B 2
#define C 2
#define NUM_TESTS_SHORT 11
#define NUM_TESTS_CELL 8
#define NUM_TESTS 11

/*
FUNCTIONS (TRIVIAL AND OPTIMIZED) TO BE TESTED
*/
void InitAll(int ideal_num_atoms);
void InitParams(int ideal_num_atoms);
void InitConf();

/*
HELPER FUNCTIONS FOR MORE ACCURATE TIMING
*/
double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}

double wakeup_delay()
{
    double meas = 0;
    int i, j;
    struct timespec time_start, time_stop;
    double quasi_random = 0;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    j = 100;
    while (meas < 1.0)
    {
        for (i = 1; i < j; i++)
        {
            /* This iterative calculation uses a chaotic map function, specifically
               the complex quadratic map (as in Julia and Mandelbrot sets), which is
               unpredictable enough to prevent compiler optimisation. */
            quasi_random = quasi_random * quasi_random - 1.923432;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        meas = interval(time_start, time_stop);
        j *= 2; /* Twice as much delay next time, until we've taken 1 second */
    }
    return quasi_random;
}

/*
MAIN FUNCTION CALL - TIMES ALL IMPLEMENTATIONS
*/
int main()
{
    int OPTION = 0;
    struct timespec time_start, time_stop;
    // cudaEvent_t start, stop;
    float time_stamp[OPTIONS][NUM_TESTS];
    double final_answer = 0;
    long int x, n;

    wakeup_delay();
    final_answer = wakeup_delay();

    Baseline
    OPTION = 0;
    printf("\n\nTesting option baseline serial\n\n");
    for (x = 0; x < NUM_TESTS_SHORT && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        final_answer += serial_base();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
    for (x = NUM_TESTS_SHORT; x < NUM_TESTS; x++)
    {
        time_stamp[OPTION][x] = -1;
    }

    // OPENMP - Baseline
    OPTION++;
    printf("\n\nTesting option Baseline with OpenMP Multi-Threading\n\n");
    for (x = 0; x < NUM_TESTS && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_REALTIME, &time_start);
        final_answer += parallel_base();
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }

    // Newton's 3rd Law
    OPTION++;
    printf("\n\nTesting option Newton's 3rd Law (USC)\n\n");
    for (x = 0; x < NUM_TESTS_SHORT && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        final_answer += serial_n3l();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
    for (x = NUM_TESTS_SHORT; x < NUM_TESTS; x++)
    {
        time_stamp[OPTION][x] = -1;
    }

    // Newton's 3rd Law with AVX
    OPTION++;
    printf("\n\nTesting option Newton's 3rd Law with AVX\n\n");
    for (x = 0; x < NUM_TESTS && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        final_answer += serial_AVX();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }

    // OPENMP - Newton's 3rd Law
    OPTION++;
    printf("\n\nTesting option Netwon's 3rd Law with OpenMP Multi-Threading\n\n");
    for (x = 0; x < NUM_TESTS && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_REALTIME, &time_start);
        final_answer += parallel_N3L();
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }

    // Cell List
    OPTION++;
    printf("\n\nTesting option Cell Lists (USC)\n\n");
    for (x = 0; x < NUM_TESTS_CELL && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        final_answer += serial_cell();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
    for (x = NUM_TESTS_CELL; x < NUM_TESTS; x++)
    {
        time_stamp[OPTION][x] = -1;
    }

    // Parallel Cell List
    OPTION++;
    printf("\n\nTesting option OpenMP Parallelized Cell List\n\n");
    for (x = 0; x < NUM_TESTS_CELL && (n = 4 * pow((A * x * x + B * x + C), 3)); x++)
    {
        InitAll(n);
        printf("\nTesting size %ld (%ld/%d)\n", n, x, NUM_TESTS);
        // printf("\nTime, temperature, potential energy, total energy\n");
        clock_gettime(CLOCK_REALTIME, &time_start);
        final_answer += parallel_cell();
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
    }
    for (x = NUM_TESTS_CELL; x < NUM_TESTS; x++)
    {
        time_stamp[OPTION][x] = -1;
    }

    /* output times */
    printf("\n\n# Atoms, Baseline, OpenMP Baseline, N3L, AVX N3L, OpenMP N3L, Cell List, OpenMP Cell List\n");
    {
        int i, j;
        for (i = 0; i < NUM_TESTS; i++)
        {
            printf("%.0f, ", 4 * pow((A * i * i + B * i + C), 3));
            for (j = 0; j < OPTIONS; j++)
            {
                if (j != 0)
                {
                    printf(", ");
                }
                printf("%.4f", time_stamp[j][i]);
            }
            printf("\n");
        }
    }

    printf("\n");
    printf("Sum of all results: %g\n", final_answer);

    return 0;
}

/*
INITIALIZATION
*/
void InitAll(int ideal_num_atoms)
{
    InitParams(ideal_num_atoms);
    InitConf();
    ComputeAccelN3L(); /* Computes initial accelerations */
}

void InitParams(int ideal_num_atoms)
{
    /*------------------------------------------------------------------------------
        Initializes parameters.
    ------------------------------------------------------------------------------*/
    int k;
    double rr, ri2, ri6, r1;

    double num_cells = cbrt(ideal_num_atoms / 4);
    if (num_cells < 1)
        num_cells = 1; // minimum 1 unit cell
    InitUcell[0] = (int)num_cells;
    InitUcell[1] = (int)num_cells;
    InitUcell[2] = (int)num_cells;

    /* Computes basic parameters */
    DeltaTH = 0.5 * DELTAT;
    for (k = 0; k < 3; k++)
    {
        Region[k] = InitUcell[k] / pow(DENSITY / 4.0, 1.0 / 3.0);
        RegionH[k] = 0.5 * Region[k];
    }

    // Compute the cell sizes for cell lists
    for (k = 0; k < 3; k++)
    {
        lc[k] = Region[k] / RCUT;
        rc[k] = Region[k] / lc[k];
    }

    /* Constants for potential truncation */
    rr = RCUT * RCUT;
    ri2 = 1.0 / rr;
    ri6 = ri2 * ri2 * ri2;
    r1 = sqrt(rr);
    Uc = 4.0 * ri6 * (ri6 - 1.0);
    Duc = -48.0 * ri6 * (ri6 - 0.5) / r1;
}

/*----------------------------------------------------------------------------*/
void InitConf()
{
    /*------------------------------------------------------------------------------
        r are initialized to face-centered cubic (fcc) lattice positions.
        rv are initialized with a random velocity corresponding to Temperature.
    ------------------------------------------------------------------------------*/
    double c[3], gap[3], e[3], vSum[3], vMag;
    int j, n, k, nX, nY, nZ;
    double seed;
    /* FCC atoms in the original unit cell */
    double origAtom[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

    /* Sets up a face-centered cubic (fcc) lattice */
    for (k = 0; k < 3; k++)
        gap[k] = Region[k] / InitUcell[k];
    nAtom = 0;
    for (nZ = 0; nZ < InitUcell[2]; nZ++)
    {
        c[2] = nZ * gap[2];
        for (nY = 0; nY < InitUcell[1]; nY++)
        {
            c[1] = nY * gap[1];
            for (nX = 0; nX < InitUcell[0]; nX++)
            {
                c[0] = nX * gap[0];
                for (j = 0; j < 4; j++)
                {
                    for (k = 0; k < 3; k++)
                        r[nAtom][k] = c[k] + gap[k] * origAtom[j][k];
                    ++nAtom;
                }
            }
        }
    }

    /* Generates random velocities */
    seed = 13597.0;
    vMag = sqrt(3 * INITTEMP);
    for (k = 0; k < 3; k++)
        vSum[k] = 0.0;
    for (n = 0; n < nAtom; n++)
    {
        RandVec3(e, &seed);
        for (k = 0; k < 3; k++)
        {
            rv[n][k] = vMag * e[k];
            vSum[k] = vSum[k] + rv[n][k];
        }
    }
    /* Makes the total momentum zero */
    for (k = 0; k < 3; k++)
        vSum[k] = vSum[k] / nAtom;
    for (n = 0; n < nAtom; n++)
        for (k = 0; k < 3; k++)
            rv[n][k] = rv[n][k] - vSum[k];
}

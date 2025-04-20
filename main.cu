#include <stdio.h>
#include <math.h>
#include <time.h>

#include "common.h"
#include "trivial.h"
#include "cell.h"

#define DIMENSIONS 2    // 2D simulation
#define NUM_TIME_STEPS 1000 // The number of time steps to simulate
//! Note: I have no idea how many time steps are needed or reasonable, I just picked a number
#define BOX_LENGTH 10.0f // The length of the box in which the particles are contained
#define OPTIONS 6 // The number of different variants of the simulation

// The number of particles in the simulation
// Number of particles = A * test_number * test_number + B * test_number + C
// with A, B, C being the parameters of a quadratic function and test_number being a number in the range [0, NUM_TESTS)
#define A 1
#define B 2
#define C 3
#define NUM_TESTS 20
#define MAX_NUM_PARTICLES (A * NUM_TESTS * NUM_TESTS + B * NUM_TESTS + C)
//! TODO: Determine a good parameters for this to test a wide range of hardware configurations (various cache sizes, etc.)

int main() {
    int option = 0;
    struct timespec time_start, time_stop;
    cudaEvent_t start, stop;
    float time_stamp[OPTIONS][NUM_TESTS];

    
}
# Molecular-Dynamics-Simulation

EC527 High Performance Programming with Multicore and GPUs Final Project: Molecular Dynamics Simulator with Cell Lists and Range-Limited Forces

## Project Overview

This project implements a molecular dynamics simulation of **liquid argon** using the **Lennard-Jones potential** for interatomic forces. There will be two versions of the simulation: a trivial O(N^2) version and an optimized O(N) version using cell lists. The simulation will be run on a CPU (both single-threaded and multi-threaded in **C** using **OpenMP**) and a GPU (using **CUDA**). Performance will be compared across these implementations, and the results will be visualized using **Python**.

## Acknowledgements

This project utilizes the following sources for reference and inspiration:
- [USC's Scientifc Computing and Visualization Course Website](https://aiichironakano.github.io/cs596-lecture.html)

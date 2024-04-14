# ANN-AMG

This repository contains codes accompanying the paper "A Deep Learning algorithm to accelerate Algebraic Multigrid (AMG) methods in Finite Element solvers of 3D elliptic PDEs", introducing a novel deep learning algorithm that minimizes the computational cost of the AMG method when used as a finite element solver. We show that our algorithm requires minimal changes to any existing code. The proposed Artificial Neural Network (ANN) tunes the value of the strong threshold parameter by interpreting the sparse matrix of the linear system as a gray scale image and exploiting a pooling operator to transform it into a small multi-channel image.

## Contents

The code is divided into two main folders: 
- the `data-generation` folder contains the code for solving various problems with the AMG method and thus building a dataset. It contains also two script to automate the process of data generation (compiling and running the C++ code) and pre-processing (compressing the generated data in train-validation-test datasets).
- the `data-modeling` folder contains the code for analyzing the data, building, training and evaluating the ANN used in the paper.

We also provide a `Dockerfile` in the `environment` folder to run all the code inside a pre-made CPU-only environment. However, this code was designed to run on a cluster. Thus, replication of the results in a Docker container may by prohibitively expensive or may not exactly match the one shown in the paper.

# Cite
Please cite this as:
```
M. Caldana, P.F. Antonietti and L. Dede', A deep learning algorithm to accelerate algebraic multigrid methods in finite element solvers of 3D elliptic PDEs, Computer and Mathematics with Applications, https://doi.org/10.1016/j.camwa.2024.05.013
```
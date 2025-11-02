# CUDA KNN Project - for fun

This project implements a **K-Nearest Neighbors (KNN)** classifier in both **CUDA/C++** and **Python (scikit-learn)**, and does a little benchmark of performance across different distance metrics and neighbor counts.

## Quick Start

CUDA project is built via cmake. To try it out yourself (ensure you have all dependencies etc. cmake/cuda toolchain etc)

```code
cd build
cmake ..
make
./CUDA_KNN <distance_function> <k neighbors>
```

For distance functions you can choose one out of {L1, L2, LINF, Cosine}. Also known as {Manhattan, euclidean, chebyshev, and cosine}. 

Data is generated in the csvs folder by running `cd csvs; python dataset_gen.py` feel free to play with parameters etc. 


### Structure

#### CUDA

`main.cu`: launches 2 kernels, one for computing distances and one for evaluating the nearest neighbors. 

1st kernel is in `distance_kernels/distance.cu`, which uses distances declared in `include/distance_kernels.h`. 

2nd kernel is in `evaluate/evaluation.cu`

#### Python

Basic sklearn KNN in root. Run via `python sklearn_imp.py <distance> <neighbors>`. 

#### Benchmark

Once you have built cuda repo you can run the baseline (`./benchmark.sh`) to compare the CPU python implementation by sklearn with my CUDA GPU implementation in terms of speed. 


### Benchmark Evaluation (USING TESLA V100 GPU)

I ran some benchmarks comparing speeds of my cuda implementation vs the default python implementation, which showed some pretty interesting timing differences. Output accuracy between both is the same. The sklearn implementation is better on smaller datasets, but fails to scale efficiently. 

Smaller train/test datasets display weaker improvement from GPU due to hinderance by launch/malloc/memcpy overhead. As the data expands GPU time doesn't change meaning parallelism is fully making up the difference. GPU is not fully utilized. CPU time explodes as the dataset increases.   

Train test split with 80% train 20% test. 4000 total elements, 10 features, 9 classes. 

```
$ ./benchmark.sh
Distance,K,CUDA_avg_ms,Python_avg_ms,Speedup
L1,1,120.92,67.46,0.56
L1,5,119.63,80.20,0.67
L1,10,119.62,81.44,0.68
L1,20,121.65,87.48,0.72
L1,43,125.95,89.46,0.71
L2,1,119.01,50.06,0.42
L2,5,120.19,63.10,0.53
L2,10,120.96,68.78,0.57
L2,20,120.39,75.74,0.63
L2,43,124.40,88.46,0.71
Cosine,1,117.53,65.22,0.55
Cosine,5,118.49,86.68,0.73
Cosine,10,118.34,88.78,0.75
Cosine,20,119.48,87.66,0.73
Cosine,43,126.53,91.92,0.73
LINF,1,117.01,30.72,0.26
LINF,5,120.98,37.92,0.31
LINF,10,123.31,42.86,0.35
LINF,20,120.42,47.68,0.40
LINF,43,125.23,58.04,0.46
```
```
Train test split with 80% train 20% test. 40000 total elements, 10 features, 9 classes. 

$ ./benchmark.sh

Distance,K,CUDA_avg_ms,Python_avg_ms,Speedup
L1,1,159.34,2729.64,17.13
L1,5,156.69,4216.08,26.91
L1,10,160.71,4910.34,30.55
L1,20,156.44,5710.90,36.51
L1,43,206.60,6644.38,32.16
L2,1,124.73,1154.90,9.26
L2,5,139.27,1851.60,13.29
L2,10,147.78,2291.24,15.50
L2,20,156.60,2841.48,18.14
L2,43,185.95,3660.30,19.68
Cosine,1,136.00,2041.28,15.01
Cosine,5,155.28,3990.58,25.70
Cosine,10,151.88,4015.86,26.44
Cosine,20,152.23,4003.90,26.30
Cosine,43,185.42,4020.96,21.69
LINF,1,132.95,447.88,3.37
LINF,5,138.76,674.38,4.86
LINF,10,144.08,816.94,5.67
LINF,20,158.30,1018.84,6.44
LINF,43,181.23,1352.32,7.46
```

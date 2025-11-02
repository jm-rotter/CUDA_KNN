#include "distance_kernels.h"

template <typename DistanceFunc>
__global__ void compute_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features){

	int i_train = blockIdx.x * blockDim.x + threadIdx.x;
	int i_test = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i_test >= n_test || i_train >= n_train) return;

	DistanceFunc distanceFunc;
	dists[i_train + i_test * n_train] = distanceFunc(n_features, &X_train[i_train * n_features], &X_test[i_test*n_features]);
}

template __global__ void compute_distance<L1Distance>(float *X_train, float *X_test, float *dists, int n_train, int n_test, int n_features);
template __global__ void compute_distance<L2Distance>(float *X_train, float *X_test, float *dists, int n_train, int n_test, int n_features);
template __global__ void compute_distance<Cosine>(float *X_train, float *X_test, float *dists, int n_train, int n_test, int n_features);
template __global__ void compute_distance<LInfDistance>(float *X_train, float *X_test, float *dists, int n_train, int n_test, int n_features);

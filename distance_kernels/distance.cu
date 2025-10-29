#include "distance_kernels.h"

template <typename DistanceFunc>
__global__ void compute_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features){

	int i_train = blockIdx.x * blockDim.x + threadIdx.x;
	int i_test = blockIdx.y * blockDim.y + threadIdx.y;

	
	if(i_test >= n_test || i_train >= n_train) return;

	dists[i_test + i_train*n_test] = DistanceFunc(n_features, i_train, i_test, float*);
}

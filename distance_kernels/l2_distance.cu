#include "distance_kernels.h"
__global__ void compute_l2_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features){

	int i_train = blockIdx.x * blockDim.x + threadIdx.x;
	int i_test = blockIdx.y * blockDim.y + threadIdx.y;

	if(i_test < n_test && i_train < n_train) {
		float dist = 0.0f;
		for(int k = 0; k < n_features; k++) {
			float diff = (X_test[i_test * n_features + k] - X_train[i_train *n_features + k]);
			dist += diff * diff;
		}
		dist = sqrtf(dist);
		dists[i_test + i_train*n_test] = dist;
	}
}

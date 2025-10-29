#pragma once
#include <math.h>

template <typename DistanceFunc>
__global__ void compute_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features);

struct L1Distance {
	__device__ float operator()(const int n_features, const float* X_train, const float* y_train){
		int distance = 0;
		for(int i = 0; i < n_features; i++){
			distance += fabsf(X_train[i]-y_train[i]);
		}
		return distance;
	}
};

struct L2Distance {
	__device__ float operator()(const int n_features, const float* X_train, const float y_train){
		float distance = 0;
		for(int i = 0; i < n_features; i++) {
			float square = X_train[i] - y_train[i];
			distance += square * square;
		}
		return sqrtf(distance);
	}
};

enum distanceType {L1, L2};

distanceType parse_distance_type(const std::string& arg);



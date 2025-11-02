#pragma once
#include <math.h>
#include <string>

template <typename DistanceFunc>
__global__ void compute_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features);

struct L1Distance {
	__device__ float operator()(const int n_features, const float* X_train, const float* X_test){
		float distance = 0.0f;
		for(int i = 0; i < n_features; i++){
			distance += fabsf(X_train[i]-X_test[i]);
		}
		return distance;
	}
};

struct L2Distance {
	__device__ float operator()(const int n_features, const float* X_train, const float* X_test){
		float distance = 0.0f;
		for(int i = 0; i < n_features; i++) {
			float square = X_train[i] - X_test[i];
			distance = fmaf(square, square, distance);
		}
		return sqrtf(distance);
	}
};

struct Cosine {
	__device__ float operator()(const int n_features, const float* X_train, const float* X_test){
		float dot = 0.0f; float norma = 0.0f; float normb = 0.0f;
		for(int i = 0; i < n_features; i++) {
			dot = fmaf(X_train[i], X_test[i], dot);
			norma = fmaf(X_train[i], X_train[i], norma);
			normb = fmaf(X_test[i], X_test[i], normb);
		}
		float denom = sqrtf(norma* normb);
		float cosine_sim = (denom > 0.5) ? dot/denom: 0.0f; 
		return 1.0 - cosine_sim;
	}
};

struct LInfDistance {
	__device__ float operator()(int n_features, const float* X_train, const float* X_test) {
		float max_diff = 0.0f;
		for(int i = 0; i < n_features; i++) {
			float diff = fabsf(X_train[i] - X_test[i]);
			if(diff > max_diff) max_diff = diff;
		}
		return max_diff;
	}
};

enum distanceType {L1, L2, COSINE, LINF};

distanceType parse_distance_type(const std::string& arg);

const char* distanceTypeToString(distanceType type);



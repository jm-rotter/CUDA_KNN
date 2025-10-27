__global__ void compute_l1_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features);

__global__ void compute_class(
		float *dists, int n_train, int n_test, int* y_train, int* y_pred, int k, int classes);

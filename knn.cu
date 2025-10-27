#define MAX_K 32
#define MAX_CLASSES 16


__global__ void compute_l1_distance(
		float *X_train, float *X_test, float *dists, 
		int n_train, int n_test, int n_features){

	int i_train = blockIdx.x * blockDim.x + threadIdx.x;
	int i_test = blockIdx.y * blockDim.y + threadIdx.y;

	if(i_test < n_test && i_train < n_train) {
		float dist = 0.0f;
		for(int k = 0; k < n_features; k++) {
			dist += fabs(X_test[i_test * n_features + k] - X_train[i_train *n_features + k]);
		}
		dists[i_test + i_train*n_test] = dist;
	}
		
}

__global__ void compute_class(
		float *dists, int n_train, int n_test, int* y_train, int* y_pred, int k, int classes) {
	int index = threadIdx.x + threadIdx.y*blockDim.x;

	if (index >= n_test) {
		return;
	}
	
	float topk_dist[MAX_K]; 
	int topk_labels[MAX_K];

	for(int i = 0; i < k; i++){
		topk_dist[i] = 1e10f;
		topk_labels[i] = -1;
	}

	for(int i = 0; i < n_train; i++) {
		float d = dists[index * n_train + i];

		int max_idx = 0; 

	    for(int j = 1; j < k; j++) {
			if(topk_dist[j] > topk_dist[max_idx]){
				max_idx = j;
			}
		}	

		if(d < topk_dist[max_idx]){
			topk_dist[max_idx] = d;
			topk_labels[max_idx] = y_train[i];
		}
	}
	
	int votes[MAX_CLASSES] = {0}; 
	for(int i = 0; i < k; i++) {
		if(topk_labels[i] >= 0){
			votes[topk_labels[i]]++;
		}
	}

	int max_votes = 0;
	int pred = -1;
	for(int i = 0; i < classes; i++) {
		if(votes[i] > max_votes){
			max_votes = votes[i];
			pred = i;
		}
	}

	y_pred[index] = pred;
	
}


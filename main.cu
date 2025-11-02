#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>


#include "distance_kernels.h"
#include "evaluation.h"
#include "globals.h"
#include "includes/distance_kernels.h"

//from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

std::vector<float> read_csv_float(const std::string& filename, int& n_rows, int* n_features) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<float> data;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    std::stringstream ss(line);
    std::string cell;
    *n_features = 0;
    while (std::getline(ss, cell, ',')) {
        data.push_back(std::stof(cell));
        (*n_features)++;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) {
            data.push_back(std::stof(cell));
        }
    }

    file.close();
    n_rows = data.size() / *n_features;
    return data;
}

std::vector<int> read_csv_int(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<int> data;
    std::string line;
    while(std::getline(file, line)) {
        if(line.empty()) continue;
        data.push_back(std::stoi(line));
    }
    file.close();

    return data;
}

int main(int argc, char* argv[]) {
	if(argc != 3){
		std::cout << "Expected: ./CUDA_KNN <Distance Function> <K Nearest Neighbors" << "\n";
		std::cout << "Distance Function options: L1, L2, ..." << "\n";
		std::cout << "KNN Options: 1<=x<=32" << "\n";
		std::cout << "./CUDA_KNN L1 2" << "\n";
		exit(1);

	}
	distanceType dist_type = parse_distance_type(argv[1]);
	int k = std::stoi(argv[2]);
	if (k < 1 || k > MAX_K) {
		std::cout << "Bounds not between MAX_K and 1 (See globals.h)\n";
		exit(1);
	}
	int n_train;
	int n_test;
	int n_features;


	//training data
	std::vector<float> X_train = read_csv_float(std::string(PROJECT_SOURCE_DIR) + "/csvs/X_train.csv", n_train, &n_features);
	std::vector<int> y_train = read_csv_int(std::string(PROJECT_SOURCE_DIR) + "/csvs/y_train.csv");

	std::vector<float> X_test = read_csv_float(std::string(PROJECT_SOURCE_DIR) + "/csvs/X_test.csv", n_test, &n_features);
	std::vector<int> y_test = read_csv_int(std::string(PROJECT_SOURCE_DIR) + "/csvs/y_test.csv");
	
	//std::cout << "Train: Read " << n_train << " rows of " << n_features << " features\n";
	//std::cout << "Test: Read " << n_test << " rows of " << n_features << " features\n";
	

	using namespace std::chrono;
	auto cpu_start = high_resolution_clock::now();
	
	float* d_X_train = nullptr;
	float* d_X_test = nullptr;
	int* d_y_train = nullptr;
	int* d_y_test = nullptr;
	float* d_dists = nullptr;
	int* d_y_pred = nullptr;

	std::vector<int> y_pred(n_test);

	gpuErrchk(cudaMalloc(&d_X_test, X_test.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_X_train, X_train.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_dists, n_train*n_test*sizeof(float)));

	gpuErrchk(cudaMemcpy(d_X_test, X_test.data(), X_test.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_X_train, X_train.data(), X_train.size() * sizeof(float), cudaMemcpyHostToDevice));

	int dimx = 32; 
	int dimy = 32;
	dim3 block_dim(dimx, dimy);
	dim3 grid_dim((n_train+dimx-1)/dimx, (n_test+dimy-1)/dimy);//Round up


	switch (dist_type) {
		case L1: 
			compute_distance<L1Distance><<<grid_dim, block_dim>>>(d_X_train, d_X_test, d_dists, n_train, n_test, n_features);
			break;
		case L2:
			compute_distance<L2Distance><<<grid_dim, block_dim>>>(d_X_train, d_X_test, d_dists, n_train, n_test, n_features);
			break;
		case COSINE:
			compute_distance<Cosine><<<grid_dim, block_dim>>>(d_X_train, d_X_test, d_dists, n_train, n_test, n_features);
			break;
		case LINF:
			compute_distance<LInfDistance><<<grid_dim, block_dim>>>(d_X_train, d_X_test, d_dists, n_train, n_test, n_features);
			break;
	}	

	
	gpuErrchk(cudaMalloc(&d_y_train, y_train.size() * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_y_pred, y_test.size() * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_y_test, y_test.size() * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y_test, y_test.data(), y_test.size() * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaDeviceSynchronize());

	int threads_per_block = 128;
	int num_blocks = (n_test + threads_per_block - 1)/threads_per_block;

	compute_class<<<num_blocks,threads_per_block>>>(d_dists, n_train, n_test, d_y_train, d_y_pred, k, MAX_CLASSES);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(y_pred.data(), d_y_pred, n_test * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(d_y_train));
	gpuErrchk(cudaFree(d_y_test));
	gpuErrchk(cudaFree(d_X_train));
	gpuErrchk(cudaFree(d_X_test));
	gpuErrchk(cudaFree(d_dists));
	gpuErrchk(cudaFree(d_y_pred));

	auto cpu_stop = high_resolution_clock::now();
	double total_ms = duration<double, std::milli>(cpu_stop - cpu_start).count();
	printf("CUDA time: %.3f ms\n", total_ms);

	int correct = 0;
	int incorrect = 0;
	for(int i = 0; i < n_test; i++) {
		if(y_pred[i] == y_test[i]){
			correct++;
		}
		else{incorrect++;}
	}

	//std::cout << "Prediction:" << std::endl;
	//for(int i=0; i<n_test; i++)
		//std::cout << y_pred[i] << " ";
	//std::cout << "\n";

	//std::cout << "Actual:" << std::endl;
	//for(int i=0; i<n_test; i++)
		//std::cout << y_test[i] << " ";
	//std::cout << "\n";

	std::cout << "Correct: " << correct << std::endl;
	std::cout << "Incorrect: " << incorrect << std::endl;

	std::cout << "Acc:" << (float)correct/(correct+incorrect) << std::endl;

    return 0;
}


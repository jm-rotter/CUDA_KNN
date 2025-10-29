#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>


#include "distance_kernels.h"
#include "evaluation.h"
#include "globals.h"

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

std::vector<float> read_csv_float(const std::string& filename, int& n_rows, int n_features) { std::ifstream file(filename);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<float> data;
    std::string line;
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')) {
            data.push_back(std::stof(cell));
        }
    }
    file.close();

    n_rows = data.size() / n_features;
    if(data.size() % n_features != 0){
        throw std::runtime_error("CSV data size is not divisible by n_features");
    }

    return data;
}

// Read CSV of ints (one value per line)
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

int main() {
	distanceType dist_type = parse_distance_type(argv[1]);
	int n_features = 4;
	int n_train;
	int n_test;

	//training data
	std::vector<float> X_train = read_csv_float(std::string(PROJECT_SOURCE_DIR) + "/csvs/X_train.csv", n_train, n_features);
	std::cout << "Read " << n_train << " rows of " << n_features << " features\n";
	std::vector<int> y_train = read_csv_int(std::string(PROJECT_SOURCE_DIR) + "/csvs/y_train.csv");
	std::cout << "Read " << y_train.size() << " labels\n";

	//test data
	std::vector<float> X_test = read_csv_float(std::string(PROJECT_SOURCE_DIR) + "/csvs/X_test.csv", n_test, n_features);
	std::cout << "Read " << n_test << " rows of " << n_features << " features\n";
	std::vector<int> y_test = read_csv_int(std::string(PROJECT_SOURCE_DIR) + "/csvs/y_test.csv");
	std::cout << "Read " << y_train.size() << " labels\n";
	

	// Print first row
	for(int i=0; i<n_features; i++)
		std::cout << X_train[i] << " ";
	std::cout << "\n";

	for(int i=0; i<n_features; i++)
		std::cout << X_test[i] << " ";
	std::cout << "\n";

	std::cout << y_train[0] << std::endl;
	std::cout << y_test[0] << std::endl;

	
	float* d_X_train = nullptr;
	float* d_X_test = nullptr;
	int* d_y_train = nullptr;
	int* d_y_test = nullptr;
	float* d_dists = nullptr;
	int* d_y_pred = nullptr;

	std::vector<int> y_pred(n_test);

	gpuErrchk(cudaMalloc(&d_X_test, X_test.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_X_train, X_train.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_y_test, y_test.size() * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_dists, n_train*n_test*sizeof(float)));

	gpuErrchk(cudaMemcpy(d_X_test, X_test.data(), X_test.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_X_train, X_train.data(), X_train.size() * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block_dim(16, 16);
	dim3 grid_dim((n_train+16-1)/16, (n_test+16-1)/16);//Round up

	compute_l1_distance<<<grid_dim, block_dim>>>(d_X_train, d_X_test, d_dists, n_train, n_test, n_features);



	gpuErrchk(cudaMalloc(&d_y_train, y_train.size() * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_y_pred, y_test.size() * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());


	//std::vector<float> l1_dist(n_train*n_test);
	
	//gpuErrchk(cudaMemcpy(l1_dist.data(), d_dists, l1_dist.size() * sizeof(float), cudaMemcpyDeviceToHost));

    //std::cout << "Res 1st ker\n";
	//for(int i = 0; i < l1_dist.size(); i++) {
		//std::cout << l1_dist[i] << ", ";
	//}
	//std::cout << '\n';

	gpuErrchk(cudaMemcpy(d_y_test, y_test.data(), y_test.size() * sizeof(int), cudaMemcpyHostToDevice));



	compute_class<<<1,block_dim>>>(d_dists, n_train, n_test, d_y_train, d_y_pred, 6, 3);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(y_pred.data(), d_y_pred, n_test * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());


	//acc
	int correct = 0;
	int incorrect = 0;
	for(int i = 0; i < n_test; i++) {
		if(y_pred[i] == y_test[i]){
			correct++;
		}
		else{incorrect++;}
	}

	std::cout << "Prediction:" << std::endl;
	for(int i=0; i<n_test; i++)
		std::cout << y_pred[i] << " ";
	std::cout << "\n";

	std::cout << "Actual:" << std::endl;
	for(int i=0; i<n_test; i++)
		std::cout << y_test[i] << " ";
	std::cout << "\n";

	std::cout << "Correct: " << correct << std::endl;
	std::cout << "Incorrect: " << incorrect << std::endl;

	std::cout << "Acc:" << (float)correct/(correct+incorrect) << std::endl;

    return 0;
}


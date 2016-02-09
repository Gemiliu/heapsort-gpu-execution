#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "check.h"
#include <iostream>
#include <math.h>

static const int blockSize = 127; // array size
static const int iterations = 10; // number of iterations
int countBlocks = 250;

inline void cudaCheck(const cudaError_t &err, const std::string &mes) {
	if (err != cudaSuccess) {
		std::cout << (mes + " - " + cudaGetErrorString(err)) << std::endl;
		exit(EXIT_FAILURE);
	}
}

__device__ void swap(float *a, float *b) {
	const float t = *a;
	*a = *b;
	*b = t;
}

__device__ void maxHeapify(float *maxHeap, int heapSize, int idx) {
	int largest = idx;  // Initialize largest as root
	int left = (idx << 1) + 1;  // left = 2*idx + 1
	int right = (idx + 1) << 1; // right = 2*idx + 2

	// See if left child of root exists and is greater than root
	if (left < heapSize && maxHeap[left] > maxHeap[largest]) {
		largest = left;
	}

	// See if right child of root exists and is greater than
	// the largest so far
	if (right < heapSize && maxHeap[right] > maxHeap[largest]) {
		largest = right;
	}

	// Change root, if needed
	if (largest != idx) {
		swap(&maxHeap[largest], &maxHeap[idx]);
		maxHeapify(maxHeap, heapSize, largest);
	}
}

// A utility function to create a max heap of given capacity
__device__ void createAndBuildHeap(float *array, int size) {
	// Start from bottommost and rightmost internal mode and heapify all
	// internal modes in bottom up way
	for (int i = (size - 2) / 2; i >= 0; --i) {
		maxHeapify(array, size, i);
	}
}

__global__ void heapSortKernel(float *iA, int size) {
	//A = A + blockIdx.x * blockSize;
	iA = iA + blockIdx.x * blockSize;
	__shared__ float A[blockSize];
	for (int i = threadIdx.x; i < blockSize; i += blockDim.x) {
		A[i] = iA[i];
	}
	__syncthreads();
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x == 0) {
		// Build a heap from the input data.
		createAndBuildHeap(A, size);

		// Repeat following steps while heap size is greater than 1.
		// The last element in max heap will be the minimum element
		int changedSizeOfHeap = size;
		while (changedSizeOfHeap > 1) {
			// The largest item in Heap is stored at the root. Replace
			// it with the last item of the heap followed by reducing the
			// size of heap by 1.
			swap(A, &A[changedSizeOfHeap - 1]);
			--changedSizeOfHeap;  // Reduce heap size

			// Finally, heapify the root of tree.
			maxHeapify(A, changedSizeOfHeap, 0);
		}
	}
	for (int i = threadIdx.x; i < blockSize; i += blockDim.x) {
		iA[i] = A[i];
	}
}

int main(int argc,char *argv[]) {
	if (argc == 2) {
		countBlocks = atoi(argv[1]);
	}
	std::cout << "count blocks = " << countBlocks << std::endl;

	cudaError_t err = cudaSuccess;
	// Print the vector length to be used, and compute its size
	int numElements = blockSize * countBlocks;
	size_t size = numElements * sizeof(float);

	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);
	if (h_A == NULL) {
		std::cout << "Failed to allocate host vectors!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);
	cudaCheck(err, "failed to allocated device vector A");

	double proccesingTime = 0.0f;
	double commonProccesingTime = 0.0f;
	double proccesingTimeWithCopying = 0.0f;
	for (int k = 0; k < iterations; ++k) {
		Time timer;
		timer.begin("common");
		// Initialize the host input vectors
		for (int i = 0; i < numElements; ++i) {
			//h_A[i] = rand()/(float)RAND_MAX;
			h_A[i] = rand() % 1000 + 1;
		}
		/*for (int i = 0; i < countBlocks; ++i) {
			for (int j = 0; j < blockSize; ++j) {
				const int index = i * blockSize + j;
				std::cout << h_A[index] << " ";
			}
			std::cout << std::endl;
		}*/

		timer.begin("with copying");
		err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		cudaCheck(err, "failed to copy vector A to device");

		// Launch the CUDA Kernel
		int threadsPerBlock = 32;
		int blocksPerGrid = countBlocks;
		timer.begin("sort");
		heapSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, blockSize);
		cudaDeviceSynchronize();
		timer.end("sort");
		err = cudaGetLastError();
		cudaCheck(err, "failed to launch kernel");

		err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
		cudaCheck(err, "failed to copy vector A to host");
		timer.end("with copying");
		/*for (int i = 0; i < countBlocks; ++i) {
			for (int j = 0; j < blockSize; ++j) {
				const int index = i * blockSize + j;
				std::cout << h_A[index] << " ";
			}
			std::cout << std::endl;
		}*/
		// Verify that the result vector is correct
		for (int i = 0; i < countBlocks; ++i) {
			for (int j = 0; j < (blockSize - 1); ++j) {
				const int index = i * blockSize + j;
				if ((h_A[index] - h_A[index + 1]) > 0.00001) {
					std::cout << "Result verification failed at element " << i << "!" << std::endl;
					exit(EXIT_FAILURE);
				}
			}
		}
		timer.end("common");
		proccesingTime +=timer.getTimeSecFloat("sort");
		proccesingTimeWithCopying +=timer.getTimeSecFloat("with copying");
		commonProccesingTime += timer.getTimeSecFloat("common");
	}
	std::cout << "Test PASSED" << std::endl;
	const float avgProccesingTime = proccesingTime / iterations;
	const float avgProccesingTimeWithCopying = proccesingTimeWithCopying / iterations;
	std::cout << "avg proccesing time = " << avgProccesingTime << " sec" << std::endl;
	std::cout << "avg proccesing time(with copying) = " << avgProccesingTimeWithCopying << " sec" << std::endl;
	const int countOperations = countBlocks * (blockSize * log2f(blockSize));
	std::cout << "Computational throughput = " << countOperations / (avgProccesingTime * 10e6) << " MB/s" << std::endl;
	std::cout << "Computational throughput(with copying) = " << countOperations / (avgProccesingTimeWithCopying * 10e6) << " MB/s" << std::endl;

    // Free device global memory
    err = cudaFree(d_A);
	cudaCheck(err, "failed to free device vector A");
	// Free host memory
    free(h_A);
    err = cudaDeviceReset();
  	cudaCheck(err, "failed to deinitialize the device");

    std::cout << "Done." << std::endl;
    return 0;
}

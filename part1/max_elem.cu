#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <mutex>

using namespace std;

mutex mtx;

__global__ void find_maximum_kernel(double *array, double *max, unsigned int n);

int main()
{
  // Declare arrays, mutex, and size
	unsigned int N = 1024*1024*20;
	double *h_array, *d_array, *h_max, *d_max;

  // Declare timers
  float cuda_elapsed_time;
	cudaEvent_t cuda_start, cuda_stop;
  clock_t seq_start, seq_stop, seq_elapsed_time;

  cout << "Enter size of array: ";
  cin >> N;

	// allocate memory
	h_array = (double*)malloc(N*sizeof(double));
	h_max = (double*)malloc(sizeof(double));
	cudaMalloc((void**)&d_array, N*sizeof(double));
	cudaMalloc((void**)&d_max, sizeof(double));
	cudaMemset(d_max, 0, sizeof(double));

	// fill host array with data
	for(unsigned int i=0;i<N;i++){
		h_array[i] = N*double(rand()) / RAND_MAX;
	}

	// set up timing variables
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);

	// copy from host to device
	cudaEventRecord(cuda_start, 0);
	cudaMemcpy(d_array, h_array, N*sizeof(double), cudaMemcpyHostToDevice);

	// START CUDA
  kernel_max_wrapper(d_array, d_max, N);

	// copy from device to host
	cudaMemcpy(h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(cuda_stop, 0);
	cudaEventSynchronize(cuda_stop);
	cudaEventElapsedTime(&cuda_elapsed_time, cuda_start, cuda_stop);
	cudaEventDestroy(cuda_start);
	cudaEventDestroy(cuda_stop);

  cout << "----------------------------------------------------------" << endl;
  cout << "Max: " << *h_max << endl;
  cout << "[CUDA] Elapsed time: " << cuda_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << endl;

  cout << "Starting sequential version." << endl;

	seq_start = clock();

	*h_max = -1000000000.0;
	for(unsigned int j=0;j<N;j++){
		if(h_array[j] > *h_max){
			*h_max = h_array[j];
		}
	}

	seq_stop = clock();
	seq_elapsed_time = 1000*(seq_stop - seq_start)/CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "Max: " << *h_max << endl;
  cout << "[SEQUENTIAL] Elapsed time: " << seq_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

	free(h_array);
	free(h_max);
	cudaFree(d_array);
	cudaFree(d_max);

  return 0;
}

void kernel_max_wrapper(double *arr, double *max, unsigned int N)
{
  // 1 dimensional
  dim3 gridSize = 256;
  dim3 blockSize = 256;
  find_maximum_kernel<<< gridSize, blockSize >>>(arr, max, N);
}

__global__ void find_maximum_kernel(double *arr, double *max, unsigned int N)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

	double temp = -1.0;
	while(index + offset < N){
		temp = max(temp, arr[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = max(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		mtx.lock();
		*max = max(*max, cache[0]);
		mtx.unlock();
	}
}

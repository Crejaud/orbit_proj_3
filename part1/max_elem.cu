#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <mutex>

using namespace std;

__global__ void find_maximum_kernel(double *array, double *max, int *mtx, unsigned int n);

int main()
{
  // Declare arrays, mutex, and size
	// default size is 20971520
	unsigned int N = 20971520;
	double *seq_array, *cuda_array, *seq_max, *cuda_max;
	int *mtx;

  // Declare timers
  float cuda_elapsed_time;
	cudaEvent_t cuda_start, cuda_stop;
  clock_t seq_start, seq_stop, seq_elapsed_time;

  cout << "Enter size of array: ";
  cin >> N;

	// allocate memory for seq
	seq_array = (double*)malloc(N*sizeof(double));
	seq_max = (double*)malloc(sizeof(double));

	// set array of seq to random double values
	for(unsigned int i=0; i<N; i++){
		seq_array[i] = N*double(rand()) / RAND_MAX;
	}

	// allocate memory for cuda
	cudaMalloc((void**)&cuda_array, N*sizeof(double));
	cudaMalloc((void**)&cuda_max, sizeof(double));
	cudaMalloc((void**)&mtx, sizeof(int));

	// set values of max and mtx to all 0
	cudaMemset(cuda_max, 0, sizeof(double));
	cudaMemset(mtx, 0, sizeof(int));

	// set up timing variables
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);

	// copy from host to device
	cudaEventRecord(cuda_start, 0);
	cudaMemcpy(cuda_array, seq_array, N*sizeof(double), cudaMemcpyHostToDevice);

	// START CUDA
  kernel_max_wrapper(cuda_array, cuda_max, mtx, N);

	// copy from device to host
	cudaMemcpy(seq_max, cuda_max, sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(cuda_stop, 0);
	cudaEventSynchronize(cuda_stop);
	cudaEventElapsedTime(&cuda_elapsecuda_time, cuda_start, cuda_stop);

	// destroy timers
	cudaEventDestroy(cuda_start);
	cudaEventDestroy(cuda_stop);

  cout << "----------------------------------------------------------" << endl;
  cout << "Max: " << *seq_max << endl;
  cout << "[CUDA] Elapsed time: " << cuda_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << endl;

  cout << "Starting sequential version." << endl;

	seq_start = clock();

	*seq_max = -1000000000.0;
	for(unsigned int j=0;j<N;j++){
		if(seq_array[j] > *seq_max){
			*seq_max = seq_array[j];
		}
	}

	seq_stop = clock();
	seq_elapsed_time = 1000*(seq_stop - seq_start)/CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "Max: " << *seq_max << endl;
  cout << "[SEQUENTIAL] Elapsed time: " << seq_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

	// free and cuda free
	free(seq_array);
	free(seq_max);
	cudaFree(cuda_array);
	cudaFree(cuda_max);

  return 0;
}

void kernel_max_wrapper(double *arr, double *max, int *mtx, unsigned int N)
{
  // 1 dimensional
  dim3 gridSize = 256;
  dim3 blockSize = 256;
  find_maximum_kernel<<< gridSize, blockSize >>>(arr, max, mtx, N);
}

__global__ void find_maximum_kernel(double *arr, double *max, int *mtx, unsigned int N)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int span = gridDim.x*blockDim.x;

	__sharecuda__ double cache[256];

	double temp = -1000000000.0;
	for (unsigned int offset = 0; index + offset < N; offset += span)
		temp = max(temp, arr[index + offset]);

	cache[threadIdx.x] = temp;

	__syncthreads();

	// cuda reduction
	for (unsigned int i = blockDim.x/2; i != 0; i /= 2) {
		if (threadIdx.x < i)
			cache[threadIdx.x] = max(cache[threadIdx.x], cache[threadIdx.x + i]);
		__syncthreads();
	}

	// atomic setting of max!
	if(threadIdx.x == 0){
		// lock mtx
		while(atomicCAS(mtx, 0, 1) != 0);
		// set max!
		*max = max(*max, cache[0]);
		// unlock mtx
		atomicExch(mtx, 0);
	}
}

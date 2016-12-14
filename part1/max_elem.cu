#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>

using namespace std;

void kernel_max_wrapper(unsigned long long *arr, unsigned long long *max, int *mtx, unsigned long long int N);
__global__ void find_maximum_kernel(unsigned long long *arr, unsigned long long *max, int *mtx, unsigned long long int N);

int main()
{
  // Declare arrays, mutex, and size
	// default size is 20971520
	unsigned long long int N = 20971520;
	unsigned long long *seq_array, *cuda_array, *seq_max, *cuda_max;
	int *mtx;

 	 // Declare timers
  	float cuda_elapsed_time;
	cudaEvent_t cuda_start, cuda_stop;
 	double seq_start, seq_stop, seq_elapsed_time;

 	cout << "Enter size of array: ";
  	cin >> N;

	// allocate memory for seq
	seq_array = (unsigned long long*)malloc(N*sizeof(unsigned long long));
	seq_max = (unsigned long long*)malloc(sizeof(unsigned long long));

	srand(time(0));

	// set array of seq to random double values
	for(unsigned long long int i=0; i<N; i++){
		seq_array[i] = ((unsigned long long)rand() /((unsigned long long) RAND_MAX / (10000000000000.0)));
	}

	// allocate memory for cuda
	cudaMalloc((void**)&cuda_array, N*sizeof(unsigned long long));
	cudaMalloc((void**)&cuda_max, sizeof(unsigned long long));
	cudaMalloc((void**)&mtx, sizeof(int));

	// set values of max and mtx to all 0
	cudaMemset(cuda_max, 0, sizeof(unsigned long long));
	cudaMemset(mtx, 0, sizeof(int));

	// set up timing variables
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);
	cudaMemcpy(cuda_array, seq_array, N*sizeof(unsigned long long), cudaMemcpyHostToDevice);


	// copy from host to device
	cudaEventRecord(cuda_start, 0);

	// START CUDA
  	kernel_max_wrapper(cuda_array, cuda_max, mtx, N);

	// copy from device to host
	cudaEventRecord(cuda_stop, 0);
	cudaEventSynchronize(cuda_stop);
	cudaEventElapsedTime(&cuda_elapsed_time, cuda_start, cuda_stop);
	cudaMemcpy(seq_max, cuda_max, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	
	// destroy timers
	cudaEventDestroy(cuda_start);
	cudaEventDestroy(cuda_stop);

  	cout << "----------------------------------------------------------" << endl;
  	cout << "Max: " << *seq_max << endl;
  	cout << "[CUDA] Elapsed time: " << cuda_elapsed_time << " clock cycles" << endl;
  	cout << "----------------------------------------------------------" << endl;

  	cout << endl;

  	cout << "Starting sequential version." << endl;

	seq_start = (double) clock();

	*seq_max = 0;
	for(unsigned long long int j = 0; j < N ; j++){
		if(seq_array[j] > *seq_max){
			*seq_max = seq_array[j];
		}
	}

	seq_stop = (double) clock();
	seq_elapsed_time = (double) (seq_stop - seq_start)/CLOCKS_PER_SEC;
	seq_elapsed_time *= 1000.0;

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

void kernel_max_wrapper(unsigned long long *arr, unsigned long long *max, int *mtx, unsigned long long int N)
{
  // 1 dimensional
  dim3 gridSize = (N + 512 * 2048 - 1) / (512 * 2048);
  dim3 blockSize = 256;
  find_maximum_kernel<<< gridSize, blockSize >>>(arr, max, mtx, N);
}

__global__ void find_maximum_kernel(unsigned long long *arr, unsigned long long *max, int *mtx, unsigned long long int N)
{
	long long index = threadIdx.x + blockIdx.x*blockDim.x;
	long long span = gridDim.x*blockDim.x;

	__shared__ unsigned long long cache[256];

	unsigned long long temp = 0;
	for (unsigned long long int offset = 0; index + offset < N; offset += span) {
		if (temp < arr[index+offset]) {
			temp = arr[index+offset];
		}
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// cuda reduction
	for (unsigned long long int offset = blockDim.x/2; offset != 0; offset /= 2) {
		if (threadIdx.x < offset) {
			if (cache[threadIdx.x] < cache[threadIdx.x + offset]) {
				cache[threadIdx.x] = cache[threadIdx.x + offset];
			}
		}
		__syncthreads();
	}

	// atomic setting of max!
	if(threadIdx.x == 0){
		// lock mtx
		while(atomicCAS(mtx, 0, 1) != 0);
		if (*max < cache[0]) {
			*max = cache[0];
		}
		// unlock mtx
		atomicExch(mtx, 0);
	}
}

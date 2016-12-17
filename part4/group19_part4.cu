#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <time.h>

using namespace std;

//swap two int arrays
void swapPtrs(int **A, int **B){
  int *temp = *A;
  *A = *B;
  *B = temp;
}

//clear a cuda array to -1
__global__ void cudaClear(int* dev_clear, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<size){
		dev_clear[index]=-1;;
	}
	__syncthreads();
}

//does not modify dev_dups or dev_prefix, puts into dev_out the arrayB
__global__ void cudaArrayCopy(int* dev_orig, int* dev_dupli, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<size){
		dev_dupli[index]=dev_orig[index];
	}
	__syncthreads();
}

//assuming that the input dev_out is array A
__global__ void arrayC(int* dev_dups, int* dev_out, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(dev_dups[index]==1 && index<size){
		dev_out[index]=-1;
	}

	__syncthreads();
}

//generate array B
__global__ void arrayB(int* dev_dups, int* dev_prefix, int* dev_out, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(dev_dups[index]==1 && index<size){
		dev_out[dev_prefix[index]-1]=index;
	}

	__syncthreads();
}


//does not modify dev_in, puts into dev_out the find_dups array
__global__ void findDups(int* dev_in, int* dev_out, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if((index<size-1) && (index>=0)){
		if(dev_in[index]==dev_in[index+1]){
			dev_out[index]=1;
		}
		else{
			dev_out[index]=0;
		}
	}
	else if(index==size-1){
		dev_out[index]=0;
	}
	__syncthreads();
}

//see wrapper
__global__ void prefixSum(int* dev_in, int* dev_out, int twopwr, int size){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if((index<size) && (index>=twopwr)){
		dev_out[index]=dev_in[index]+dev_in[index-twopwr];
	}
	else if(index<twopwr){
		dev_out[index]=dev_in[index];
	}
	__syncthreads();
}

//calls prefixSum. Modifies dev_in!!! puts into dev_out the prefix sum array
void prefixSumWrapper(int* dev_in, int* dev_out, int log2size, int size){
	int twopwr;
	for(int i=0; i<log2size; ++i){
		twopwr=(int) pow(2,i);
		prefixSum<<<(size/256)+1,256>>>(dev_in, dev_out, twopwr, size);
		//flip array pointers so that we can avoid allocating a temp array
		//and copying back and forth between temp and orignals
		//bad side effect, dev_in will be gibberish after this function
		swapPtrs(&dev_in, &dev_out);
	}
	swapPtrs(&dev_in, &dev_out);
}

int main(){
	//generate cuda timers
	float cuda_elapsed_time, cuda_time_real;
	cudaEvent_t cuda_start, cuda_stop, cuda_real_start;
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_real_start);
	cudaEventCreate(&cuda_stop);
	//start recording total time
	cudaEventRecord(cuda_real_start, 0);

	//file stuff
	std::ofstream afile, bfile, cfile;
	remove("Adata.txt");
	remove("Bdata.txt");
	remove("Cdata.txt");
	afile.open("Adata.txt", std::ofstream::out | std::ofstream::app);
	bfile.open("Bdata.txt", std::ofstream::out | std::ofstream::app);
	cfile.open("Cdata.txt", std::ofstream::out | std::ofstream::app);

	//inits and allocs
	int *in, *dev_in, *dev_out, size, log2size, *dev_exc, *dev_orig, *temp;
	//powers of 2 only, 2^20 = 1,048,576
	size=(int) pow(2,20);
	log2size=(int)log2((float)size);
	
	in = (int*)malloc(size*sizeof(int));
	temp = (int*)malloc(size*sizeof(int));
	cudaMalloc(&dev_in, size*sizeof(int));
	cudaMalloc(&dev_out, size*sizeof(int));
	cudaMalloc(&dev_exc, size*sizeof(int));
	cudaMalloc(&dev_orig, size*sizeof(int));

	//gen nums
	srand(time(NULL));
	for(int i=0; i<size; ++i){
		in[i]=rand()%101;
		afile<<in[i]<<"\n";
		
	}

	//dev_exc contains the prefix sum in dev_exc for the array in
	//put input data into dev_in
	cudaMemcpy(dev_in, in, size*sizeof(int), cudaMemcpyHostToDevice);

	cudaArrayCopy<<<(size/256)+1,256>>>(dev_in, dev_orig, size);

	//start recording actual algorithm after initialization
	cudaEventRecord(cuda_start, 0);

	//into dev_out put the find_repeats
	findDups<<<(size/256)+1,256>>>(dev_in, dev_out, size);
	//into dev_in put find_repeats
	cudaArrayCopy<<<(size/256)+1,256>>>(dev_out, dev_in, size);
	//now, dev_exc will be the prefix sum and dev_in gibberish
	prefixSumWrapper(dev_in, dev_exc, log2size, size);
	//therefore clear dev_in for reuse
	cudaClear<<<(size/256)+1,256>>>(dev_in,size);
	//and then put array B into dev_in
	arrayB<<<(size/256)+1,256>>>(dev_out, dev_exc, dev_in, size);
	//now, the duplicate indexes (Array B) is in dev_in
	//the end of the array is signaled by -1
	//generated array c with -1 at indexes to ignore
	arrayC<<<(size/256)+1,256>>>(dev_out, dev_orig, size);
	
	//stop recording time after algorithm is complete	
	cudaEventRecord(cuda_stop, 0);
	
	//print final duplicate item
	cudaMemcpy(temp, dev_out, size*sizeof(int), cudaMemcpyDeviceToHost);
	for(int ty=size-1; ty>=0; --ty)
		{
		if(temp[ty]==1)
		{
			printf("The final duplicate is : %d at index %d\n", in[ty], ty);
			break;
		}
	}

	cudaMemcpy(in, dev_in, size*sizeof(int), cudaMemcpyDeviceToHost);

	//write results to file for array B
	for(int q=0; q<size; ++q){
		if(in[q]!=-1){
			bfile<<in[q]<<"\n";
		}
	}
	cudaMemcpy(in, dev_orig, size*sizeof(int), cudaMemcpyDeviceToHost);
	//write results to file for C
	for(int q=0; q<size; ++q){
		if(in[q]!=-1){
			cfile<<in[q]<<"\n";
		}
	}

	//print time
	cudaEventElapsedTime(&cuda_elapsed_time, cuda_start, cuda_stop);
	cudaEventElapsedTime(&cuda_time_real, cuda_real_start, cuda_stop);
	
	printf("Total cycles including memory allocation and memcopy\nTotal cuda clock cycles : %f\nAlgorithm only cuda clock cycles : %f\n", cuda_time_real, cuda_elapsed_time); 


}

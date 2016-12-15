#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>

using namespace std;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

void swapPtrs(int **A, int **B){
  int *temp = *A;
  *A = *B;
  *B = temp;
}

//twopoffset is 2^offset, offset is specificied in main
__global__ void prefixSum(int* datain, int* dataout, int twopoffset, int size)
 {
	int index=threadIdx.x;
	
	if((index<size) && (index>=twopoffset)){
		dataout[index]=datain[index]+datain[index-twopoffset];
	}
	else if(index<twopoffset){
		dataout[index]=datain[index];
	}
		printf("%d at %d\n", dataout[index], index);
	__syncthreads();
}


int main(){
	int *datain, *dataout, *data, *temp;
	int size=8;
	int log2size=(int) log2((float)size);
	//allocate memory for local & GPU data
	data = (int*)malloc(size*sizeof(int));
	temp = (int*)malloc(size*sizeof(int));

	cudaMalloc(&datain, size*sizeof(int));
	cudaMalloc(&dataout, size*sizeof(int));
	cudaCheckErrors("cudamalloc fail");

	//generate numbers
	for(int i=0; i<size; ++i){
		data[i]=rand()%11;
		cout<<data[i]<<" ";
	}
	std::cout<<"\n";

	//put data to device
	cudaMemcpy(datain, data,  size*sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudamemcpy or cuda kernel fail");

	//for each offset
	for(int i=0;i<log2size;++i){
		int t=(int) pow(2,i);
		prefixSum<<<1, 8>>>(datain, dataout, t, size);
		
		//recall the data
		cudaMemcpy(data, dataout, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout<<"after dataout->data\n";
		for(int i=0; i<size;++i){
			std::cout<<data[i]<<" ";
		}
		std::cout<<"\n";
		cudaMemcpy(datain, data,  size*sizeof(int), cudaMemcpyHostToDevice);
		//swapPtrs(&datain, &dataout);
	}

	std::cout<<"\n";
	cudaMemcpy(data, datain, sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<size;++i){
		std::cout<<data[i]<<" ";
	}
	std::cout<"\n";

}

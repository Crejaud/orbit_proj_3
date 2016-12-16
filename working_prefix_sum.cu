#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>

using namespace std;

void swapPtrs(int **A, int **B){
  int *temp = *A;
  *A = *B;
  *B = temp;
}

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

int main(){
	//file stuff
	std::ofstream outfile;
	outfile.open("inputs.txt", std::ofstream::out | std::ofstream::app);

	std::ofstream donefile;
	donefile.open("results.txt", std::ofstream::out | std::ofstream::app);

	//inits and allocs
	int *in, *out, *dev_in, *dev_out, size, log2size, twopwr;
	//powers of 2 only
	//why? not sure, seems to be an issue with
	//the offset algorithm used as a base
	//regardless, if 10000 is used, anything aboe 8192
	//(the closest power of 2) will give an incorrect sum
	//but 16384 works fine, being a power of 2
	size=1048576;
	log2size=(int)log2((float)size);
	
	in = (int*)malloc(size*sizeof(int));
	out = (int*)malloc(size*sizeof(int));

	cudaMalloc(&dev_in, size*sizeof(int));
	cudaMalloc(&dev_out, size*sizeof(int));

	//gen nums
	for(int i=0; i<size; ++i){
		in[i]=rand()%5;
		outfile<<in[i]<<"\n";
		
	} cout<<"\n";outfile.close();

	cudaMemcpy(dev_in, in,  size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out, out,  size*sizeof(int), cudaMemcpyHostToDevice);

	//loop up the chain
	for(int i=0; i<log2size; ++i){
		twopwr=(int) pow(2,i);
		printf("%d, %d\n", i, twopwr);
		prefixSum<<<(size/256)+1,256>>>(dev_in, dev_out, twopwr, size);
		//cudaMemcpy(out, dev_out,  size*sizeof(int), cudaMemcpyDeviceToHost);
//		for(int j=0; j<size; ++j){
//			in[j]=out[j];
//		}
		//cudaMemcpy(dev_in, out,  size*sizeof(int), cudaMemcpyHostToDevice);
		swapPtrs(&dev_in, &dev_out);
	}

	cudaMemcpy(out, dev_in,  size*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<size;++i){
		donefile<<out[i]<<"\n";
		
	} cout<<out[size-1]<<"\n";donefile.close();
}

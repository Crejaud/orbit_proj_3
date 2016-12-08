#include <curand.h>
#include <curand_kernel.h>
#include <conio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>

using namespace std;

#define MAX 100

int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

__global__ void random(int* res) {
  curandState_t state;
  curand_init(0, 0, 0, &state);
  *result = curand(&state) % MAX;
}

__global__ void generate_in_a_b(float *A, float a, float b, int nr_rows_A, int nr_cols_A) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < nr_rows_A*nr_cols_A) A[tid] = (b-a) * A[tid] + a;

}

__global__ void MatMulKernel(float* d_A, float* d_B, float* d_C, int height, int width) {
  __shared__ float Ads[width][height];
  __shared__ float Bds[width];
  __shared float partialSum[width][height];

  int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x;

  Ads[tx][ty] = d_A[tx * width + ty];
  if (tx == 0) d_B[ty] = B[ty * width + bx];
  __syncthreads();

  partialSum[tx][ty] = Ads[tx][ty] * Bds[ty];
  __syncthreads();

  if (ty < 4) partialSum[tx][ty] += partialSum[tx][ty + 4];
  if (ty < 2) partialSum[tx][ty] += partialSum[tx][ty + 2];
  if (ty == 0) d_C[tx * width + bx] = (partialSum[tx][ty] + partialSum[tx][ty + 1]);
}

void MatrixMultiplication(float *A, float *B, float *C, int height, int width) {
  int size = width * height * sizeof(float);
  float *Ad, *Bd, *Cd;

  cudaMalloc((void**) &Ad, size);
  cudaMalloc(Ad, A, size, cudaMemcpyHostToDevice);
  cudaMalloc((void**)*Bd, size);
  cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&Cd, size);
  cudaMemset(Cd, 0, size);

  dim3 dimGrid(width,1,1);
  dim3 dimBlock(width, height);

  MatMulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, height, width);

  cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
}


int main(void)
{
    float   *hst_Mat , *dev_Mat, *another_Mat, *devTwo_Mat;

    int* Height;
    int* Width;
    cudaMalloc((void**) &Height, sizeof(int));
    cudaMalloc((void**) &Width, sizeof(int));
    random<<<1,1>>>(Height);
    random<<<1,1>>>(Width);
    int h;
    int w;
    cudaMemcpy(&h, Height, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&w, Width, sizeof(int), cudaMemcpyDeviceToHost);
    int vSize = h*w;
    int mSize = sizeof(float)*vSize ;


    hst_Mat = (float *)malloc(mSize) ;
    cudaMalloc((void**)&dev_Mat, mSize) ;

    another_Mat = (float *)malloc(mSize);
    cudaMalloc((void**)&devTwo_Mat, mSize);

    memset(hst_Mat, 0, mSize) ;
    cudaMemset(dev_Mat, 0, mSize) ;

    memset(another_Mat, 0, mSize);
    cudaMemset(devTwo_Mat, 0, mSize);

    GPU_fill_rand(dev_Mat, h, w) ;
    GPU_fill_rand(devTwo_Mat, h, w);

    dim3 threads(32);
    dim3 blocks(iDivUp(h*w, 32));

    float a = 3.f;
    float b = 7.f;

    generate_in_a_b<<<blocks,threads>>>(dev_Mat,a,b,h,w);
    generate_in_a_b<<<blocks,threads>>>(devTwo_Mat,a,b,h,w);

    cudaMemcpy(hst_Mat, dev_Mat, mSize, cudaMemcpyDeviceToHost) ;
    cudaMemcpy(another_Mat, devTwo_Mat, mSize, cudaMemcpyDeviceToHost);

    unsigned int mem_size_P = vSize * sizeof(float);
    float* hostP = (float*) malloc(mem_size_P);
    MatrixMultiplication(hst_Mat, another_Mat, hostP, h, w);
    /*
    cout << " * Result matrix : " << endl << "     " ;
    for(int i=0 ;i<h ; i++)
    {
        for(int j=0 ; j<w ; j++)
            cout << "   " << hst_Mat[i*Width+j] ;
            cout << endl << "     " ;
    }
    cout << endl << endl ;
    */

    free(hst_Mat) ;
    free(another_Mat);
    free(hostP);
    cudaFree(dev_Mat) ;
    cudaFree(devTwo_Mat);

    system("pause") ;

    return 0;
}

#include <iostream>
#include <cublas_v2.h>
using std::cout;

int BLOCK_MAX_THREADS = 512;

double random(float start, float end)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    float r = random * (end - start);
    return start + r;
}

void createArrayWithRandomValues(float* inputArray, int size)
{
  srand(time(NULL));
  int i = 0;
  while(i<size)
  {
    inputArray[i] = random(0,10);
    i++;
  }
}

__global__ void
MatrixMultKernel(float* d_A, float* d_B, float* d_C, int rowsA, int columnsB, int denom)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int size = rowsA * columnsB;
  if(index  < size)
  {
    float dotProduct = 0;
    int rowIndex = index / columnsB; //which row of A
    int columnIndex = index % columnsB; //which column of B
    int rowIndexA = rowIndex * denom;
    for(int i=0; i<denom; i++)
    {
      float row = d_A[rowIndexA+i];
      float column = d_B[columnIndex + (columnsB * i)];
      int prod = row * column;
      dotProduct = dotProduct + prod;
    }
    d_C[index] = dotProduct;
  }
  __syncthreads();
}


void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
   int lda=m,ldb=k,ldc=m;
   const float alf = 1;
   const float bet = 0;
   const float *alpha = &alf;
   const float *beta = &bet;
   cublasHandle_t handle;
   cublasCreate(&handle);
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, lda, A, ldb, beta, C, ldc);
   cublasDestroy(handle);
}

int main()
{

  float cuda_elapsed_time, cuda_elapsed_time2;
	cudaEvent_t cuda_start, cuda_start2, cuda_stop, cuda_stop2;
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_stop);
  cudaEventCreate(&cuda_start2);
  cudaEventCreate(&cuda_stop2);

  int rowsA = 300;
  int columnsA = 200;
  int sizeA = rowsA*columnsA;
  int rowsB  = 200;
  int columnsB = 400;
  int sizeB = rowsB*columnsB;
  int sizeC = rowsA*columnsB;

  float* matrixA = new float[sizeA];
  float* matrixB = new float[sizeB];
  float* matrixC = new float[sizeC];

  createArrayWithRandomValues(matrixA, sizeA);
  createArrayWithRandomValues(matrixB, sizeB);
  /* uncomment to see inputs
  cout<<"Matrix A: \n";
  for(int i=0; i<sizeA; i++)
  {
    cout<<matrixA[i]<<" ";
  }
  cout<<"\n";
  cout<<"Matrix B: \n";
  for(int i=0; i<sizeB; i++)
  {
    cout<<matrixB[i]<<" ";
  }
  cout<<"\n";
  */

  float* dmA;
  float* dmB;
  float* dmC;

  cudaMalloc((void**) &dmA, sizeof(float)*sizeA);
  cudaMemcpy(dmA, matrixA, sizeof(float)*sizeA, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmB, sizeof(float)*sizeB);
  cudaMemcpy(dmB, matrixB, sizeof(float)*sizeB, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmC, sizeof(float)*sizeC);
  cudaMemcpy(dmC, matrixC, sizeof(float)*sizeC, cudaMemcpyHostToDevice);

  int spb = sizeC + (BLOCK_MAX_THREADS - 1);
  int numBlocks = spb / BLOCK_MAX_THREADS;
  cudaEventRecord(cuda_start, 0);
  MatrixMultKernel<<<numBlocks, BLOCK_MAX_THREADS>>>(dmA, dmB, dmC, rowsA, columnsB, columnsA);
  cudaEventRecord(cuda_stop, 0);
  cudaMemcpy(matrixC, dmC, sizeof(float)*sizeC, cudaMemcpyDeviceToHost);

  /*uncomment to check result
  for(int i=0; i<sizeC; i++)
  {
    cout<<matrixC[i]<<" ";
  }
  cout<<"\n\n";
  */

  cudaFree(dmA);
  cudaFree(dmB);
  cudaFree(dmC);

  float* mmA;
  float* mmB;
  float* mmC;
  float* res = new float[sizeC];

  cudaMalloc((void**) &mmA, sizeof(float)*sizeA);
  cudaMemcpy(mmA, matrixA, sizeof(float)*sizeA, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &mmB, sizeof(float)*sizeB);
  cudaMemcpy(mmB, matrixB, sizeof(float)*sizeB, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &mmC, sizeof(float)*sizeC);
  cudaMemcpy(mmC, res, sizeof(float)*sizeC, cudaMemcpyHostToDevice);

  cudaEventRecord(cuda_start2, 0);
  gpu_blas_mmul(mmA, mmB, mmC, columnsB, columnsA, columnsB);
  cudaEventRecord(cuda_stop2, 0);

  cudaMemcpy(res, mmC ,sizeof(float)*sizeC,cudaMemcpyDeviceToHost);

  /* uncomment to check result
  for(int i=0; i<sizeC; i++)
  {
    cout<<res[i]<<" ";
  }
  cout<<"\n";
  */

  float mse = 0.0;
  for (int i = 0; i < sizeC; ++i) {
    mse = mse + pow(res[i] - matrixC[i], 2);
  }
  mse = mse / sizeC;

  cout << "MSE: " << mse << std::endl;

  cudaEventElapsedTime(&cuda_elapsed_time, cuda_start, cuda_stop);
  cudaEventElapsedTime(&cuda_elapsed_time2, cuda_start2, cuda_stop2);
  printf("Algorithm only cuda clock cycles for regular : %f\n", cuda_elapsed_time);
  printf("Algorithm only cuda clock cycles for cublas : %f\n", cuda_elapsed_time2);

  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(res);

  cudaFree(mmA);
  cudaFree(mmB);
  cudaFree(mmC);
  return 0;
}

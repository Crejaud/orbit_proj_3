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

  // Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);

    // Do the actual multiplication
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, lda, A, ldb, beta, C, ldc);

  // Destroy the handle
 cublasDestroy(handle);
}

int main()
{
  int rowsA = 3;
  int columnsA = 2;
  int sizeA = rowsA*columnsA;
  int rowsB  = 2;
  int columnsB = 4;
  int sizeB = rowsB*columnsB;
  int sizeC = rowsA*columnsC;

  float* matrixA = new float[sizeA];
  float* matrixB = new float[sizeB];
  float* matrixC = new float[sizeC];

  createArrayWithRandomValues(matrixA, sizeA);
  createArrayWithRandomValues(matrixB, sizeb);
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
  float* dmatrixA;
  float* dmatrixB;
  float* dmatrixC;

  cudaMalloc((void**) &dmatrixA, sizeof(float)*sizeA);
  cudaMemcpy(dmatrixA, matrixA, sizeof(float)*sizeA, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmatrixB, sizeof(float)*sizeB);
  cudaMemcpy(dmatrixB, matrixB, sizeof(float)*sizeB, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmatrixC, sizeof(float)*sizeC);
  cudaMemcpy(dmatrixC, matrixC, sizeof(float)*sizeC, cudaMemcpyHostToDevice);

  int spb = sizeC + (BLOCK_MAX_THREADS - 1);
  int numBlocks = spb / BLOCK_MAX_THREADS;
  MatrixMultKernel<<<numBlocks, BLOCK_MAX_THREADS>>(dmatrixA, dmatrixB, dmatrixC, rowsA, columnsB, columnsA);
  cudaMemcpy(matrixC, dmatrixC, sizeof(float)*sizeC, cudaMemcpyDeviceToHost);
  cout<<"Printing result: \n";
  for(int i=0; i<sizeC; i++)
  {
    cout<<matrixC[i]<<" ";
  }
  cout<<"\n\n";

  cudaFree(dmatrixA);
  cudaFree(dmatrixB);
  cudaFree(dmatrixC);


  //CUBLAS PART
  //pointers for cublas
  float* mmatrixA;
  float* mmatrixB;
  float* mmatrixC;

  float* resultMatrix = new float[sizeC];

  cudaMalloc((void**) &mmatrixA, sizeof(float)*sizeA);
  cudaMemcpy(mmatrixA, matrixA, sizeof(float)*sizeA, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &mmatrixB, sizeof(float)*sizeB);
  cudaMemcpy(mmatrixB, matrixB, sizeof(float)*sizeB, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &mmatrixC, sizeof(float)*sizeC);
  cudaMemcpy(mmatrixC, resultMatrix, sizeof(float)*sizeC, cudaMemcpyHostToDevice);

   gpu_blas_mmul(mmatrixA, mmatrixB, mmatrixC, columnsB, columnsA, columnsB);

   cudaMemcpy(resultMatrix, mmatrixC ,sizeof(float)*sizeC,cudaMemcpyDeviceToHost);

   cout<<"Printing cuBLAS result: \n";
   for(int i=0; i<sizeC; i++)
   {
     cout<<resultMatrix[i]<<" ";
   }
   cout<<"\n";

   float mse = 0.0;
   for (int i = 0; i < sizeC; ++i) {
     mse += pow(resultMatrix[i] - matrixC[i], 2);
   }
   mse /= sizeC;

   cout << "cuBLAS MSE: " << mse << std::endl;

  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(resultMatrix);

  cudaFree(mmatrixA);
  cudaFree(mmatrixB);
  cudaFree(mmatrixC);
  return 0;
}

#include <iostream>
#include <cublas_v2.h>
using std::cout;

int MAX_THREADS_PER_BLOCK = 512;

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
  int sizeA = rows1*columns1;
  int rowsB  = 2;
  int columnsB = 4;
  int sizeB = rows2*columns2;
  int sizeC = rows1*columns2;

  float* matrixA = new float[sizeA];
  float* matrixB = new float[sizeB];
  float* matrixC = new float[sizeC];

  createArrayWithRandomValues(matrixA, sizeA);
  createArrayWithRandomValues(matrixB, sizeb);
  cout<<"Matrix A: \n";
  for(int i=0; i<size1; i++)
  {
    cout<<matrixA[i]<<" ";
  }
  cout<<"\n";
  cout<<"Matrix B: \n";
  for(int i=0; i<size2; i++)
  {
    cout<<matrixB[i]<<" ";
  }
  cout<<"\n";
  float* dmatrixA;
  float* dmatrixB;
  float* dmatrixC;

  cudaMalloc((void**) &dmatrixA, sizeof(float)*size1);
  cudaMemcpy(dmatrixA, matrixA, sizeof(float)*size1, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmatrixB, sizeof(float)*size2);
  cudaMemcpy(dmatrixB, matrixB, sizeof(float)*size2, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &dmatrixC, sizeof(float)*size3);
  cudaMemcpy(dmatrixC, matrixC, sizeof(float)*size3, cudaMemcpyHostToDevice);

  int numBlocks = (size3 + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK;
  MatrixMultKernel<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(dmatrixA, dmatrixB, dmatrixC, rows1, columns2, columns1);
  cudaMemcpy(matrixC, dmatrixC, sizeof(float)*size3, cudaMemcpyDeviceToHost);
  cout<<"Printing result: \n";
  for(int i=0; i<size3; i++)
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

  float* resultMatrix = new float[size3];

  cudaMalloc((void**) &mmatrixA, sizeof(float)*size1);
  cudaMemcpy(mmatrixA, matrixA, sizeof(float)*size1, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &mmatrixB, sizeof(float)*size2);
  cudaMemcpy(mmatrixB, matrixB, sizeof(float)*size2, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &mmatrixC, sizeof(float)*size3);
  cudaMemcpy(mmatrixC, resultMatrix, sizeof(float)*size3, cudaMemcpyHostToDevice);

   gpu_blas_mmul(mmatrixA, mmatrixB, mmatrixC, columns2, columns1, columns2);

   cudaMemcpy(resultMatrix, mmatrixC ,sizeof(float)*size3,cudaMemcpyDeviceToHost);

   cout<<"Printing cuBLAS result: \n";
   for(int i=0; i<size3; i++)
   {
     cout<<resultMatrix[i]<<" ";
   }
   cout<<"\n";

   float mse = 0.0;
   for (int i = 0; i < size3; ++i) {
     mse += pow(resultMatrix[i] - matrixC[i], 2);
   }
   mse /= size3;

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

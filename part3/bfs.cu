#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <queue>

using namespace std;

int iterative_bfs(int **matrix, unsigned int N, int target, bool **visited_matrix);
int recursive_bfs(int **matrix, unsigned int N, int target, bool **visited_matrix);
int recursive_bfs_helper(int **matrix, unsigned int N, int target, int x, int y, bool **visited_matrix);

void kernel_bfs_wrapper(int **matrix, int *result, int *mtx, unsigned int N, int target, bool **visited_matrix);
__global__ void bfs_kernel(int **matrix, int *target, int *mtx, unsigned int N, int target, bool **visited_matrix);

int main()
{
  // default size is 100
  unsigned int N = 100;
  int **seq_matrix,
  **cuda_matrix,
  *seq_result,
  *cuda_result,
  *mtx,
  target;
  bool **visited_matrix, **cuda_visited_matrix;

  // Declare timers
  float cuda_elapsed_time;
  cudaEvent_t cuda_start, cuda_stop;
  clock_t seq_start, seq_stop, seq_iter_elapsed_time, seq_recur_elapsed_time;

  cout << "Enter N (NxN matrix): ";
  cin >> N;

  cout << "Enter target integer [0 to 100]: ";
  cin >> target;

  // allocate memory for seq
  seq_matrix = (int**) malloc(N * sizeof(int*));
  visited_matrix = (bool**) malloc(N * sizeof(bool*));

  // set matrix to random integers from 0 to 100
  for (int i = 0; i < N; i++) {
    *seq_matrix = (int*) malloc(N * sizeof(int));
    *visited_matrix = (bool*) malloc(N * sizeof(bool));
    for (int j = 0; j < N; j++) {
      seq_matrix[i][j] = rand() % 101;
      visited_matrix[i][j] = false;
    }
  }

  // allocate memory for cuda
  cudaMalloc((void**)&cuda_matrix, N * sizeof(int*));
  for (int i = 0; i < N; i++) {
    cudaMalloc((void**)&cuda_matrix[i], N * sizeof(int));
  }
  cudaMalloc((void**)&cuda_result, sizeof(int));

  // set values of cuda target to -1 (not found)
  // set values of mtx target to 0
  cudaMemset(cuda_result, -1, sizeof(int));
  cudaMemset(mtx, 0, sizeof(int));

  // set up timing variables
  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_stop);

  // copy from host to device
  cudaEventRecord(cuda_start, 0);
  for (int i = 0; i < N; i++) {
    cudaMemcpy(cuda_matrix[i], seq_matrix[i], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_visited_matrix[i], visited_matrix[i], N * sizeof(bool), cudaMemcpyHostToDevice);
  }

  // START CUDA
  kernel_bfs_wrapper(cuda_matrix, cuda_result, mtx, N, target, cuda_visited_matrix);

  // copy from device to host
  cudaMemcpy(seq_result, cuda_result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(cuda_stop, 0);
  cudaEventSynchronize(cuda_stop);
  cudaEventElapsedTime(&cuda_elapsed_time, cuda_start, cuda_stop);

  // destroy timers
  cudaEventDestroy(cuda_start);
  cudaEventDestroy(cuda_stop);

  cout << "----------------------------------------------------------" << endl;
  cout << "Found: " << *seq_result << endl;
  cout << "[CUDA] Elapsed time: " << cuda_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << endl;

  cout << "Starting sequential iterative approach." << endl;

  // reset visited_matrix back to false
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      visited_matrix[i][j] = false;
    }
  }

  seq_start = clock();

  // call iterative bfs
  *seq_result = -1;
  *seq_result = iterative_bfs(seq_matrix, N, target, visited_matrix);

  seq_stop = clock();
  seq_iter_elapsed_time = 1000*(seq_stop - seq_start)/CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "Found: " << *seq_result << endl;
  cout << "[SEQUENTIAL - Iterative] Elapsed time: " << seq_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << "Starting sequential recursive approach." << endl;

  // reset visited_matrix back to false
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      visited_matrix[i][j] = false;
    }
  }

  seq_start = clock();

  // call recursive bfs
  *seq_result = -1;
  *seq_result = recursive_bfs(seq_matrix, N, target, visited_matrix);

  seq_stop = clock();
  seq_recur_time = 1000*(seq_stop - seq_start)/CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "Found: " << *seq_result << endl;
  cout << "[SEQUENTIAL - Recursive] Elapsed time: " << seq_elapsed_time << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;

  // free and cuda free
  for (int i = 0; i < N; i++) {
    free(seq_matrix[i]);
    free(visited_matrix[i]);
  }
  free(seq_matrix);
  free(visited_matrix);
  free(seq_result);
  for (int i = 0; i < N; i++) {
    cudaFree(cuda_matrix[i]);
    cudaFree(cuda_visited_matrix[i]);
  }
  cudaFree(cuda_matrix);
  cudaFree(cuda_visited_matrix);
  cudaFree(cuda_result);

  return 0;
}

int iterative_bfs(int **matrix, unsigned int N, int target, bool **visited_matrix) {
  // check initial spot
  if (matrix[0][0] == target)
    return matrix[0][0];

  visited_matrix[0][0] = true;
  queue<int> qx = new queue<int>;
  queue<int> qy = new queue<int>;
  qx.push(0);
  qy.push(0);
  while (!qx.empty()) {
    int x = qx.pop();
    int y = qy.pop();
    visited_matrix[x][y] = true;
    if (matrix[x][y] == target) {
      return matrix[x][y];
    }
    // check right then check down
    if (x + 1 < N && !visited_matrix[x+1][y]) {
      qx.push(x+1);
      qy.push(y);
    }
    if (y + 1 < N && !visited_matrix[x][y+1]) {
      qx.push(x);
      qy.push(y+1);
    }
  }
  return -1;
}

int recursive_bfs(int **matrix, unsigned int N, int target, bool **visited_matrix) {
  // check initial spot
  if (matrix[0][0] == target)
    return matrix[0][0];

  visited_matrix[0][0] = true;

  return recursive_bfs_helper(matrix, N, target, 0, 0);
}

int recursive_bfs_helper(int **matrix, unsigned int N, int target, int x, int y, bool **visited_matrix) {
  // check right
  if (matrix[x+1][y] == target)
    return matrix[x+1][y];
  // check down
  if (matrix[x][y+1] == target)
    return matrix[x][y+1];

  // traverse to right first, then down
  int right = -1;
  int down = -1;

  if (x+1 < N && !visited_matrix[x+1][y]) {
    visited_matrix[x+1][y] = true;
    right = recursive_bfs_helper(matrix, N, target, x+1, y);
  }
  // if found in right, then return right
  if (right != -1)
    return right;

  if (y+1 < N && !visited_matrix[x][y+1]) {
    visited_matrix[x][y+1] = true;
    down = recursive_bfs_helper(matrix, N, target, x, y+1);
  }
  // if found in down, then return down
  if (down != -1)
    return down;

  // otherwise, nothing has been found.
  return -1;
}

void kernel_bfs_wrapper(int **matrix, int *result, int *mtx, unsigned int N, int target, bool **visited_matrix)
{
  // 2 dimensional
  dim3 gridSize = (256, 256);
  dim3 blockSize = (256, 256);
  find_target_kernel<<< gridSize, blockSize >>>(matrix, max, mtx, N);
}

__global__ void find_target_kernel(int **matrix, int *result, int *mtx, unsigned int N, int target, bool **visited_matrix)
{
  int idx = threadIdx.x + blockDim.x + blockIdx.x;
  int idy = threadIdx.y + blockDim.y + blockIdx.y;

  if (idx >= N || idy >= N) {
    return;
  }

  // found!!
  if (matrix[idx][idy] == target) {
    *result = matrix[idx][idy];
  }
}

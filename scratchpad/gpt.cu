#include <stdio.h>

  // Matrix size
  #define N 5000

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* A, float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // printf("block[%d][%d][%d] thread[%d][%d] - C[%d][%d]\n", blockIdx.x, blockIdx.y, blockIdx.z , threadIdx.x, threadIdx.y, row, col);
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

void print_matrix(float * p_arr, int size);

int main() {

    printf("Matrix size: %dx%d\n", N*N);

    // Declare host matrices
    float* h_A;
    float* h_B;
    float* h_C;

    // Declare device matrices
    float* d_A;
    float* d_B;
    float* d_C;

    int size = N * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    // printf("Matrix 1: \n");
    // print_matrix(h_A, N);

    // printf("\nMatrix 2: \n");
    // print_matrix(h_B, N);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // Number of threads in each block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);  // Number of blocks

    printf("Grid  - [%3d][%3d][%3d]\n", gridDim.x, gridDim.y, gridDim.z);
    printf("Block - [%3d][%3d][%3d]\n", blockDim.x, blockDim.y, blockDim.z);
    printf("Total threads: %d\n", gridDim.x* gridDim.y* gridDim.z * blockDim.x* blockDim.y* blockDim.z);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    // printf("\nResult matrix: \n");
    // print_matrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

void print_matrix(float * p_arr, int size)
{
    return;
    // for (int i = 0; i < size; i++) {
    //     for (int j = 0; j < size; j++) {
    //         printf("%f ", p_arr[i * N + j]);
    //     }
    //     printf("\n");
    // }

}


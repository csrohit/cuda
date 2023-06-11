#include <stdio.h>

#define N 10   // Array size

// CUDA kernel for array addition
__global__ void arrayAdd(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Executing kernel blockId[%d][%d][%d], threadId[%d][%d][%d], blockDim=%d,%d,%d -- idx=%d\n",
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z, idx);
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Declare host arrays
    float* h_A;
    float* h_B;
    float* h_C;

    // Declare device arrays
    float* d_A;
    float* d_B;
    float* d_C;

    int size = N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 5;  // Number of threads in each block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks
    printf("number of blocks: %d\n", numBlocks);

    // Launch kernel
    arrayAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        printf("%f \n", h_C[i]);
    }
    printf("\n");

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


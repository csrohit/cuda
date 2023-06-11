#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 10

__global__ void VectorAdd(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    printf("adding i=%d:  %d + %d = %d\n", i, a[i], b[i], c[i]);
    return;
}

int main() {
    int *a, *b, *c;

    int *cudaA, *cudaB, *cudaC;

    a = (int *)malloc(SIZE * sizeof(int));
    b = (int *)malloc(SIZE * sizeof(int));
    c = (int *)malloc(SIZE * sizeof(int));

    cudaMalloc((void **)&cudaA, sizeof(a));
    cudaMalloc((void **)&cudaB, sizeof(b));
    cudaMalloc((void **)&cudaC, sizeof(c));

    cudaMemcpy((void *)cudaA, (const void *)a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)cudaB, (const void *)b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < SIZE; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        printf("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
    }

    VectorAdd<<<1, SIZE>>>(cudaA, cudaB, cudaC);
    
    cudaMemcpy(c, cudaC, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i)
        printf("c[%d] = %d\n", i, c[i]);
    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    free(a);
    free(b);
    free(c);

    return 0;
}

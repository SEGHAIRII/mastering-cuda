#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N 8

void init_vectors(float *A, int size){
    for (int i =0;i<size;i++){
        A[i] =  (float)rand() / (float)RAND_MAX;
    }
}

void print_vect(float* a, int size){
    for (int i =0; i < size; i++){
        printf("%f", a[i]);
    }
    printf("\n");
}

__global__
void  addvectKernel(float* a, float*b, float*c, int size){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) c[index] = a[index] + b[index];
}


int main(){
    srand((unsigned int)time(NULL));

    // Defining the source vectors
    float* A_h, *B_h, *C_h;
    int size = N * sizeof(float);
    A_h = (float*) malloc(size);
    B_h = (float*) malloc(size);
    C_h = (float*) malloc(size);
    init_vectors(A_h, N);
    init_vectors(B_h, N);

    float* A_d, *B_d, *C_d;
    // Allocating
    cudaMalloc((void**)&C_d, size);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    //launching the kernel
    int blocksPerGrid = (N + 256 - 1) / 256;
    addvectKernel<<<blocksPerGrid, 256>>>(A_d, B_d, C_d, N);
    //copying the result to host memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    print_vect(A_h, N);
    print_vect(B_h, N);
    print_vect(C_h, N);

    free(A_h);
    free(B_h);
    free(C_h);


    // Freeing
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
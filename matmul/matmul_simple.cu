#include <stdio.h>
#include <stdlib.h>
#define WIDTH 4
#define HEIGHT 4


float* init_matrix(int rows, int cols) {
    float* mat = (float*)malloc(rows * cols * sizeof(float));
    if (!mat) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    // Initialize with example values (row * col)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = (float)(i * cols + j);
        }
    }

    return mat;
}

void print_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}


__global__
void matmul_simple(float* result, float *mat_a, float* mat_b, int height_a, int width, int width_b){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if((row < height_a) && (col < width_b)){
        float value = 0;
        for (int i = 0; i < width; i++){

            value += (mat_b[row*width + i] * mat_a[i*width + col]);
        }
        result[row*width + col] = value;

    }
}


int main(){
    // initializing matrices

    float * A_h = init_matrix(HEIGHT, WIDTH);
    float * B_h = init_matrix(HEIGHT, WIDTH);
    float *C_h = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
;


    // allocating in the gpu
    float *A_d, *B_d, *C_d;
    int size = HEIGHT * WIDTH * sizeof(float);
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // transfering data
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    //launching the kernel
    dim3 block(16, 16, 1);
    dim3 grid((HEIGHT + 16 - 1) / 16, (WIDTH + 16 - 1) / 16, 1);

    matmul_simple<<<grid, block>>>(C_d, A_d, B_d, HEIGHT, WIDTH, WIDTH);
    printf("============ matrix A ============\n");
    print_matrix(A_h, HEIGHT, WIDTH);
    printf("============ matrix B ============\n");

    print_matrix(B_h, HEIGHT, WIDTH);
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    printf("============ matrix C ============\n");

    print_matrix(C_h, HEIGHT, WIDTH);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
}
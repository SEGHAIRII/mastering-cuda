#include <stdio.h>
#include <stdlib.h>


#define WIDTH 64
#define HEIGHT 64
#define MASK_SIZE 3


unsigned char* generate_image_data(int width, int height) {
    // Allocate memory: width * height pixels * 3 channels (RGB)
    unsigned char* data = (unsigned char*)malloc(width * height);
    if (!data) {
        fprintf(stderr, "Failed to allocate image data\n");
        return NULL;
    }

    // Fill with a simple RGB gradient
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x);
            data[index + 0] = (unsigned char)(x * 255 / width);
        }
    }

    return data;
}

void print_image(unsigned char * img, int length, int width, int height){
    for (int i = 0; i < length; i++){
        printf("%f", img[length]);
    }
    printf("\n");
}

__global__
void blurKernel(unsigned char * in,unsigned char * out, int width, int height, int mask_size){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int value = 0;
    int denominator = (2 * mask_size + 1) * (2 * mask_size + 1);
    if(row < height && col < width){
        // in the condition we take the minimum of (mask max row idx, image height) to insure we are padding with 0 in a way 
        for (int mask_row_idx = row - mask_size; (mask_row_idx < (row + mask_size)) && (mask_row_idx < height); mask_row_idx++){
                for (int mask_col_idx = col - mask_size; (mask_col_idx < (col + mask_size)) && (mask_col_idx < width); mask_col_idx++){
                    value += in[mask_row_idx*width + mask_col_idx] / denominator;
        }
    }
    out[row * width + col] = value;
    }
}


int main(){
    //allocating images in the host
    unsigned char* img_h = generate_image_data(WIDTH, HEIGHT);
    unsigned char* blur_img_h = (unsigned char*)malloc(WIDTH * HEIGHT);

    //transfering to the gpu memory
    //allocating memory in the gpu memory
    unsigned char* img_d, *blur_img_d;
    int size = HEIGHT * WIDTH * sizeof(unsigned char);
    cudaMalloc((void **) &img_d, size);
    cudaMalloc((void ** ) &blur_img_d, size);
    //transfering the data
    cudaMemcpy(img_d, img_h, size, cudaMemcpyHostToDevice);
    //launching the kernel
    dim3 grid(2,2,1);
    dim3 block(32,32,1);

    blurKernel<<<grid, block>>>(img_d, blur_img_d, WIDTH, HEIGHT, MASK_SIZE);

    cudaMemcpy(blur_img_h, blur_img_d, size, cudaMemcpyDeviceToHost);


    //freeing the memory
    cudaFree(img_d);
    cudaFree(blur_img_d);
    print_image(img_h, 10, WIDTH, HEIGHT);
    print_image(blur_img_h, 10, WIDTH, HEIGHT);

    free(img_h);
    free(blur_img_h);

    return 0;
}
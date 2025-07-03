#include <stdio.h>
#include <stdlib.h>

#define width_i 64
#define height_i 64

unsigned char* generate_image_data(int width, int height) {
    // Allocate memory: width * height pixels * 3 channels (RGB)
    unsigned char* data = (unsigned char*)malloc(3 * width * height);
    if (!data) {
        fprintf(stderr, "Failed to allocate image data\n");
        return NULL;
    }

    // Fill with a simple RGB gradient
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = 3 * (y * width + x);
            data[index + 0] = (unsigned char)(x * 255 / width);   // Red
            data[index + 1] = (unsigned char)(y * 255 / height);  // Green
            data[index + 2] = 128;                                // Blue (constant)
        }
    }

    return data;
}


__global__
void grayscaleKernel(unsigned char * in,
                    unsigned char * out,
                int width, 
                int height)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockDim.y * blockDim.y + threadIdx.y;

    if (x_idx < width && y_idx < height){
        int offset = y_idx * width + x_idx;
        int offset_in = 3 * offset;
        out[offset] = 0.21*in[offset_in] + 0.71*in[offset_in + 1] + 0.07*in[offset_in + 2];

    }

}
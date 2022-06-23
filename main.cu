#include <iostream>
#include <cuda.h>
#include <unistd.h>
#include "palette.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define HIGH_RES_X 10000
#define HIGH_RES_Y 10000
#define HIGH_RES_CHUNK_X 5000
#define HIGH_RES_CHUNK_Y 5000

#define LOW_RES_X 10000
#define LOW_RES_Y 10000
#define LOW_RES_CHUNK_X 5000
#define LOW_RES_CHUNK_Y 5000

#define palette palette_low

__global__ void mandelbrot(uint8_t* r, uint8_t* g, uint8_t* b,
                           double zoom, double2 target, dim3 offset, dim3 size, int max_iter) {
    uint64_t px = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    uint64_t py = blockDim.y * blockIdx.y + threadIdx.y + offset.y;

    uint64_t idx = (py * size.y) + px;

    if (px >= size.x || py >= size.y)
        return;

    double x0 = ((double)px / size.x) * (4 / zoom) - 2 / zoom + target.x;
    double y0 = ((double)py / size.y) * (4 / zoom) - 2 / zoom + target.y;

    double x = x0;
    double y = y0;

    double zx;

    uint64_t iteration = 0;

    while (x * x + y * y < 4 && iteration < max_iter) {
        zx = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = zx;
        iteration++;
    }

    r[idx] = palette[iteration][0];
    g[idx] = palette[iteration][1];
    b[idx] = palette[iteration][2];
}

int main() {
    uint8_t *r, *g, *b;
    cudaMallocManaged(&r, LOW_RES_X * LOW_RES_Y * sizeof(uint8_t));
    cudaMallocManaged(&g, LOW_RES_X * LOW_RES_Y * sizeof(uint8_t));
    cudaMallocManaged(&b, LOW_RES_X * LOW_RES_Y * sizeof(uint8_t));

    dim3 chunk_size { LOW_RES_CHUNK_X, LOW_RES_CHUNK_Y };

    dim3 threadsPerBlock = dim3(32, 32);

    dim3 blocksPerGrid = dim3(
        (chunk_size.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (chunk_size.y + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    for (unsigned int off_x = 0; off_x < LOW_RES_X; off_x += chunk_size.x) {
        for (unsigned int off_y = 0; off_y < LOW_RES_Y; off_y += chunk_size.y) {
            printf("Rendering range (%i-%i, %i-%i)\n", off_x, off_x + chunk_size.x, off_y, off_y + chunk_size.y);

            mandelbrot<<<blocksPerGrid, threadsPerBlock>>>
                        (r, g, b,
                         1000, {-0.745, 0.095}, {off_x, off_y},
                         {LOW_RES_X, LOW_RES_Y}, MAX_ITER_LOW);

            cudaDeviceSynchronize();
        }
    }

    uint8_t *all;
    all = (uint8_t*)malloc(LOW_RES_X * LOW_RES_Y * 3 * sizeof(uint8_t));

    for (uint64_t i = 0; i < LOW_RES_X * LOW_RES_Y * 3; i++) {
        switch (i % 3) {
            case 0:
                all[i] = r[i / 3];
                break;
            case 1:
                all[i] = g[i / 3];
                break;
            case 2:
                all[i] = b[i / 3];
                break;
        }
    }

    stbi_write_png("out.png", LOW_RES_X, LOW_RES_Y, 3, all, LOW_RES_X * 3 * sizeof(uint8_t));

    return 0;
}

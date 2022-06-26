#include <iostream>
#ifndef CUDA_BUILD
#include <hip/hip_runtime.h>
#endif
#include "palette.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

uint64_t RES_X = 25000;
uint64_t RES_Y = 25000;
uint32_t RES_CHUNK_X = 5000;
uint32_t RES_CHUNK_Y = 5000;

#define palette palette_high

template<typename T>
struct dim2 {
    T x;
    T y;
};

template<typename T>
__global__ void mandelbrot(uint8_t* out, double zoom, double2 target, dim2<T> offset, dim2<T> size, int max_iter) {
    uint64_t px = blockDim.x * blockIdx.x + threadIdx.x + offset.x;
    uint64_t py = blockDim.y * blockIdx.y + threadIdx.y + offset.y;

    uint64_t idx = 3 * ((py * size.y) + px);

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

    out[idx] = palette[iteration][0];
    out[idx + 1] = palette[iteration][1];
    out[idx + 2] = palette[iteration][2];
}

int main() {
    uint8_t *out;

#ifdef CUDA_BUILD
    cudaMallocManaged(&out, RES_X * RES_Y * 3 * sizeof(uint8_t));
#else
    hipMallocManaged(&out, RES_X * RES_Y * 3 * sizeof(uint8_t));
#endif

    dim3 chunk_size { RES_CHUNK_X, RES_CHUNK_Y };

    dim3 threadsPerBlock = dim3(32, 32);

    dim3 blocksPerGrid = dim3(
        (chunk_size.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (chunk_size.y + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    for (uint64_t off_x = 0; off_x < RES_X; off_x += chunk_size.x) {
        for (uint64_t off_y = 0; off_y < RES_Y; off_y += chunk_size.y) {
	
            printf("Rendering range (%lu-%lu, %lu-%lu)\n", off_x, off_x + chunk_size.x, off_y, off_y + chunk_size.y);
#ifdef CUDA_BUILD
            mandelbrot<uint64_t><<<blocksPerGrid, threadsPerBlock>>>(
#else
            hipLaunchKernelGGL(mandelbrot<uint64_t>, blocksPerGrid, threadsPerBlock, 0, 0,
#endif
                               out,
                               1000, {-0.745, 0.095}, {off_x, off_y},
                               {RES_X, RES_Y}, MAX_ITER_HIGH
                               );



        }
#ifdef CUDA_BUILD
        cudaDeviceSynchronize();
#else
        hipDeviceSynchronize();
#endif
    }

    stbi_write_png("out.png", RES_X, RES_Y, 3, out, RES_X * 3 * sizeof(uint8_t));

    return 0;
}

#include "engine.h"


interpolator tickLogic(int tickCount) {
    interpolator result;
    result.tickCount = tickCount;
    

    return result;
}



__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int modVal = interp->tickCount * 5;
        uint32_t red   = (x + modVal) % 255;
        uint32_t green = (y + modVal) % 255;
        uint32_t blue  = (x + y + modVal) % 255;
        buffer[idx] = 0xFF000000 | (red << 16) | (green << 8) | blue;
    }
}


__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp) {
    // Define thread block and grid dimensions for the child kernel.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the child kernel to compute pixels.
    computePixel<<<numBlocks, threadsPerBlock>>>(buffer, width, height, interp);
    
    // Wait for the child kernel to finish before completing.
    __threadfence();
}
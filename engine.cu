#include "engine.h"


interpolator tickLogic(int tickCount) {
    interpolator result;
    result.tickCount = tickCount;
    

    return result;
}



__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int modVal = interp->tickCount * 10;
        uint32_t red   = 128 + 127*sinf((float)x/128 + ((float)modVal*inpf)/60);
        uint32_t green = 128 + 127*cosf((float)y/128 + ((float)modVal*inpf)/60);
        uint32_t blue  = 128 + 12*sinf((float)x/128 + (float)y/128 + ((float)modVal*inpf)/6);
        buffer[idx] = 0xFF000000 | (red << 16) | (green << 8) | blue;
    }
}


__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor) {
    // Define thread block and grid dimensions for the child kernel.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the child kernel to compute pixels.
    computePixel<<<numBlocks, threadsPerBlock>>>(buffer, width, height, interp,interpolationFactor);
    
    // Wait for the child kernel to finish before completing.
    __threadfence();
}

__device__ void interpolatorUpdateHandler() {
    
}
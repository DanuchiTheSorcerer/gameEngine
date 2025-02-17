#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// A simple structure holding game state for interpolation.
// Extend this structure with more state as needed.
typedef struct interpolator {
    int tickCount;
    // Additional game state fields go here.
} interpolator;

// tickLogic
// Processes game logic for the given tick count and returns an interpolator 
// containing the updated game state.
interpolator tickLogic(int tickCount);

// -----------------------------------------------------------------------------
// Device Function:
// Computes the color of a single pixel based on its (x, y) coordinate and the
// current game state stored in 'interp'.
// This is a __device__ function and can be used by various kernels.
// -----------------------------------------------------------------------------
__global__ void computePixel(int x, int y, const interpolator* interp);

// -----------------------------------------------------------------------------
// Kernel Function:
// Computes the entire frame by iterating over all pixels. This version is
// designed to be launched with a single thread (or a few threads) that loops
// over the full frame rather than launching one thread per pixel.
// -----------------------------------------------------------------------------
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_H

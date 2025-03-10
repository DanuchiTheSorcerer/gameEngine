#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// A simple structure holding game state for interpolation.
// Extend this structure with more state as needed.

struct interpolator {int tickCount;};
// tickLogic
// Processes game logic for the given tick count and returns an interpolator 
// containing the updated game state.
interpolator tickLogic(int tickCount);

//compute a frame
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor);

//put gpu response to new interpolator
__device__ void interpolatorUpdateHandler();

#ifdef __cplusplus
}
#endif

#endif // ENGINE_H

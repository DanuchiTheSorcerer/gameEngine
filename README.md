Game engine using Win32 and CUDA

For using the engine:

All your code should be in engine.cu, and engine.h
engine.cu must include:

1. #include "engine.h"
2. typedef struct interpolator (engine.h)
3. interpolator tickLogic(int tickCount)
4. __device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp)

You can customise the interpolator to contain any data you'd like, and the computeFrame() function to use interpolators in any way you'd like
Interpolations by interpolator coming soon
This only runs on windows computers with nvidia gpu architectures above a certain point
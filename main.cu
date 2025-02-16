#include <cuda_runtime.h>
#include <windows.h>
#include <chrono>
#include <climits>
#include "winlib.h"

int width = 1000;
int height = 1000;
float targetTPS = 60;
int refreshRate;
BITMAPINFO bmi;
int activeDisplayBufferIndex = 0;

struct frameBuffer {
    int state = 0; // 0 is untouched  1 is being worked on   2 is being copied (or has been copied if on DRAM)
    uint32_t* pixels; // pointer to pixels. Will be assigned at GPU allocation
};

struct interpolator {
    //fully customisable by the person using the engine. Contains all the information needed to compute a frame, minus the interpolation index
    int tickCount; // example data
};

struct gpuMeta {
    frameBuffer buffers[3]; //array of buffers
    int bufferRecencyOrder[3];   // rank of buffer indicies
    interpolator interpolators[2]; // array of interpolators
    int activeInterpolator = 0; // index of the current interpolator being used

    //flags, set only by the CPU and interpreted by GPU when it has the chance
    bool shouldSwitchInterpolator = false;  // for switching interpolator, set after new one has been copied
    bool shouldEndKernel = false; // for ending the persistent kernel
};

gpuMeta* gpuMetaData; // data on the GPU
frameBuffer displayBuffers[3]; //pined memory buffers on DRAM for displaying

void initAll(HINSTANCE hInstance) {
    //intiate winlib
    WinLib_Init(hInstance);

    //get refresh rate
    DEVMODE dm = {0};
    dm.dmSize = sizeof(DEVMODE);
    EnumDisplaySettings(nullptr, ENUM_CURRENT_SETTINGS, &dm);
    refreshRate = dm.dmDisplayFrequency;

    //configure bit map
    ZeroMemory(&bmi, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    //cuda memory allocations
    cudaMalloc(&gpuMetaData,sizeof(gpuMeta));   // allocate GPU data
    gpuMeta initialData = {};  

    for (int i = 0; i<3;i++) {
        cudaMalloc(&initialData.buffers[i].pixels, width * height * sizeof(uint32_t));  //allocate buffers 
        cudaHostAlloc(&displayBuffers[i].pixels, width * height * sizeof(uint32_t), cudaHostAllocDefault); //allocate display buffers to pinned memory for quick copies
    }

    cudaMemcpy(gpuMetaData,&initialData,sizeof(gpuMeta), cudaMemcpyHostToDevice);
}

void PaintWindow(HDC hdc) {
    //draw (this is called by winlib when handling an invalidated rect)
    StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height,displayBuffers[activeDisplayBufferIndex].pixels, &bmi,DIB_RGB_COLORS, SRCCOPY);
}

__global__ void computeFrame(uint32_t** buffer, interpolator* interpolator,int width,int height) { //pointer to buffer
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    (*buffer)[idx] = 0xFF000000 | 
        ((x + interpolator->tickCount) % 255) << 16 |
        ((y + interpolator->tickCount) % 255) << 8 |
        ((x + y + interpolator->tickCount) % 255);
}

__global__ void frameComputeLoop(gpuMeta* gpuMetaData, int width, int height,cudaStream_t stream) {
    while(!gpuMetaData->shouldEndKernel) {
        //check for new interpolators

        if (gpuMetaData->shouldSwitchInterpolator) {
            gpuMetaData->activeInterpolator = 1 - gpuMetaData->activeInterpolator;
            gpuMetaData->shouldSwitchInterpolator = false;
        }

        // compute a buffer

        dim3 threadsPerBlock(16, 16);     
        dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

        int updatedBufferIndex;

        if (gpuMetaData->buffers[gpuMetaData->bufferRecencyOrder[2]].state != 2) {  // compute on the most outdated buffer if it is not being copied

            updatedBufferIndex = gpuMetaData->bufferRecencyOrder[2];
            gpuMetaData->buffers[updatedBufferIndex].state = 1;
            computeFrame<<<numBlocks,threadsPerBlock>>>(&gpuMetaData->buffers[updatedBufferIndex].pixels,&gpuMetaData->interpolators[gpuMetaData->activeInterpolator],width,height);

            gpuMetaData->buffers[updatedBufferIndex].state = 0;

        } else { // if most outdated buffer is being copied, then use the second most outdated

            updatedBufferIndex = gpuMetaData->bufferRecencyOrder[1];
            gpuMetaData->buffers[updatedBufferIndex].state = 1;
            computeFrame<<<numBlocks,threadsPerBlock>>>(&gpuMetaData->buffers[updatedBufferIndex].pixels,&gpuMetaData->interpolators[gpuMetaData->activeInterpolator],width,height);
            gpuMetaData->buffers[updatedBufferIndex].state = 0;

        }

        // update recency ordering

        int newOrdering[3];
        newOrdering[0] = updatedBufferIndex;
        newOrdering[1] = gpuMetaData->bufferRecencyOrder[0];
        newOrdering[2] = gpuMetaData->bufferRecencyOrder[updatedBufferIndex % 2 + 1];

        gpuMetaData->bufferRecencyOrder[0] = newOrdering[0];
        gpuMetaData->bufferRecencyOrder[1] = newOrdering[1];
        gpuMetaData->bufferRecencyOrder[2] = newOrdering[2];


    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    initAll(hInstance); //initialize
    WinWindow* window = WinLib_CreateWindow("CUDA Powered Engine", width, height, hInstance); //open the window

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);



    auto lastTickTime = std::chrono::high_resolution_clock::now();
    auto lastDisplayTime = std::chrono::high_resolution_clock::now();
    int tickCount = 0;
    MSG msg;
    while (GetMessage(&msg,NULL,0,0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> timeSinceTick = now-lastTickTime;
        std::chrono::duration<float> timeSinceDisplay = now-lastDisplayTime;
        
        //if it is time for a new tick, than run it
        if (timeSinceTick.count() > 1.0f/targetTPS) {
            //game logic

            tickCount++;
            //update interpolator
            interpolator newInterpolator;
            newInterpolator.tickCount = tickCount;
            if (!gpuMetaData->shouldSwitchInterpolator) { // if the gpu hasn't gotten around to the last switch yet, dont update (overload gpu = bad)
                //cuda copy command

                //cuda set shouldSwitchInterpolator command

                if (tickCount == 1) { // start up the frame calculations after the first interpolator is made
                    frameComputeLoop<<<1,1,0,stream>>>(gpuMetaData,width,height,stream);

                }
            }
            lastTickTime = now;
        }

        //if it is time for a new image to be displayed, do so
        if (timeSinceDisplay.count() > 1.0f/refreshRate) {
            //cuda copy frame command
            //copy from gpuMetaData->buffers[gpuMetaData->bufferRecencyOrder[0]]  to     displayBuffers[(activeDisplayBufferIndex + 1) % 3]

            //update activeDisplayBufferIndex
            activeDisplayBufferIndex = (activeDisplayBufferIndex + 1) % 3;
            //request a repaint by windows
            InvalidateRect(window->hwnd,NULL,FALSE);
            lastDisplayTime = now;
        }
    }

    
    WinLib_DestroyWindow(window);
    bool endKernel = true;
    cudaMemcpyAsync(&gpuMetaData->shouldEndKernel, &endKernel, sizeof(bool), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFree(gpuMetaData);
    return 0;
}
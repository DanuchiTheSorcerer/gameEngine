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
int currentActiveInterpolator = 0; //cpu side
cudaEvent_t interpolatorCopyEvent;
cudaEvent_t fpsCopyEvent;
int frameCounter;

struct interpolator {
    //fully customisable by the person using the engine. Contains all the information needed to compute a frame, minus the interpolation index
    int tickCount; // example data
};

struct gpuMeta {
    uint32_t* frame; //array of buffers
    interpolator interpolators[2]; // array of interpolators
    int activeInterpolator = 0; // index of the current interpolator being used
    uint32_t* pointerToDisplay;
    int framesCalculated;

    //flags, set only by the CPU and interpreted by GPU when it has the chance
    bool shouldSwitchInterpolator = false;  // for switching interpolator, set after new one has been copied
    bool shouldEndKernel = false; // for ending the persistent kernel
    bool shouldCopyFrame = false;  // for copying to mapped memory
};

gpuMeta* gpuMetaData; // data on the GPU
uint32_t* displayFrame; //pined mapped memory buffer on DRAM for displaying

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

    cudaMalloc(&initialData.frame, width * height * sizeof(uint32_t));  //allocate frame
    cudaHostAlloc(&displayFrame, width * height * sizeof(uint32_t), cudaHostAllocMapped); //allocate display frame to pinned mapped memory for quick copies by the GPU

    uint32_t* deviceDisplayPtr = nullptr;
    cudaHostGetDevicePointer(&deviceDisplayPtr, displayFrame, 0);
    initialData.pointerToDisplay = deviceDisplayPtr; // copy pinter so that the GPU can copy frames to CPU mapped memory

    cudaMemcpy(gpuMetaData,&initialData,sizeof(gpuMeta), cudaMemcpyHostToDevice);

    cudaEventCreate(&interpolatorCopyEvent);
    cudaEventCreate(&fpsCopyEvent);
}

void PaintWindow(HDC hdc) {
    //draw (this is called by winlib when handling an invalidated rect)
    StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height,displayFrame, &bmi,DIB_RGB_COLORS, SRCCOPY);
}

__global__ void computeFrame(uint32_t* buffer, interpolator* interpolator,int width,int height) { //pointer to buffer
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    buffer[idx] = 0xFF000000 | 
        ((x + interpolator->tickCount*5) % 255) << 16 |
        ((y + interpolator->tickCount*5) % 255) << 8 |
        ((x + y + interpolator->tickCount*5) % 255);
}

__global__ void copyFrameKernel(uint32_t* dst, const uint32_t* src, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixels) {
        dst[idx] = src[idx];
    }
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



        // Launch the frame computation
        computeFrame<<<numBlocks, threadsPerBlock>>>(gpuMetaData->frame,&gpuMetaData->interpolators[gpuMetaData->activeInterpolator],width, height);

        gpuMetaData->framesCalculated = gpuMetaData->framesCalculated + 1;
        //copy frame if needed
        if (gpuMetaData->shouldCopyFrame) {
            int totalPixels = width * height;
            int threads = 256;
            int blocks = (totalPixels + threads - 1) / threads;
            copyFrameKernel<<<blocks, threads>>>(gpuMetaData->pointerToDisplay, gpuMetaData->frame, totalPixels);
        
            // Optionally, use __threadfence() to ensure memory visibility if needed.
            __threadfence();
        
            // Once copy is complete, reset the flag.
            gpuMetaData->shouldCopyFrame = false;
        }


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
    auto lastFpsLogTime = std::chrono::high_resolution_clock::now();

    MSG msg;
    //main loop
    while (true) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        
            // Handle WM_QUIT properly
            if (msg.message == WM_QUIT) {
                //memory cleanup
                WinLib_DestroyWindow(window);
                bool endKernel = true;
                cudaMemcpyAsync(&gpuMetaData->shouldEndKernel, &endKernel, sizeof(bool), cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
                cudaFree(gpuMetaData);
                cudaEventDestroy(interpolatorCopyEvent);
                cudaEventDestroy(fpsCopyEvent);
                cudaFreeHost(displayFrame);
                return 0;
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> timeSinceTick = now-lastTickTime;
        std::chrono::duration<float> timeSinceDisplay = now-lastDisplayTime;
        std::chrono::duration<float> timeSinceFpsLog = now - lastFpsLogTime;

        if (timeSinceFpsLog.count() >= 1.0f) {
            int framesBefore = frameCounter;
            cudaMemcpyAsync( &frameCounter,&gpuMetaData->framesCalculated,sizeof(int),cudaMemcpyDeviceToHost);

            cudaEventRecord(interpolatorCopyEvent);
            cudaEventSynchronize(interpolatorCopyEvent); 
            char buffer[64];
            sprintf(buffer, "GPU FPS: %d\n", frameCounter-framesBefore);
            OutputDebugString(buffer);

            // Reset counter and update time;
            lastFpsLogTime = now;
        }
        
        //if it is time for a new tick, than run it
        if (timeSinceTick.count() > 1.0f/targetTPS) {
            //game logic

            tickCount++;
            //update interpolator
            interpolator newInterpolator;
            newInterpolator.tickCount = tickCount;

            // Determine the inactive interpolator slot.
            int inactiveIndex = 1 - currentActiveInterpolator;
            
            // Asynchronously copy the new interpolator data to the inactive slot on the GPU.
            cudaMemcpyAsync(&gpuMetaData->interpolators[inactiveIndex], &newInterpolator,sizeof(interpolator),cudaMemcpyHostToDevice);

            cudaEventRecord(interpolatorCopyEvent);
            cudaEventSynchronize(interpolatorCopyEvent);                

            
            // Set the flag so the GPU will switch to the new interpolator on its next frame.
            bool switchFlag = true;
            cudaMemcpyAsync(&gpuMetaData->shouldSwitchInterpolator, &switchFlag , sizeof(bool) , cudaMemcpyHostToDevice);
            
            cudaEventRecord(interpolatorCopyEvent);
            cudaEventSynchronize(interpolatorCopyEvent);    
            

            // Update our CPU-side record of the active slot.
            // The GPU will switch to 'inactiveIndex' upon processing the flag.
            currentActiveInterpolator = inactiveIndex;

            if (tickCount == 1) { // start up the frame calculations after the first interpolator is made
                frameComputeLoop<<<1,1,0,stream>>>(gpuMetaData,width,height,stream);
            }
            lastTickTime = now;
        }

        //if it is time for a new image to be displayed, do so
        if (timeSinceDisplay.count() > 1.0f / refreshRate) {

            //flag GPU to copy
            bool copyFlag = true;
            cudaMemcpyAsync(&gpuMetaData->shouldCopyFrame,&copyFlag,sizeof(bool),cudaMemcpyHostToDevice);

            InvalidateRect(window->hwnd,NULL,FALSE);


            lastDisplayTime = now;
        }
    }
    return 0;
}
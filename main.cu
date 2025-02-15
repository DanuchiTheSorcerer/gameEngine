#include <iostream>
#include <cuda_runtime.h>
#include <windows.h>
#include "winlib.h"
#include <chrono>
#include <string>



int WIDTH = 1000;
int HEIGHT = 1000;
const int BUFFER_COUNT = 2;

uint32_t* gpu_buffers[BUFFER_COUNT];  // Framebuffers on GPU
uint32_t* cpu_buffers[BUFFER_COUNT];  // CPU-accessible buffers
cudaEvent_t frameReady[BUFFER_COUNT]; // CUDA events for sync
int currentBuffer = 0;  // Tracks the active buffer

BITMAPINFO bmi;
HDC hdcMem;

// CUDA kernel to fill the framebuffer
__global__ void renderKernel(uint32_t* framebuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        framebuffer[index] = 0xFF000000 | ((x) % 255) << 16 | ((-y) % 255) << 8 | (x+y)%255;
    }
}

// Function to initialize GPU memory
void InitCUDA() {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        cudaMalloc(&gpu_buffers[i], WIDTH * HEIGHT * sizeof(uint32_t));
        cudaMallocHost(&cpu_buffers[i], WIDTH * HEIGHT * sizeof(uint32_t));  // Pinned memory
        cudaEventCreate(&frameReady[i]);
    }
}

// Function to render a frame
void RenderFrame() {
    int nextBuffer = (currentBuffer + 1) % BUFFER_COUNT;

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    renderKernel<<<numBlocks, threadsPerBlock>>>(gpu_buffers[nextBuffer], WIDTH, HEIGHT);
    cudaEventRecord(frameReady[nextBuffer]);  // Signal completion

    // Wait for previous frame to be ready before copying
    cudaEventSynchronize(frameReady[currentBuffer]);
    cudaMemcpyAsync(cpu_buffers[currentBuffer], gpu_buffers[currentBuffer], WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);

}

// Function to display the framebuffer in WM_PAINT
void PaintWindow(HDC hdc) {
    StretchDIBits(hdc, 0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, cpu_buffers[currentBuffer], &bmi, DIB_RGB_COLORS, SRCCOPY);
    currentBuffer = (currentBuffer + 1) % BUFFER_COUNT;  // Swap buffers
}

// Win32 message loop with rendering
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    DEVMODE dm;
    memset(&dm, 0, sizeof(dm));
    dm.dmSize = sizeof(dm);
    EnumDisplaySettings(nullptr, ENUM_CURRENT_SETTINGS, &dm);

    // Initialize window using winlib
    WinLib_Init(hInstance);
    WinWindow* myWindow = WinLib_CreateWindow("My Win32 Window", WIDTH, HEIGHT, hInstance);

    // Initialize framebuffer
    InitCUDA();

    // Setup DIB
    ZeroMemory(&bmi, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = WIDTH;
    bmi.bmiHeader.biHeight = -HEIGHT;  // Negative to flip vertically
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(myWindow->hwnd);
    hdcMem = CreateCompatibleDC(hdc);
    ReleaseDC(myWindow->hwnd, hdc);

    
    std::string refreshRateStr = "Refresh rate: " + std::to_string(dm.dmDisplayFrequency) + " Hz \n";
    OutputDebugString(refreshRateStr.c_str());

    int frameCount = 0;
    int frameCheck = 0;
    auto lastFrameTime = std::chrono::steady_clock::now();
    auto tickStartTime = std::chrono::high_resolution_clock::now();
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        // Process Windows messages
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        // fps calculation
        auto currentFrameTime = std::chrono::steady_clock::now();
        std::chrono::duration<float> deltaTime = currentFrameTime - lastFrameTime;
        if (deltaTime.count() > 1) {
            std::string debugString = std::to_string(frameCheck) + "fps\n";
            OutputDebugString(debugString.c_str());
            frameCheck = 0;
            lastFrameTime = currentFrameTime;
        }
        RenderFrame();
        frameCount++;
        frameCheck++;
        
        std::chrono::duration<float> timeInTick = std::chrono::high_resolution_clock::now() - tickStartTime;
        if (timeInTick.count() > 1.0f / 60.0f) {
            tickStartTime = std::chrono::high_resolution_clock::now();
        }

        // Request a repaint AFTER processing events
        InvalidateRect(myWindow->hwnd, NULL, FALSE);


    }


    // Cleanup
    for (int i = 0; i < BUFFER_COUNT; i++) {
        cudaFree(gpu_buffers[i]);
        cudaFreeHost(cpu_buffers[i]);
        cudaEventDestroy(frameReady[i]);
    }

    // Destroy window using winlib
    WinLib_DestroyWindow(myWindow);
    return 0;
}

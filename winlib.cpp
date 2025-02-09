#include "winlib.h"
#include <stdlib.h>

static const char* WINLIB_CLASS_NAME = "WinLibWindowClass";


/**
 * The window procedure for handling messages sent to windows created by this library.
 */
static LRESULT CALLBACK WinLib_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            PaintWindow(hdc);
            EndPaint(hwnd, &ps);
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

/////////////////////////////////////////////////////////////////////////This thing registers the window class (which has all the info about the window) with the OS
BOOL WinLib_Init(HINSTANCE hInstance) {
    // Fill in the WNDCLASSEX structure.
    WNDCLASSEX wc = {0};
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;   // Redraw on horizontal/vertical size changes.
    wc.lpfnWndProc   = WinLib_WindowProc;         // Our window procedure.
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = NULL;  // Set a default background color.
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = WINLIB_CLASS_NAME;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    // Register the window class.
    return RegisterClassEx(&wc);
}
///////////////////////////////////////////////////////////////////////This thing creates the window with the registered class and other stuff it plugs in
WinWindow* WinLib_CreateWindow(const char* title, int width, int height, HINSTANCE hInstance) {
    // Allocate memory for our window structure.
    WinWindow* window = (WinWindow*)malloc(sizeof(WinWindow));
    if (!window)
        return NULL;

    // Create the window using the previously registered class.
    window->hwnd = CreateWindowEx(
        0,                      // Optional window styles.
        WINLIB_CLASS_NAME,      // Window class name.
        title,                  // Window title.
        WS_OVERLAPPEDWINDOW,    // Window style (includes title bar, border, etc.).
        CW_USEDEFAULT, CW_USEDEFAULT,  // Initial position (x, y).
        width, height,          // Window dimensions.
        NULL,                   // Parent window (none in this case).
        NULL,                   // Menu handle.
        hInstance,              // Application instance handle.
        NULL                    // Additional application data.
    );

    // If window creation failed, free allocated memory and return NULL.
    if (!window->hwnd) {
        free(window);
        return NULL;
    }

    // Make the window visible and update it.
    ShowWindow(window->hwnd, SW_SHOW);
    UpdateWindow(window->hwnd);
    return window;
}

//////////////////////////////////////////////////////////////////////////EXTERMINATE a window
void WinLib_DestroyWindow(WinWindow* window) {
    if (window) {
        if (window->hwnd) {
            // Destroy the actual window.
            DestroyWindow(window->hwnd);
        }
        // Free the allocated memory.
        free(window);
    }
}
///////////////////////////////////////////////////////////////////////////////////This is for processing all of the pending events
void WinLib_PollEvents(void) {
    MSG msg;
    // Use PeekMessage in a loop so that this function returns immediately
    // if there are no messages, allowing it to be called in a non-blocking manner.
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

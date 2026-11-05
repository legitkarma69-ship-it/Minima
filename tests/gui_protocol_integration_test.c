#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include "../MinimaGUI/gui_protocol.h"
#include "../MinimaGUI/gui_chat_state.h"

#define WM_TEST_CHAT_READY (WM_APP + 3)
static HWND g_send_button;

static LRESULT CALLBACK test_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    (void)wparam; (void)lparam;
    if (message == WM_TEST_CHAT_READY) {
        minima_gui_set_chat_send_ready(g_send_button, TRUE, TRUE);
        return 0;
    }
    return DefWindowProcW(hwnd, message, wparam, lparam);
}

static void post_test_chat_ready(void* context) {
    PostMessageW((HWND)context, WM_TEST_CHAT_READY, 0, 0);
}

static int pump_test_messages(void) {
    MSG message; int count = 0;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message); DispatchMessageW(&message); ++count;
    }
    return count;
}

static int check_ready_path(HWND parent, const char* line, size_t length, const char* label) {
    EnableWindow(g_send_button, FALSE);
    const int marker = minima_protocol_dispatch(line, length, post_test_chat_ready, parent);
    if (marker != MINIMA_PROTOCOL_CHAT_READY || pump_test_messages() != 1 || !IsWindowEnabled(g_send_button)) {
        fprintf(stderr, "%s: Send button did not become enabled\n", label);
        return 0;
    }
    return 1;
}

int main(void) {
    const wchar_t* class_name = L"MinimaGuiProtocolIntegrationTest";
    WNDCLASSW wc; ZeroMemory(&wc, sizeof(wc));
    wc.lpfnWndProc = test_window_proc; wc.hInstance = GetModuleHandleW(NULL); wc.lpszClassName = class_name;
    if (!RegisterClassW(&wc)) { fprintf(stderr, "RegisterClassW failed: %lu\n", GetLastError()); return 1; }
    HWND parent = CreateWindowExW(0, class_name, L"", WS_POPUP, 0, 0, 100, 100, NULL, NULL, wc.hInstance, NULL);
    if (!parent) { fprintf(stderr, "CreateWindowExW failed: %lu\n", GetLastError()); return 1; }
    g_send_button = CreateWindowW(L"BUTTON", L"Send", WS_CHILD | BS_PUSHBUTTON | WS_DISABLED,
        0, 0, 80, 24, parent, (HMENU)100, wc.hInstance, NULL);
    if (!g_send_button) { fprintf(stderr, "Could not create test Send button\n"); DestroyWindow(parent); return 1; }

    int ok = check_ready_path(parent, "CHAT_READY\r\n", 12, "CRLF reader-to-window path");
    ok = check_ready_path(parent, "CHAT_READY\n", 11, "LF reader-to-window path") && ok;
    EnableWindow(g_send_button, FALSE);
    const int closed = minima_protocol_dispatch("CHAT_CLOSED\r\n", 13, post_test_chat_ready, parent);
    const int queued = pump_test_messages();
    if (closed != MINIMA_PROTOCOL_CHAT_CLOSED || queued != 0 || IsWindowEnabled(g_send_button)) {
        fprintf(stderr, "CHAT_CLOSED incorrectly enabled the Send button\n"); ok = 0;
    }
    DestroyWindow(parent); UnregisterClassW(class_name, wc.hInstance);
    if (!ok) return 1;
    puts("GUI reader-to-window readiness integration: PASS (LF, CRLF, CLOSED)");
    return 0;
}

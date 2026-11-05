#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>

#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Comdlg32.lib")

#define IDC_CORPUS       1001
#define IDC_ADD          1002
#define IDC_REMOVE       1003
#define IDC_STEPS        1004
#define IDC_RESUME       1005
#define IDC_START        1006
#define IDC_STOP         1007
#define IDC_LOG          1008
#define IDC_STATUS       1009
#define WM_APPEND_LOG    (WM_APP + 1)
#define WM_ENGINE_EXIT   (WM_APP + 2)

static HWND g_window, g_corpus, g_steps, g_resume, g_log, g_status;
static HANDLE g_process, g_thread;
static volatile LONG g_running;
static wchar_t g_engine_path[MAX_PATH];

static void append_log(const wchar_t* text) {
    int end = GetWindowTextLengthW(g_log);
    SendMessageW(g_log, EM_SETSEL, (WPARAM)end, (LPARAM)end);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)text);
    SendMessageW(g_log, EM_SCROLLCARET, 0, 0);
}

static void set_running(BOOL running) {
    InterlockedExchange(&g_running, running ? 1 : 0);
    EnableWindow(GetDlgItem(g_window, IDC_START), !running);
    EnableWindow(GetDlgItem(g_window, IDC_STOP), running);
    EnableWindow(GetDlgItem(g_window, IDC_ADD), !running);
    EnableWindow(GetDlgItem(g_window, IDC_REMOVE), !running);
    EnableWindow(g_steps, !running);
    EnableWindow(g_resume, !running);
    SetWindowTextW(g_status, running ? L"Training process running" : L"Idle");
}

static DWORD WINAPI reader_thread(void* parameter) {
    HANDLE pipe = (HANDLE)parameter;
    char buffer[2048];
    DWORD got;
    char* carry = (char*)calloc(1, 8192);
    size_t used = 0;
    if (!carry) {
        CloseHandle(pipe);
        return 1;
    }

    while (ReadFile(pipe, buffer, sizeof(buffer), &got, NULL) && got) {
        for (DWORD i = 0; i < got; ++i) {
            if (used + 2 >= 8192) used = 0;
            carry[used++] = buffer[i];
            if (buffer[i] == '\n' || used >= 8000) {
                carry[used] = 0;
                int needed = MultiByteToWideChar(CP_UTF8, 0, carry, (int)used, NULL, 0);
                if (needed <= 0)
                    needed = MultiByteToWideChar(CP_ACP, 0, carry, (int)used, NULL, 0);
                if (needed > 0) {
                    wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
                    if (wide) {
                        MultiByteToWideChar(CP_UTF8, 0, carry, (int)used, wide, needed);
                        PostMessageW(g_window, WM_APPEND_LOG, 0, (LPARAM)wide);
                    }
                }
                used = 0;
            }
        }
    }
    if (used) {
        carry[used] = 0;
        int needed = MultiByteToWideChar(CP_UTF8, 0, carry, (int)used, NULL, 0);
        if (needed > 0) {
            wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
            if (wide) {
                MultiByteToWideChar(CP_UTF8, 0, carry, (int)used, wide, needed);
                PostMessageW(g_window, WM_APPEND_LOG, 0, (LPARAM)wide);
            }
        }
    }
    free(carry);
    CloseHandle(pipe);

    if (g_process) {
        WaitForSingleObject(g_process, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(g_process, &code);
        PostMessageW(g_window, WM_ENGINE_EXIT, (WPARAM)code, 0);
    }
    return 0;
}

static BOOL append_argument(wchar_t* command, size_t capacity, const wchar_t* arg) {
    size_t used = wcslen(command);
    size_t len = wcslen(arg);
    if (used + len + 4 >= capacity) return FALSE;
    command[used++] = L' ';
    command[used++] = L'"';
    for (size_t i = 0; i < len; ++i) {
        if (arg[i] == L'"') return FALSE;
        command[used++] = arg[i];
    }
    command[used++] = L'"';
    command[used] = 0;
    return TRUE;
}

static void add_corpus_files(void) {
    wchar_t file_buffer[32768] = { 0 };
    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_window;
    ofn.lpstrFilter = L"Text files (*.txt)\0*.txt\0All files (*.*)\0*.*\0";
    ofn.lpstrFile = file_buffer;
    ofn.nMaxFile = (DWORD)(sizeof(file_buffer) / sizeof(file_buffer[0]));
    ofn.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST |
        OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    if (!GetOpenFileNameW(&ofn)) return;

    wchar_t* first = file_buffer;
    wchar_t* next = first + wcslen(first) + 1;
    if (!*next) {
        SendMessageW(g_corpus, LB_ADDSTRING, 0, (LPARAM)first);
        return;
    }
    while (*next) {
        wchar_t full[MAX_PATH * 2];
        _snwprintf_s(full, _countof(full), _TRUNCATE, L"%s\\%s", first, next);
        SendMessageW(g_corpus, LB_ADDSTRING, 0, (LPARAM)full);
        next += wcslen(next) + 1;
    }
}

static void start_engine(void) {
    if (InterlockedCompareExchange(&g_running, 0, 0)) return;

    wchar_t steps[32], command[32768];
    GetWindowTextW(g_steps, steps, _countof(steps));
    wchar_t* end = NULL;
    long count = wcstol(steps, &end, 10);
    if (!end || *end || count < 1 || count > 2147483647L) {
        MessageBoxW(g_window, L"Enter a positive training-step count.", L"ProjectMinima", MB_ICONWARNING);
        return;
    }

    _snwprintf_s(command, _countof(command), _TRUNCATE, L"\"%s\" --steps %ld", g_engine_path, count);
    if (SendMessageW(g_resume, BM_GETCHECK, 0, 0) == BST_CHECKED) {
        wcscat_s(command, _countof(command), L" --resume");
    }

    int corpus_count = (int)SendMessageW(g_corpus, LB_GETCOUNT, 0, 0);
    for (int i = 0; i < corpus_count; ++i) {
        wchar_t path[MAX_PATH * 2];
        SendMessageW(g_corpus, LB_GETTEXT, i, (LPARAM)path);
        if (!append_argument(command, _countof(command), path)) {
            MessageBoxW(g_window, L"Corpus command line is too long or contains an unsupported quote.", L"ProjectMinima", MB_ICONERROR);
            return;
        }
    }

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE read_pipe = NULL, write_pipe = NULL;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        MessageBoxW(g_window, L"Could not create the engine output pipe.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    wchar_t working_dir[MAX_PATH];
    wcscpy_s(working_dir, _countof(working_dir), g_engine_path);
    wchar_t* slash = wcsrchr(working_dir, L'\\');
    if (slash) *slash = 0;

    if (!CreateProcessW(g_engine_path, command, NULL, NULL, TRUE,
        CREATE_NO_WINDOW, NULL, working_dir, &si, &pi)) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        MessageBoxW(g_window, L"Could not start Minima.exe. Make sure it is beside MinimaGUI.exe.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    CloseHandle(write_pipe);
    g_process = pi.hProcess;
    CloseHandle(pi.hThread);
    set_running(TRUE);
    append_log(L"\r\n--- Starting Minima engine ---\r\n");
    g_thread = CreateThread(NULL, 0, reader_thread, read_pipe, 0, NULL);
    if (!g_thread) {
        TerminateProcess(g_process, 1);
        CloseHandle(g_process);
        g_process = NULL;
        CloseHandle(read_pipe);
        set_running(FALSE);
        MessageBoxW(g_window, L"Could not create the output-reader thread.", L"ProjectMinima", MB_ICONERROR);
    }
}

static void stop_engine(void) {
    if (g_process && InterlockedCompareExchange(&g_running, 0, 0)) {
        /* The engine checkpoints at its configured reporting intervals.
           Termination can lose work since the most recent checkpoint. */
        if (MessageBoxW(g_window,
            L"Stop Minima now? Work since the last saved checkpoint may be lost.",
            L"Stop training", MB_YESNO | MB_ICONWARNING) == IDYES)
            TerminateProcess(g_process, 2);
    }
}

static void create_controls(HWND hwnd) {
    HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    CreateWindowW(L"STATIC", L"Corpus files", WS_CHILD | WS_VISIBLE, 12, 12, 130, 20, hwnd, NULL, NULL, NULL);
    g_corpus = CreateWindowW(L"LISTBOX", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | LBS_EXTENDEDSEL | WS_VSCROLL,
        12, 34, 520, 105, hwnd, (HMENU)IDC_CORPUS, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Add files...", WS_CHILD | WS_VISIBLE, 12, 146, 100, 26, hwnd, (HMENU)IDC_ADD, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Remove selected", WS_CHILD | WS_VISIBLE, 120, 146, 130, 26, hwnd, (HMENU)IDC_REMOVE, NULL, NULL);

    CreateWindowW(L"STATIC", L"Additional training steps:", WS_CHILD | WS_VISIBLE, 12, 184, 160, 20, hwnd, NULL, NULL, NULL);
    g_steps = CreateWindowW(L"EDIT", L"2000", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        176, 181, 90, 24, hwnd, (HMENU)IDC_STEPS, NULL, NULL);
    g_resume = CreateWindowW(L"BUTTON", L"Resume existing checkpoint", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
        282, 182, 220, 24, hwnd, (HMENU)IDC_RESUME, NULL, NULL);

    CreateWindowW(L"BUTTON", L"Start training", WS_CHILD | WS_VISIBLE, 12, 216, 120, 30, hwnd, (HMENU)IDC_START, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Stop", WS_CHILD | WS_VISIBLE | WS_DISABLED, 142, 216, 80, 30, hwnd, (HMENU)IDC_STOP, NULL, NULL);
    g_status = CreateWindowW(L"STATIC", L"Idle", WS_CHILD | WS_VISIBLE, 240, 221, 280, 20, hwnd, (HMENU)IDC_STATUS, NULL, NULL);

    CreateWindowW(L"STATIC", L"Engine output / training metrics", WS_CHILD | WS_VISIBLE, 12, 258, 300, 20, hwnd, NULL, NULL, NULL);
    g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
        12, 280, 720, 260, hwnd, (HMENU)IDC_LOG, NULL, NULL);

    HWND controls[] = { g_corpus, g_steps, g_resume, g_log };
    for (int i = 0; i < (int)_countof(controls); ++i)
        SendMessageW(controls[i], WM_SETFONT, (WPARAM)font, TRUE);
}

static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        g_window = hwnd;
        create_controls(hwnd);
        return 0;
    case WM_SIZE: {
        RECT r;
        GetClientRect(hwnd, &r);
        MoveWindow(g_corpus, 12, 34, r.right - 24, 105, TRUE);
        MoveWindow(g_log, 12, 280, r.right - 24, r.bottom - 292, TRUE);
        return 0;
    }
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_ADD: add_corpus_files(); return 0;
        case IDC_REMOVE: {
            int selected[128];
            LRESULT n = SendMessageW(g_corpus, LB_GETSELITEMS, 128, (LPARAM)selected);
            for (int i = (int)n - 1; i >= 0; --i)
                SendMessageW(g_corpus, LB_DELETESTRING, selected[i], 0);
            return 0;
        }
        case IDC_START: start_engine(); return 0;
        case IDC_STOP: stop_engine(); return 0;
        }
        break;
    case WM_APPEND_LOG:
        append_log((const wchar_t*)lp);
        free((void*)lp);
        return 0;
    case WM_ENGINE_EXIT:
        if (g_process) { CloseHandle(g_process); g_process = NULL; }
        if (g_thread) { CloseHandle(g_thread); g_thread = NULL; }
        set_running(FALSE);
        append_log(wp == 0 ? L"\r\n--- Engine exited successfully ---\r\n" : L"\r\n--- Engine exited with an error ---\r\n");
        return 0;
    case WM_CLOSE:
        if (g_process && InterlockedCompareExchange(&g_running, 0, 0)) {
            if (MessageBoxW(hwnd, L"Training is still running. Stop it and close?",
                L"ProjectMinima", MB_YESNO | MB_ICONWARNING) != IDYES) return 0;
            TerminateProcess(g_process, 2);
            WaitForSingleObject(g_process, 3000);
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command_line, int show) {
    (void)previous; (void)command_line;
    INITCOMMONCONTROLSEX controls = { sizeof(controls), ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS };
    InitCommonControlsEx(&controls);

    DWORD n = GetModuleFileNameW(NULL, g_engine_path, _countof(g_engine_path));
    if (!n || n >= _countof(g_engine_path)) return 1;
    wchar_t* slash = wcsrchr(g_engine_path, L'\\');
    if (!slash) return 1;
    wcscpy_s(slash + 1, _countof(g_engine_path) - (size_t)(slash + 1 - g_engine_path), L"Minima.exe");

    WNDCLASSW wc;
    ZeroMemory(&wc, sizeof(wc));
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"ProjectMinimaWin32";
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    if (!RegisterClassW(&wc)) return 1;

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"ProjectMinima",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 760, 600,
        NULL, NULL, instance, NULL);
    if (!hwnd) return 1;
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}

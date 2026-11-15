#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include "engine_io.h"
#include "gui_protocol.h"

typedef struct MinimaEngineReaderContext {
    HWND window;
    HANDLE process;
    UINT append_log_message;
    UINT engine_exit_message;
    UINT chat_ready_message;
    HANDLE output_pipe;
} MinimaEngineReaderContext;

static void post_chat_ready(void* parameter) {
    MinimaEngineReaderContext* context = (MinimaEngineReaderContext*)parameter;
    PostMessageW(context->window, context->chat_ready_message, 0, 0);
}

static DWORD WINAPI reader_thread(void* parameter) {
    MinimaEngineReaderContext* context = (MinimaEngineReaderContext*)parameter;
    HANDLE pipe = context->output_pipe;
    char buffer[2048];
    DWORD got;
    char* carry = (char*)calloc(1, 65536);
    size_t used = 0;
    if (!carry) {
        CloseHandle(pipe);
        if (context->process) TerminateProcess(context->process, 1);
        PostMessageW(context->window, context->engine_exit_message, 1, 0);
        free(context);
        return 1;
    }

    while (ReadFile(pipe, buffer, sizeof(buffer), &got, NULL) && got) {
        for (DWORD i = 0; i < got; ++i) {
            if (used + 2 >= 65536) used = 0;
            carry[used++] = buffer[i];
            if (buffer[i] == '\n' || used >= 60000) {
                carry[used] = 0;
                const int marker = minima_protocol_dispatch(
                    carry, used, post_chat_ready, context);
                if (marker == MINIMA_PROTOCOL_CHAT_READY ||
                    marker == MINIMA_PROTOCOL_CHAT_CLOSED) {
                    used = 0;
                    continue;
                }
                UINT code_page = CP_UTF8;
                int needed = MultiByteToWideChar(code_page, 0, carry,
                    (int)used, NULL, 0);
                if (needed <= 0) {
                    code_page = CP_ACP;
                    needed = MultiByteToWideChar(code_page, 0, carry,
                        (int)used, NULL, 0);
                }
                if (needed > 0) {
                    wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
                    if (wide) {
                        MultiByteToWideChar(code_page, 0, carry, (int)used, wide, needed);
                        PostMessageW(context->window, context->append_log_message,
                            0, (LPARAM)wide);
                    }
                }
                used = 0;
            }
        }
    }

    if (used) {
        carry[used] = 0;
        UINT code_page = CP_UTF8;
        int needed = MultiByteToWideChar(code_page, 0, carry, (int)used, NULL, 0);
        if (needed <= 0) {
            code_page = CP_ACP;
            needed = MultiByteToWideChar(code_page, 0, carry, (int)used, NULL, 0);
        }
        if (needed > 0) {
            wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
            if (wide) {
                MultiByteToWideChar(code_page, 0, carry, (int)used, wide, needed);
                PostMessageW(context->window, context->append_log_message,
                    0, (LPARAM)wide);
            }
        }
    }

    free(carry);
    CloseHandle(pipe);
    if (context->process) {
        WaitForSingleObject(context->process, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(context->process, &code);
        PostMessageW(context->window, context->engine_exit_message, (WPARAM)code, 0);
    }
    free(context);
    return 0;
}

HANDLE minima_engine_start_reader(
    HANDLE output_pipe, HWND window, HANDLE process,
    UINT append_log_message, UINT engine_exit_message, UINT chat_ready_message) {
    if (!output_pipe || !window || !process) return NULL;
    MinimaEngineReaderContext* context =
        (MinimaEngineReaderContext*)calloc(1, sizeof(*context));
    if (!context) return NULL;
    context->window = window;
    context->process = process;
    context->append_log_message = append_log_message;
    context->engine_exit_message = engine_exit_message;
    context->chat_ready_message = chat_ready_message;
    context->output_pipe = output_pipe;

    HANDLE thread = CreateThread(NULL, 0, reader_thread, context, 0, NULL);
    if (!thread) free(context);
    return thread;
}

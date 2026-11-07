#ifndef MINIMA_GUI_ENGINE_IO_H
#define MINIMA_GUI_ENGINE_IO_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

HANDLE minima_engine_start_reader(
    HANDLE output_pipe,
    HWND window,
    HANDLE process,
    UINT append_log_message,
    UINT engine_exit_message,
    UINT chat_ready_message);

#endif

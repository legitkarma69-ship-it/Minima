#ifndef MINIMA_GUI_CHAT_STATE_H
#define MINIMA_GUI_CHAT_STATE_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* Shared by the real window procedure and the Windows integration test. */
static void minima_gui_set_chat_send_ready(HWND send_button,
    BOOL chat_mode, BOOL engine_running) {
    if (send_button)
        EnableWindow(send_button, chat_mode && engine_running);
}

#endif

#ifndef MINIMA_GUI_ACTIONS_H
#define MINIMA_GUI_ACTIONS_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef struct {
    void (*add_corpus)(HWND);
    void (*remove_corpus)(HWND);
    void (*resume_changed)(HWND);
    void (*start_training)(HWND);
    void (*stop_training)(HWND);
    void (*start_chat)(HWND);
    void (*rebuild_tokenizer)(HWND);
    void (*auto_settings)(HWND);
    void (*basic_defaults)(HWND);
    void (*advanced_toggle)(HWND);
    void (*chat_send)(HWND);
    void (*chat_test)(HWND);
    void (*chat_stop)(HWND);
    void (*save_checkpoint)(HWND);
} GuiActionHandlers;
BOOL gui_actions_dispatch(HWND hwnd, UINT command_id,
    const GuiActionHandlers* handlers);
#endif

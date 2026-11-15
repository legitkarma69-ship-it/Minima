#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "gui_actions.h"
#include "gui_layout.h"

BOOL gui_actions_dispatch(HWND hwnd, UINT command_id,
    const GuiActionHandlers* handlers) {
    if (!handlers) return FALSE;
    switch (command_id) {
    case IDC_ADD: if (handlers->add_corpus) handlers->add_corpus(hwnd); return TRUE;
    case IDC_REMOVE: if (handlers->remove_corpus) handlers->remove_corpus(hwnd); return TRUE;
    case IDC_RESUME: if (handlers->resume_changed) handlers->resume_changed(hwnd); return TRUE;
    case IDC_START: if (handlers->start_training) handlers->start_training(hwnd); return TRUE;
    case IDC_STOP: if (handlers->stop_training) handlers->stop_training(hwnd); return TRUE;
    case IDC_CHAT_START: if (handlers->start_chat) handlers->start_chat(hwnd); return TRUE;
    case IDC_REBUILD_TOKENIZER: if (handlers->rebuild_tokenizer) handlers->rebuild_tokenizer(hwnd); return TRUE;
    case IDC_AUTO_SETTINGS: if (handlers->auto_settings) handlers->auto_settings(hwnd); return TRUE;
    case IDC_BASIC_DEFAULTS: if (handlers->basic_defaults) handlers->basic_defaults(hwnd); return TRUE;
    case IDC_ADVANCED_TOGGLE: if (handlers->advanced_toggle) handlers->advanced_toggle(hwnd); return TRUE;
    case IDC_CHAT_SEND: if (handlers->chat_send) handlers->chat_send(hwnd); return TRUE;
    case IDC_CHAT_TEST: if (handlers->chat_test) handlers->chat_test(hwnd); return TRUE;
    case IDC_CHAT_STOP: if (handlers->chat_stop) handlers->chat_stop(hwnd); return TRUE;
    case IDC_SAVE_NOW: if (handlers->save_checkpoint) handlers->save_checkpoint(hwnd); return TRUE;
    default: return FALSE;
    }
}

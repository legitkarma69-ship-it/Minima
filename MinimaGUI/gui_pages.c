#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include "gui_pages.h"
#include "gui_layout.h"

void gui_pages_create_shell(HWND parent, HWND* tabs, HWND* corpus_group,
    HWND* model_group, HWND* training_group, HWND* advanced_toggle,
    HWND* monitor_group, HWND* output_group) {
    if (tabs) {
        *tabs = CreateWindowW(WC_TABCONTROLW, L"",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP,
            12, 6, 920, 28, parent, (HMENU)IDC_MAIN_TABS,
            GetModuleHandleW(NULL), NULL);
        if (*tabs) {
            TCITEMW item;
            ZeroMemory(&item, sizeof(item));
            item.mask = TCIF_TEXT;
            item.pszText = L"Training";
            TabCtrl_InsertItem(*tabs, 0, &item);
            item.pszText = L"Advanced";
            TabCtrl_InsertItem(*tabs, 1, &item);
            item.pszText = L"Chat & Generation";
            TabCtrl_InsertItem(*tabs, 2, &item);
        }
    }
    if (corpus_group)
        *corpus_group = CreateWindowW(L"BUTTON", L"Corpus",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 8, 6, 920, 168,
            parent, NULL, NULL, NULL);
    if (model_group)
        *model_group = CreateWindowW(L"BUTTON", L"Model",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 8, 168, 585, 38,
            parent, NULL, NULL, NULL);
    if (training_group)
        *training_group = CreateWindowW(L"BUTTON", L"Training",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 8, 202, 585, 38,
            parent, NULL, NULL, NULL);
    if (advanced_toggle) {
        *advanced_toggle = CreateWindowW(L"BUTTON", L"Show advanced settings",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
            8, 236, 220, 24, parent, (HMENU)IDC_ADVANCED_TOGGLE,
            NULL, NULL);
        if (*advanced_toggle)
            SendMessageW(*advanced_toggle, BM_SETCHECK, BST_CHECKED, 0);
    }
    if (monitor_group)
        *monitor_group = CreateWindowW(L"BUTTON", L"Monitor",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 590, 150, 340, 205,
            parent, NULL, NULL, NULL);
    if (output_group)
        *output_group = CreateWindowW(L"BUTTON", L"Output / chat",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 8, 375, 920, 350,
            parent, NULL, NULL, NULL);
}

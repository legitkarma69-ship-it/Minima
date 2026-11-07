#ifndef MINIMA_GUI_PAGES_H
#define MINIMA_GUI_PAGES_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
void gui_pages_create_shell(HWND parent, HWND* tabs, HWND* corpus_group,
    HWND* model_group, HWND* training_group, HWND* advanced_toggle,
    HWND* monitor_group, HWND* output_group);
#endif

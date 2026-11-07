#ifndef MINIMA_GUI_GRAPH_H
#define MINIMA_GUI_GRAPH_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void minima_graph_set_metrics_path(const wchar_t* path);
void minima_graph_set_dpi(UINT dpi);
void minima_graph_refresh(void);
LRESULT CALLBACK minima_graph_window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

#endif

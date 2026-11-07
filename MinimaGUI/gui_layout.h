#ifndef MINIMA_GUI_LAYOUT_H
#define MINIMA_GUI_LAYOUT_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define IDC_CORPUS 1001
#define IDC_STEPS 1004
#define IDC_LOG 1008
#define IDC_CHAT_INPUT 1011
#define IDC_ADD 1002
#define IDC_REMOVE 1003
#define IDC_RESUME 1005
#define IDC_START 1006
#define IDC_STOP 1007
#define IDC_STATUS 1009
#define IDC_CHAT_START 1010
#define IDC_CHAT_SEND 1012
#define IDC_CHAT_STOP 1013
#define IDC_SAVE_NOW 1014
#define IDC_REBUILD_TOKENIZER 1015
#define IDC_AUTO_SETTINGS 1016
#define IDC_ADVANCED_TOGGLE 1017
#define IDC_CHAT_TEST 1018
#define IDC_MAIN_TABS 1019
#define IDC_BASIC_DEFAULTS 1020
typedef struct { HWND hwnd; RECT rect; } GuiInitialRect;
extern HWND g_window, g_corpus, g_log, g_main_tabs;
extern HWND g_chat_input, g_chat_test, g_graph;
extern HWND g_group_corpus, g_group_model, g_group_training;
extern HWND g_group_advanced, g_group_monitor, g_group_output, g_output_label;
extern HWND g_advanced_controls[32];
extern int g_advanced_control_count;
extern GuiInitialRect g_initial_rects[160];
extern int g_initial_rect_count;
extern int g_active_page;
extern BOOL g_advanced_expanded;
extern UINT g_dpi;
void gui_layout_layout_controls(HWND hwnd);
void gui_layout_select_main_page(HWND hwnd, int page);
void gui_layout_scale_children(HWND hwnd, UINT old_dpi, UINT new_dpi);
#endif

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
#define IDC_CLEAR_CHAT 1021
#define IDC_CLEAR_LOGS 1022
#define IDC_RUN_EVAL 1023
#define IDC_CPU_LIMIT_LABEL 1024
#define IDC_RAM_LIMIT_LABEL 1025
#define IDC_CPU_LIMIT 1026
#define IDC_RAM_LIMIT 1027
#define IDC_ARENA_BUDGET_LABEL 1028
#define IDC_ARENA_BUDGET 1029
#define IDC_CPU_WARNING_LABEL 1030
#define IDC_RAM_WARNING_LABEL 1031
#define IDC_CPU_WARNING 1032
#define IDC_RAM_WARNING 1033
#define IDC_RESOURCE_PROFILE_LABEL 1034
#define IDC_RESOURCE_PROFILE 1035
#define IDC_PAUSE 1036
#define IDC_SAVE_PROFILE 1037
#define IDC_LOAD_PROFILE 1038
typedef struct { HWND hwnd; RECT rect; } GuiInitialRect;
extern HWND g_window, g_corpus, g_log, g_main_tabs, g_chat_display, g_log_label;
extern HWND g_chat_input, g_chat_test, g_graph;
extern HWND g_cpu_limit, g_ram_limit, g_arena_budget, g_cpu_warning, g_ram_warning, g_resource_profile, g_resource_status;
extern HWND g_clear_chat, g_clear_logs, g_eval_button;
extern HWND g_group_corpus, g_group_model, g_group_training;
extern HWND g_group_advanced, g_group_monitor, g_group_output, g_output_label;
extern HWND g_advanced_controls[64];
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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include "gui_layout.h"

static int scale_px(int value) {
    return MulDiv(value, (int)(g_dpi ? g_dpi : 96), 96);
}

static BOOL is_advanced_page_control(HWND child) {
    if (child == g_group_advanced) return TRUE;
    if (child == g_chat_test) return TRUE;
    for (int i = 0; i < g_advanced_control_count; ++i)
        if (g_advanced_controls[i] == child) return TRUE;
    return FALSE;
}

static BOOL is_chat_page_control(HWND child) {
    return child == g_group_output || child == g_output_label || child == g_log ||
        child == g_chat_input ||
        child == GetDlgItem(g_window, IDC_CHAT_START) ||
        child == GetDlgItem(g_window, IDC_CHAT_SEND) ||
        child == GetDlgItem(g_window, IDC_CHAT_STOP);
}

static BOOL CALLBACK apply_page_visibility(HWND child, LPARAM page_parameter) {
    const int page = (int)page_parameter;
    if (child == g_main_tabs) return TRUE;
    BOOL visible;
    if (page == 0)
        visible = !is_advanced_page_control(child) && !is_chat_page_control(child);
    else if (page == 1)
        visible = is_advanced_page_control(child) &&
            (child == g_group_advanced || g_advanced_expanded);
    else
        visible = is_chat_page_control(child);
    ShowWindow(child, visible ? SW_SHOWNA : SW_HIDE);
    return TRUE;
}

void gui_layout_select_main_page(HWND hwnd, int page) {
    if (page < 0 || page > 2) return;
    g_active_page = page;
    if (g_main_tabs) TabCtrl_SetCurSel(g_main_tabs, page);
    EnumChildWindows(hwnd, apply_page_visibility, (LPARAM)page);
    gui_layout_layout_controls(hwnd);
    InvalidateRect(hwnd, NULL, TRUE);
}

void gui_layout_layout_controls(HWND hwnd) {
    RECT client; GetClientRect(hwnd,&client);
    const int width=client.right, height=client.bottom;
    const int pad=scale_px(12), gap=scale_px(12), inner=width-2*pad;
    if(inner<=0)return;
    const int corpus_h=scale_px(126), work_top=corpus_h+scale_px(44);
    const int bottom_h=scale_px(82), action_h=scale_px(38);
    const int output_top=(g_active_page==2?scale_px(46):work_top+scale_px(368));
    const int output_bottom=height-bottom_h-scale_px(8);
    const int output_h=output_bottom-output_top;
    const int left_w=(inner-gap)*56/100, right_x=pad+left_w+gap, right_w=inner-left_w-gap;
    const int advanced_y=work_top+scale_px(106);
    if(g_main_tabs)MoveWindow(g_main_tabs,pad,scale_px(6),inner,scale_px(28),TRUE);
    if(g_group_corpus)MoveWindow(g_group_corpus,scale_px(8),scale_px(40),width-scale_px(16),corpus_h,TRUE);
    if(g_corpus)MoveWindow(g_corpus,pad,scale_px(64),inner,scale_px(58),TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_ADD),pad,scale_px(128),scale_px(104),scale_px(26),TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_REMOVE),pad+scale_px(112),scale_px(128),scale_px(136),scale_px(26),TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_BASIC_DEFAULTS),pad+scale_px(256),scale_px(128),scale_px(132),scale_px(26),TRUE);
    if(g_group_model)MoveWindow(g_group_model,pad,work_top,left_w,scale_px(48),TRUE);
    if(g_group_training)MoveWindow(g_group_training,pad,work_top+scale_px(54),left_w,scale_px(48),TRUE);
    if(g_group_advanced)MoveWindow(g_group_advanced,pad,advanced_y,scale_px(220),scale_px(24),TRUE);
    if(g_group_monitor)MoveWindow(g_group_monitor,right_x,work_top,right_w,scale_px(306),TRUE);
    if(g_graph)MoveWindow(g_graph,right_x+scale_px(8),work_top+scale_px(22),right_w-scale_px(16),scale_px(272),TRUE);

    const float xs=(float)(left_w-scale_px(20))/(float)scale_px(670);
    for(int i=0;i<g_initial_rect_count;i++){
        HWND child=g_initial_rects[i].hwnd; RECT r=g_initial_rects[i].rect;
        if(!IsWindow(child)||child==g_group_corpus||child==g_group_model||child==g_group_training||
           child==g_group_advanced||child==g_group_monitor||child==g_group_output||child==g_main_tabs||child==g_corpus||
           child==g_graph||child==g_log||child==g_chat_input||child==g_chat_test)continue;
        int id=GetDlgCtrlID(child);
        if(id==IDC_ADD||id==IDC_REMOVE||id==IDC_BASIC_DEFAULTS||id==IDC_START||id==IDC_STOP||id==IDC_CHAT_START||
           id==IDC_SAVE_NOW||id==IDC_REBUILD_TOKENIZER||id==IDC_AUTO_SETTINGS||id==IDC_STATUS||
           id==IDC_CHAT_SEND||id==IDC_CHAT_STOP||id==IDC_LOG)continue;
        int oy=r.top;
        if(oy>=scale_px(168)&&oy<scale_px(250)){
            int x=pad+scale_px(8)+(int)((r.left-scale_px(12))*xs);
            int y=work_top+scale_px(19)+(oy-scale_px(168));
            int w=(int)((r.right-r.left)*xs);
            MoveWindow(child,x,y,w>scale_px(20)?w:scale_px(20),r.bottom-r.top,TRUE);
        }else if(oy>=scale_px(258)&&oy<scale_px(355)){
            int x=pad+scale_px(8)+(int)((r.left-scale_px(12))*xs);
            int y=advanced_y+scale_px(28)+(oy-scale_px(258));
            int w=(int)((r.right-r.left)*xs);
            MoveWindow(child,x,y,w>scale_px(20)?w:scale_px(20),r.bottom-r.top,TRUE);
        }
    }
    const int actions_y=work_top+scale_px(320);
    MoveWindow(GetDlgItem(hwnd,IDC_START),pad,actions_y,scale_px(116),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_STOP),pad+scale_px(122),actions_y,scale_px(100),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_CHAT_START),pad+scale_px(228),actions_y,scale_px(96),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_SAVE_NOW),pad+scale_px(330),actions_y,scale_px(112),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_REBUILD_TOKENIZER),pad+scale_px(448),actions_y,scale_px(132),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_AUTO_SETTINGS),pad+scale_px(586),actions_y,scale_px(112),action_h,TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_STATUS),pad+scale_px(706),actions_y+scale_px(8),scale_px(150),scale_px(22),TRUE);
    if(g_group_output)MoveWindow(g_group_output,pad,output_top,inner,output_h,TRUE);
    if(g_output_label)MoveWindow(g_output_label,pad+scale_px(8),output_top+scale_px(4),scale_px(300),scale_px(18),TRUE);
    if(g_log)MoveWindow(g_log,pad,output_top+scale_px(24),inner,output_h-scale_px(68),TRUE);
    const int chat_y=height-bottom_h+scale_px(4);
    if(g_chat_input)MoveWindow(g_chat_input,g_active_page==2?pad+scale_px(112):pad,chat_y,inner-scale_px(340)-scale_px(112),scale_px(180),TRUE);
    if(g_active_page==2)MoveWindow(GetDlgItem(hwnd,IDC_CHAT_START),pad,chat_y,scale_px(104),scale_px(28),TRUE);
    if(g_chat_test)MoveWindow(g_chat_test,right_x,advanced_y+scale_px(28),scale_px(120),scale_px(28),TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_CHAT_SEND),width-pad-scale_px(218),chat_y,scale_px(108),scale_px(28),TRUE);
    MoveWindow(GetDlgItem(hwnd,IDC_CHAT_STOP),width-pad-scale_px(104),chat_y,scale_px(104),scale_px(28),TRUE);
}



typedef struct { HWND parent; UINT old_dpi; UINT new_dpi; } DpiScaleContext;

static BOOL CALLBACK scale_child_for_dpi(HWND child, LPARAM parameter) {
    DpiScaleContext* context = (DpiScaleContext*)parameter;
    RECT rect;
    if (!GetWindowRect(child, &rect)) return TRUE;
    MapWindowPoints(NULL, context->parent, (POINT*)&rect, 2);
    const int x = MulDiv(rect.left, (int)context->new_dpi, (int)context->old_dpi);
    const int y = MulDiv(rect.top, (int)context->new_dpi, (int)context->old_dpi);
    const int width = MulDiv(rect.right - rect.left, (int)context->new_dpi, (int)context->old_dpi);
    const int height = MulDiv(rect.bottom - rect.top, (int)context->new_dpi, (int)context->old_dpi);
    MoveWindow(child, x, y, width, height, FALSE);
    return TRUE;
}

void gui_layout_scale_children(HWND hwnd, UINT old_dpi, UINT new_dpi) {
    if (!old_dpi) old_dpi = 96;
    if (!new_dpi) new_dpi = old_dpi;
    DpiScaleContext context = { hwnd, old_dpi, new_dpi };
    EnumChildWindows(hwnd, scale_child_for_dpi, (LPARAM)&context);
}

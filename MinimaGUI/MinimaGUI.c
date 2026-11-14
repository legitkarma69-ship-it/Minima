#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "gui_protocol.h"
#include "gui_chat_state.h"
#include "graph.h"
#include "settings.h"
#include "engine_io.h"
#include "gui_layout.h"
#include "gui_pages.h"
#include "gui_actions.h"

#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Comdlg32.lib")
#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#define WM_APPEND_LOG    (WM_APP + 1)
#define WM_ENGINE_EXIT   (WM_APP + 2)
#define WM_CHAT_READY    (WM_APP + 3)
#define IDT_GRAPH_REFRESH 1

HWND g_window, g_corpus, g_log;
static HWND g_steps, g_resume, g_status;
HWND g_main_tabs;
static MinimaSettingsContext g_settings_context;
int g_active_page;
static HWND g_tab_controls[64];
static int g_tab_control_count;
static HWND g_lr, g_warmup, g_total_steps, g_log_interval, g_validation_batches;
static HWND g_grad_clip;
static HWND g_checkpoint_interval, g_seed, g_tokenizer_bytes, g_validation_interval;
HWND g_graph;
static HWND g_temperature, g_top_k, g_top_p, g_repetition, g_max_tokens;
HWND g_chat_input, g_chat_test;
static HWND g_vocab, g_dim, g_layers, g_heads, g_ffn, g_max_seq;
static HWND g_lr_schedule, g_tokenizer_sampling, g_corpus_mix, g_batch_size, g_sequence_length;
static wchar_t g_metrics_path[MAX_PATH];
static wchar_t g_settings_path[MAX_PATH];
static HANDLE g_process, g_thread, g_stdin_write;
static BOOL g_chat_mode;
static BOOL g_chat_send_when_ready;
static volatile LONG g_running;
static BOOL g_close_when_done;
static BOOL g_tokenizer_rebuild_mode;
static BOOL g_use_saved_tokenizer;
static BOOL g_fresh_used_saved_tokenizer;
static wchar_t g_prebuilt_vocab[64],g_prebuilt_tokenizer_bytes[64],g_prebuilt_sampling[64];
static wchar_t g_engine_path[MAX_PATH];
static HWND g_tooltip;
HWND g_group_corpus, g_group_model, g_group_training, g_output_label;
HWND g_group_advanced, g_group_monitor, g_group_output;
BOOL g_advanced_expanded = FALSE;
UINT g_dpi = 96;
static HFONT g_ui_font;
HWND g_advanced_controls[64];
int g_advanced_control_count;
GuiInitialRect g_initial_rects[160];
int g_initial_rect_count;
static BOOL CALLBACK capture_initial_rect(HWND child, LPARAM unused) {
    (void)unused;
    if (g_initial_rect_count >= (int)_countof(g_initial_rects)) return FALSE;
    RECT r; if (!GetWindowRect(child, &r)) return TRUE;
    MapWindowPoints(NULL, g_window, (POINT*)&r, 2);
    g_initial_rects[g_initial_rect_count].hwnd=child;
    g_initial_rects[g_initial_rect_count].rect=r;
    ++g_initial_rect_count; return TRUE;
}

static void register_advanced_control(HWND control) {
    if (control && g_advanced_control_count < (int)_countof(g_advanced_controls))
        g_advanced_controls[g_advanced_control_count++] = control;
}

static int scale_px(int value) {
    return MulDiv(value, (int)(g_dpi ? g_dpi : 96), 96);
}

static BOOL CALLBACK set_child_font(HWND child, LPARAM font) {
    SendMessageW(child, WM_SETFONT, (WPARAM)font, TRUE);
    return TRUE;
}

static void rebuild_ui_font(HWND hwnd) {
    HFONT old_font = g_ui_font;
    g_ui_font = CreateFontW(-MulDiv(9, (int)(g_dpi ? g_dpi : 96), 72),
        0, 0, 0, FW_NORMAL,
        FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");
    if (!g_ui_font) g_ui_font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    EnumChildWindows(hwnd, set_child_font, (LPARAM)g_ui_font);
    if (old_font && old_font != (HFONT)GetStockObject(DEFAULT_GUI_FONT))
        DeleteObject(old_font);
}

static void set_advanced_expanded(HWND hwnd, BOOL expanded) {
    g_advanced_expanded = expanded ? TRUE : FALSE;
    SendMessageW(g_group_advanced, BM_SETCHECK,
        g_advanced_expanded ? BST_CHECKED : BST_UNCHECKED, 0);
    for (int i = 0; i < g_advanced_control_count; ++i)
        ShowWindow(g_advanced_controls[i], g_advanced_expanded ? SW_SHOWNA : SW_HIDE);
    if (g_settings_path[0])
        WritePrivateProfileStringW(L"UI", L"advanced_expanded",
            g_advanced_expanded ? L"1" : L"0", g_settings_path);
    gui_layout_layout_controls(hwnd);
    InvalidateRect(hwnd, NULL, TRUE);
}

static void add_tooltip(HWND target, const wchar_t* text) {
    if (!g_tooltip || !target || !text) return;
    TOOLINFOW info;
    ZeroMemory(&info, sizeof(info));
    info.cbSize = sizeof(info);
    info.uFlags = TTF_IDISHWND | TTF_SUBCLASS;
    info.hwnd = g_window;
    info.uId = (UINT_PTR)target;
    info.lpszText = (LPWSTR)text;
    SendMessageW(g_tooltip, TTM_ADDTOOLW, 0, (LPARAM)&info);
}

static void save_chat_text(const wchar_t* text) {
    if(!g_chat_mode || !g_engine_path[0])return;
    wchar_t path[MAX_PATH];wcscpy_s(path,_countof(path),g_engine_path);
    wchar_t* slash=wcsrchr(path,L'\\');
    if(slash)wcscpy_s(slash+1,_countof(path)-(size_t)(slash+1-path),L"minima_chat.txt");
    int n=WideCharToMultiByte(CP_UTF8,0,text,-1,NULL,0,NULL,NULL);
    if(n<=1)return;
    char* utf8=(char*)malloc((size_t)n);if(!utf8)return;
    WideCharToMultiByte(CP_UTF8,0,text,-1,utf8,n,NULL,NULL);
    FILE* f=_wfopen(path,L"ab");if(f){fwrite(utf8,1,(size_t)n-1,f);fclose(f);}
    free(utf8);
}

static void append_log(const wchar_t* text) {
    int end=GetWindowTextLengthW(g_log);
    SendMessageW(g_log,EM_SETSEL,(WPARAM)end,(LPARAM)end);
    SendMessageW(g_log,EM_REPLACESEL,FALSE,(LPARAM)text);
    SendMessageW(g_log,EM_SCROLLCARET,0,0);
    save_chat_text(text);
}

static void update_fresh_only_controls(void) {
    BOOL fresh=SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED;
    BOOL enabled=!InterlockedCompareExchange(&g_running,0,0) && fresh;
    EnableWindow(g_seed,enabled);
    EnableWindow(g_tokenizer_bytes,enabled);
    EnableWindow(g_tokenizer_sampling,enabled);
}

static void set_running(BOOL running) {
    InterlockedExchange(&g_running, running ? 1 : 0);
    EnableWindow(GetDlgItem(g_window, IDC_START), !running);
    EnableWindow(GetDlgItem(g_window, IDC_STOP), running);
    EnableWindow(GetDlgItem(g_window, IDC_SAVE_NOW), running && !g_chat_mode);
    EnableWindow(GetDlgItem(g_window, IDC_CHAT_START), !running);
    EnableWindow(GetDlgItem(g_window, IDC_ADD), !running);
    EnableWindow(GetDlgItem(g_window, IDC_REMOVE), !running);
    EnableWindow(g_steps,!running);EnableWindow(g_resume,!running);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_START),!running);
    EnableWindow(GetDlgItem(g_window,IDC_REBUILD_TOKENIZER),!running);
    EnableWindow(GetDlgItem(g_window,IDC_AUTO_SETTINGS),!running);
    EnableWindow(GetDlgItem(g_window,IDC_BASIC_DEFAULTS),!running);
    HWND fields[]={g_lr,g_warmup,g_total_steps,g_log_interval,g_validation_batches,g_checkpoint_interval,
        g_seed,g_tokenizer_bytes,g_validation_interval,g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,
        g_batch_size,g_sequence_length,g_lr_schedule,g_tokenizer_sampling,g_corpus_mix,g_grad_clip};
    for(int i=0;i<(int)_countof(fields);i++)if(fields[i])EnableWindow(fields[i],!running);
    EnableWindow(g_chat_input, !running || g_chat_mode);
    /* Sampling controls remain editable during chat; each test uses a one-shot override. */
    EnableWindow(g_temperature,!running || g_chat_mode);
    EnableWindow(g_top_k,!running || g_chat_mode);
    EnableWindow(g_top_p,!running || g_chat_mode);
    EnableWindow(g_repetition,!running || g_chat_mode);
    EnableWindow(g_max_tokens,!running || g_chat_mode);
    EnableWindow(g_chat_test,FALSE);
    SetWindowTextW(g_status,running?(g_chat_mode?L"Starting chat":L"Training running"):L"Idle");
    update_fresh_only_controls();
}

static BOOL append_argument(wchar_t* command, size_t capacity, const wchar_t* arg) {
    size_t used = wcslen(command);
    size_t len = wcslen(arg);
    if (used + len + 4 >= capacity) return FALSE;
    command[used++] = L' ';
    command[used++] = L'"';
    for (size_t i = 0; i < len; ++i) {
        if (arg[i] == L'"') return FALSE;
        command[used++] = arg[i];
    }
    command[used++] = L'"';
    command[used] = 0;
    return TRUE;
}

static BOOL add_corpus_path(const wchar_t* path) {
    int count = (int)SendMessageW(g_corpus, LB_GETCOUNT, 0, 0);
    if (count < 0) return FALSE;
    for (int i = 0; i < count; ++i) {
        wchar_t existing[MAX_PATH * 2];
        if (SendMessageW(g_corpus, LB_GETTEXT, i, (LPARAM)existing) >= 0 &&
            _wcsicmp(existing, path) == 0)
            return TRUE; /* Ignore duplicate selections. */
    }
    if (count >= 32) return FALSE;
    LRESULT result = SendMessageW(g_corpus, LB_ADDSTRING, 0, (LPARAM)path);
    return result != LB_ERR && result != LB_ERRSPACE;
}

static void add_corpus_files(void) {
    wchar_t file_buffer[32768] = { 0 };
    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_window;
    ofn.lpstrFilter = L"Text files (*.txt)\0*.txt\0All files (*.*)\0*.*\0";
    ofn.lpstrFile = file_buffer;
    ofn.nMaxFile = (DWORD)(sizeof(file_buffer) / sizeof(file_buffer[0]));
    ofn.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST |
        OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    if (!GetOpenFileNameW(&ofn)) return;

    BOOL limit_reached = FALSE;
    wchar_t* first = file_buffer;
    wchar_t* next = first + wcslen(first) + 1;
    if (!*next) {
        if (!add_corpus_path(first))
            limit_reached = TRUE;
    } else {
        while (*next) {
            wchar_t full[MAX_PATH * 2];
            _snwprintf_s(full, _countof(full), _TRUNCATE, L"%s\\%s", first, next);
            if (!add_corpus_path(full))
                limit_reached = TRUE;
            next += wcslen(next) + 1;
        }
    }
    if (limit_reached) {
        MessageBoxW(g_window,
            L"The corpus list supports up to 32 distinct files. Additional selections were not added.",
            L"ProjectMinima", MB_ICONWARNING);
    }
}

static BOOL append_option(wchar_t* command, size_t capacity,
    const wchar_t* name, HWND control) {
    wchar_t value[128];
    GetWindowTextW(control, value, _countof(value));
    if (!value[0]) return FALSE;
    size_t used=wcslen(command), nl=wcslen(name), vl=wcslen(value);
    if (used+nl+vl+3>=capacity) return FALSE;
    command[used++]=L' '; wcscpy_s(command+used,capacity-used,name); used+=nl;
    command[used++]=L' '; wcscpy_s(command+used,capacity-used,value);
    return TRUE;
}

static BOOL append_combo_option(wchar_t* command,size_t capacity,const wchar_t* name,HWND combo) {
    int index=(int)SendMessageW(combo,CB_GETCURSEL,0,0);
    if(index<0)return FALSE;
    wchar_t value[64];SendMessageW(combo,CB_GETLBTEXT,index,(LPARAM)value);
    size_t used=wcslen(command),nl=wcslen(name),vl=wcslen(value);
    if(used+nl+vl+3>=capacity)return FALSE;
    command[used++]=L' ';wcscpy_s(command+used,capacity-used,name);used+=nl;
    command[used++]=L' ';wcscpy_s(command+used,capacity-used,value);
    return TRUE;
}

static BOOL engine_sidecar_exists(const wchar_t* filename) {
    wchar_t path[MAX_PATH];
    wcscpy_s(path,_countof(path),g_engine_path);
    wchar_t* slash=wcsrchr(path,L'\\');
    if(!slash)return FALSE;
    wcscpy_s(slash+1,_countof(path)-(size_t)(slash+1-path),filename);
    DWORD attributes=GetFileAttributesW(path);
    return attributes!=INVALID_FILE_ATTRIBUTES && !(attributes&FILE_ATTRIBUTE_DIRECTORY);
}

static BOOL prebuilt_tokenizer_settings_match(void) {
    wchar_t vocab[64],bytes[64],sampling[64];
    GetWindowTextW(g_vocab,vocab,_countof(vocab));
    GetWindowTextW(g_tokenizer_bytes,bytes,_countof(bytes));
    int index=(int)SendMessageW(g_tokenizer_sampling,CB_GETCURSEL,0,0);
    if(index<0)return FALSE;
    SendMessageW(g_tokenizer_sampling,CB_GETLBTEXT,(WPARAM)index,(LPARAM)sampling);
    return engine_sidecar_exists(L"minima.bpe") &&
        wcscmp(vocab,g_prebuilt_vocab)==0 &&
        wcscmp(bytes,g_prebuilt_tokenizer_bytes)==0 &&
        wcscmp(sampling,g_prebuilt_sampling)==0;
}
static void remember_prebuilt_tokenizer_settings(void) {
    GetWindowTextW(g_vocab,g_prebuilt_vocab,_countof(g_prebuilt_vocab));
    GetWindowTextW(g_tokenizer_bytes,g_prebuilt_tokenizer_bytes,_countof(g_prebuilt_tokenizer_bytes));
    int index=(int)SendMessageW(g_tokenizer_sampling,CB_GETCURSEL,0,0);
    g_prebuilt_sampling[0]=0;
    if(index>=0)SendMessageW(g_tokenizer_sampling,CB_GETLBTEXT,(WPARAM)index,(LPARAM)g_prebuilt_sampling);
    WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"1",g_settings_path);
    WritePrivateProfileStringW(L"Training",L"prebuilt_vocab",g_prebuilt_vocab,g_settings_path);
    WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer_bytes",g_prebuilt_tokenizer_bytes,g_settings_path);
    WritePrivateProfileStringW(L"Training",L"prebuilt_sampling",g_prebuilt_sampling,g_settings_path);
}

static void start_engine(BOOL chat_mode, BOOL tokenizer_only) {
    if (InterlockedCompareExchange(&g_running, 0, 0)) return;
    if (!chat_mode && !tokenizer_only &&
        SendMessageW(g_resume, BM_GETCHECK, 0, 0) == BST_CHECKED &&
        g_use_saved_tokenizer) {
        MessageBoxW(g_window,
            L"The tokenizer was rebuilt after the saved checkpoint was created. Resume cannot use that checkpoint. Uncheck Resume to start a fresh model with the rebuilt tokenizer.",
            L"Checkpoint uses the previous tokenizer", MB_ICONWARNING);
        return;
    }
    BOOL use_prebuilt_tokenizer = (!chat_mode && !tokenizer_only &&
        SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED && g_use_saved_tokenizer);
    if(use_prebuilt_tokenizer && !prebuilt_tokenizer_settings_match()) {
        g_use_saved_tokenizer=FALSE; use_prebuilt_tokenizer=FALSE;
        WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"0",g_settings_path);
        append_log(L"Tokenizer settings changed since rebuild; the next fresh run will rebuild it again.\r\n");
    }
    if(chat_mode && g_use_saved_tokenizer) {
        MessageBoxW(g_window,L"The tokenizer was rebuilt, but the model checkpoint still uses the old tokenizer. Start a fresh training run before chatting.",L"ProjectMinima",MB_ICONWARNING);
        return;
    }
    if(chat_mode && (!engine_sidecar_exists(L"minima.model") ||
                     !engine_sidecar_exists(L"minima.bpe"))){
        MessageBoxW(g_window,L"No checkpoint/tokenizer found. Train a model first, then start chat.",
            L"ProjectMinima",MB_ICONWARNING);
        return;
    }

    wchar_t command[32768];
    if(chat_mode){
        _snwprintf_s(command,_countof(command),_TRUNCATE,L"\"%s\" --chat --resume",g_engine_path);
        if(!append_option(command,_countof(command),L"--temperature",g_temperature) ||
           !append_option(command,_countof(command),L"--top-k",g_top_k) ||
           !append_option(command,_countof(command),L"--top-p",g_top_p) ||
           !append_option(command,_countof(command),L"--repetition-penalty",g_repetition) ||
           !append_option(command,_countof(command),L"--max-new-tokens",g_max_tokens) ||
           !append_option(command,_countof(command),L"--vocab-size",g_vocab) ||
           !append_option(command,_countof(command),L"--dim",g_dim) ||
           !append_option(command,_countof(command),L"--layers",g_layers) ||
           !append_option(command,_countof(command),L"--heads",g_heads) ||
           !append_option(command,_countof(command),L"--ffn-dim",g_ffn) ||
                 !append_option(command,_countof(command),L"--max-seq",g_max_seq)){
            MessageBoxW(g_window,L"Check the generation settings.",L"ProjectMinima",MB_ICONWARNING);return;
        }
    } else if(tokenizer_only) {
        _snwprintf_s(command,_countof(command),_TRUNCATE,L"\"%s\" --retrain-tokenizer-only",g_engine_path);
        if(!append_option(command,_countof(command),L"--tokenizer-sample-bytes",g_tokenizer_bytes) ||
           !append_option(command,_countof(command),L"--vocab-size",g_vocab) ||
           !append_combo_option(command,_countof(command),L"--tokenizer-sampling",g_tokenizer_sampling)) {
            MessageBoxW(g_window,L"Check the tokenizer settings.",L"ProjectMinima",MB_ICONWARNING);return;
        }
    } else {
        wchar_t steps[32];GetWindowTextW(g_steps,steps,_countof(steps));
        wchar_t* end=NULL;long count=wcstol(steps,&end,10);
        if(!end||*end||count<1||count>2147483647L){MessageBoxW(g_window,L"Enter a positive training-step count.",L"ProjectMinima",MB_ICONWARNING);return;}
        _snwprintf_s(command,_countof(command),_TRUNCATE,L"\"%s\" --steps %ld",g_engine_path,count);
        if(SendMessageW(g_resume,BM_GETCHECK,0,0)==BST_CHECKED)wcscat_s(command,_countof(command),L" --resume");
        if(!append_option(command,_countof(command),L"--lr",g_lr) ||
           !append_option(command,_countof(command),L"--grad-clip",g_grad_clip) ||
           !append_option(command,_countof(command),L"--warmup",g_warmup) ||

           !append_option(command,_countof(command),L"--log-interval",g_log_interval) ||
           !append_option(command,_countof(command),L"--validation-batches",g_validation_batches) ||
           !append_option(command,_countof(command),L"--validation-interval",g_validation_interval) ||
           !append_option(command,_countof(command),L"--checkpoint-interval",g_checkpoint_interval) ||
           !append_option(command,_countof(command),L"--seed",g_seed) ||
           !append_option(command,_countof(command),L"--tokenizer-sample-bytes",g_tokenizer_bytes) ||
        !append_option(command,_countof(command),L"--vocab-size",g_vocab) ||
        !append_option(command,_countof(command),L"--dim",g_dim) ||
        !append_option(command,_countof(command),L"--layers",g_layers) ||
        !append_option(command,_countof(command),L"--heads",g_heads) ||
        !append_option(command,_countof(command),L"--ffn-dim",g_ffn) ||
        !append_option(command,_countof(command),L"--max-seq",g_max_seq) ||
           !append_option(command,_countof(command),L"--batch-size",g_batch_size) ||
           !append_option(command,_countof(command),L"--sequence-length",g_sequence_length) ||
           !append_combo_option(command,_countof(command),L"--lr-schedule",g_lr_schedule) ||
           !append_combo_option(command,_countof(command),L"--corpus-mix",g_corpus_mix) ||
           !append_combo_option(command,_countof(command),L"--tokenizer-sampling",g_tokenizer_sampling)){
            MessageBoxW(g_window,L"Check the training settings.",L"ProjectMinima",MB_ICONWARNING);return;
        }
        if(SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED &&
           !append_option(command,_countof(command),L"--total-steps",g_total_steps)){
            MessageBoxW(g_window,L"Check the total training-step horizon.",L"ProjectMinima",MB_ICONWARNING);return;
        }
        if(use_prebuilt_tokenizer)wcscat_s(command,_countof(command),L" --use-existing-tokenizer");
        wcscat_s(command,_countof(command),L" --metrics \"minima_metrics.csv\" --stop-file \"minima.stop\" --save-file \"minima.save\"");
        wchar_t stop_path[MAX_PATH];wcscpy_s(stop_path,_countof(stop_path),g_engine_path);
        wchar_t* stop_slash=wcsrchr(stop_path,L'\\');
        if(stop_slash)wcscpy_s(stop_slash+1,_countof(stop_path)-(size_t)(stop_slash+1-stop_path),L"minima.stop");
        DeleteFileW(stop_path);
        wchar_t save_path[MAX_PATH];wcscpy_s(save_path,_countof(save_path),g_engine_path);
        wchar_t* save_slash=wcsrchr(save_path,L'\\');
        if(save_slash)wcscpy_s(save_slash+1,_countof(save_path)-(size_t)(save_slash+1-save_path),L"minima.save");
        DeleteFileW(save_path);
        if(SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED)
            DeleteFileW(g_metrics_path); /* A fresh model starts a fresh loss graph. */
    }
    int corpus_count = chat_mode ? 0 : (int)SendMessageW(g_corpus, LB_GETCOUNT, 0, 0);
    for (int i = 0; i < corpus_count; ++i) {
        wchar_t path[MAX_PATH * 2];
        SendMessageW(g_corpus, LB_GETTEXT, i, (LPARAM)path);
        if (!append_argument(command, _countof(command), path)) {
            MessageBoxW(g_window, L"Corpus command line is too long or contains an unsupported quote.", L"ProjectMinima", MB_ICONERROR);
            return;
        }
    }

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE read_pipe=NULL,write_pipe=NULL,child_stdin=NULL;
    if(!CreatePipe(&read_pipe,&write_pipe,&sa,0)){
        MessageBoxW(g_window, L"Could not create the engine output pipe.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    SetHandleInformation(read_pipe,HANDLE_FLAG_INHERIT,0);
    if(chat_mode){
        if(!CreatePipe(&child_stdin,&g_stdin_write,&sa,0)){
            CloseHandle(read_pipe);CloseHandle(write_pipe);
            MessageBoxW(g_window,L"Could not create chat input pipe.",L"ProjectMinima",MB_ICONERROR);return;
        }
        SetHandleInformation(g_stdin_write,HANDLE_FLAG_INHERIT,0);
    }

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    HANDLE null_input=INVALID_HANDLE_VALUE;
    if(!chat_mode){
        null_input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
        if(null_input==INVALID_HANDLE_VALUE){
            CloseHandle(read_pipe);CloseHandle(write_pipe);
            MessageBoxW(g_window,L"Could not open NUL for engine standard input.",L"ProjectMinima",MB_ICONERROR);return;
        }
    }
    si.hStdInput = chat_mode ? child_stdin : null_input;

    wchar_t working_dir[MAX_PATH];
    wcscpy_s(working_dir, _countof(working_dir), g_engine_path);
    wchar_t* slash = wcsrchr(working_dir, L'\\');
    if (slash) *slash = 0;

    if (!CreateProcessW(g_engine_path, command, NULL, NULL, TRUE,
        CREATE_NO_WINDOW, NULL, working_dir, &si, &pi)) {
        if(null_input!=INVALID_HANDLE_VALUE)CloseHandle(null_input);
        CloseHandle(read_pipe);CloseHandle(write_pipe);
        if(child_stdin)CloseHandle(child_stdin);
        if(g_stdin_write){CloseHandle(g_stdin_write);g_stdin_write=NULL;}
        MessageBoxW(g_window,L"Could not start Minima.exe. Make sure it is beside MinimaGUI.exe.",L"ProjectMinima",MB_ICONERROR);
        return;
    }
    CloseHandle(write_pipe);
    if(null_input!=INVALID_HANDLE_VALUE)CloseHandle(null_input);
    if(child_stdin)CloseHandle(child_stdin);
    g_process = pi.hProcess;
    g_tokenizer_rebuild_mode=tokenizer_only;
    g_fresh_used_saved_tokenizer=use_prebuilt_tokenizer;
    CloseHandle(pi.hThread);
    g_chat_mode=chat_mode;
    set_running(TRUE);
    if(tokenizer_only)SetWindowTextW(g_status,L"Rebuilding tokenizer");
    EnableWindow(GetDlgItem(g_window,IDC_STOP),!chat_mode && !tokenizer_only);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_STOP),chat_mode);
    EnableWindow(GetDlgItem(g_window,IDC_SAVE_NOW),!chat_mode && !tokenizer_only);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_SEND),FALSE);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_STOP),chat_mode);
    append_log(chat_mode?L"\r\n--- Starting checkpoint chat ---\r\n":
        (tokenizer_only?L"\r\n--- Rebuilding tokenizer from selected corpus ---\r\n":L"\r\n--- Starting Minima training engine ---\r\n"));
    g_thread = minima_engine_start_reader(read_pipe, g_window, g_process,
        WM_APPEND_LOG, WM_ENGINE_EXIT, WM_CHAT_READY);
    if (!g_thread) {
        TerminateProcess(g_process, 1);
        if(g_stdin_write){CloseHandle(g_stdin_write);g_stdin_write=NULL;}
        CloseHandle(g_process);
        g_process = NULL;
        CloseHandle(read_pipe);
        g_chat_mode=FALSE;
        g_tokenizer_rebuild_mode=FALSE;
        g_fresh_used_saved_tokenizer=FALSE;
        set_running(FALSE);
        MessageBoxW(g_window, L"Could not create the output-reader thread.", L"ProjectMinima", MB_ICONERROR);
    }
}

static BOOL get_positive_control_value(HWND control, long* value) {
    wchar_t text[64],*end=NULL;
    GetWindowTextW(control,text,_countof(text));
    long parsed=wcstol(text,&end,10);
    if(!text[0] || !end || *end || parsed<1 || parsed>2147483647L)return FALSE;
    *value=parsed; return TRUE;
}
static BOOL add_file_size(const wchar_t* path, unsigned long long* total) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if(!GetFileAttributesExW(path,GetFileExInfoStandard,&data) ||
       (data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY))return FALSE;
    ULARGE_INTEGER size; size.HighPart=data.nFileSizeHigh; size.LowPart=data.nFileSizeLow;
    if(*total>ULLONG_MAX-size.QuadPart)return FALSE;
    *total+=size.QuadPart; return TRUE;
}
static void apply_auto_settings(void) {
    if(InterlockedCompareExchange(&g_running,0,0))return;
    if(SendMessageW(g_resume,BM_GETCHECK,0,0)==BST_CHECKED) {
        MessageBoxW(g_window,L"Automatic settings configure a fresh training schedule. Uncheck Resume first; an existing checkpoint's schedule cannot be changed automatically.",L"ProjectMinima",MB_ICONINFORMATION);
        return;
    }
    unsigned long long corpus_bytes=0;
    int count=(int)SendMessageW(g_corpus,LB_GETCOUNT,0,0);
    BOOL valid=TRUE;
    if(count>0) {
        for(int i=0;i<count;i++) {
            wchar_t path[MAX_PATH*2]; SendMessageW(g_corpus,LB_GETTEXT,(WPARAM)i,(LPARAM)path);
            if(!add_file_size(path,&corpus_bytes)){valid=FALSE;break;}
        }
    } else {
        DWORD a=GetFileAttributesW(L"..\\data\\books.txt");
        DWORD b=GetFileAttributesW(L"..\\data\\chatlogs.txt");
        BOOL found=FALSE;
        if(a!=INVALID_FILE_ATTRIBUTES && !(a&FILE_ATTRIBUTE_DIRECTORY)) {
            if(!add_file_size(L"..\\data\\books.txt",&corpus_bytes))valid=FALSE;
            found=TRUE;
        }
        if(valid && b!=INVALID_FILE_ATTRIBUTES && !(b&FILE_ATTRIBUTE_DIRECTORY)) {
            if(!add_file_size(L"..\\data\\chatlogs.txt",&corpus_bytes))valid=FALSE;
            found=TRUE;
        }
        if(!found) {
            DWORD root_a=GetFileAttributesW(L"data\\books.txt");
            DWORD root_b=GetFileAttributesW(L"data\\chatlogs.txt");
            if(root_a!=INVALID_FILE_ATTRIBUTES && !(root_a&FILE_ATTRIBUTE_DIRECTORY)) {
                if(!add_file_size(L"data\\books.txt",&corpus_bytes))valid=FALSE;
                found=TRUE;
            }
            if(valid && root_b!=INVALID_FILE_ATTRIBUTES && !(root_b&FILE_ATTRIBUTE_DIRECTORY)) {
                if(!add_file_size(L"data\\chatlogs.txt",&corpus_bytes))valid=FALSE;
                found=TRUE;
            }
        }
        if(!found)valid=FALSE;
    }
    if(!valid || corpus_bytes<4096) {
        MessageBoxW(g_window,L"Could not estimate the corpus size. Check that the selected text files exist and are accessible.",L"ProjectMinima",MB_ICONWARNING);
        return;
    }
    long batch=0,sequence=0;
    if(!get_positive_control_value(g_batch_size,&batch) || !get_positive_control_value(g_sequence_length,&sequence)) {
        MessageBoxW(g_window,L"Enter valid Batch and Seq values before applying automatic settings.",L"ProjectMinima",MB_ICONWARNING);
        return;
    }
    /* Current BPE experiments average about three source bytes per token. This is an estimate only. */
    long double estimated_train_tokens=((long double)corpus_bytes/3.0L)*0.95L;
    long double raw_steps=estimated_train_tokens/((long double)batch*(long double)sequence);
    if(raw_steps<1.0L)raw_steps=1.0L;
    if(raw_steps>2147483647.0L) {
        MessageBoxW(g_window,L"The estimated training schedule exceeds the supported step limit.",L"ProjectMinima",MB_ICONWARNING);
        return;
    }
    long steps=(long)raw_steps; if((long double)steps<raw_steps)steps++;
    long warmup=steps/100; if(warmup<10)warmup=10; if(warmup>500)warmup=500;
    long interval=steps/50;
    if(interval<1000)interval=1000;
    if(interval>10000)interval=10000;
    unsigned long long sample=corpus_bytes/20; if(sample<65536ULL)sample=65536ULL; if(sample>1048576ULL)sample=1048576ULL;
    wchar_t value[64];
    _snwprintf_s(value,_countof(value),_TRUNCATE,L"%ld",steps); SetWindowTextW(g_steps,value); SetWindowTextW(g_total_steps,value);
    _snwprintf_s(value,_countof(value),_TRUNCATE,L"%ld",warmup); SetWindowTextW(g_warmup,value);
    _snwprintf_s(value,_countof(value),_TRUNCATE,L"%ld",interval);
    SetWindowTextW(g_log_interval,value); SetWindowTextW(g_validation_interval,value); SetWindowTextW(g_checkpoint_interval,value);
    _snwprintf_s(value,_countof(value),_TRUNCATE,L"%llu",sample); SetWindowTextW(g_tokenizer_bytes,value);
    wchar_t message[320];
    _snwprintf_s(message,_countof(message),_TRUNCATE,L"Auto settings: %llu corpus bytes, approximately %.0f training tokens, %ld steps (about one token-pass). Batch, sequence length, and model size were left unchanged. Actual token counts may differ after BPE.",corpus_bytes,(double)estimated_train_tokens,steps);
    append_log(L"\r\n"); append_log(message); append_log(L"\r\n");
}
static void rebuild_tokenizer(void) {
    if(InterlockedCompareExchange(&g_running,0,0))return;
    if(SendMessageW(g_resume,BM_GETCHECK,0,0)==BST_CHECKED) {
        MessageBoxW(g_window,L"Uncheck Resume before rebuilding the tokenizer. A new tokenizer changes token IDs and makes the existing model checkpoint incompatible.",L"ProjectMinima",MB_ICONWARNING);
        return;
    }
    if(MessageBoxW(g_window,L"Rebuild minima.bpe using the current corpus and tokenizer settings?\r\n\r\nThe existing tokenizer will be backed up as minima.bpe.previous. The current model checkpoint will no longer match it. You must start a fresh training run afterward.",L"Rebuild tokenizer",MB_YESNO|MB_ICONWARNING)!=IDYES)return;
    g_use_saved_tokenizer=FALSE;
    WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"0",g_settings_path);
    wchar_t source[MAX_PATH],backup[MAX_PATH];
    wcscpy_s(source,_countof(source),g_engine_path);
    wchar_t* slash=wcsrchr(source,L'\\'); if(!slash)return;
    wcscpy_s(slash+1,_countof(source)-(size_t)(slash+1-source),L"minima.bpe");
    wcscpy_s(backup,_countof(backup),source);
    wchar_t* backup_name=wcsrchr(backup,L'\\'); if(!backup_name)return;
    wcscpy_s(backup_name+1,_countof(backup)-(size_t)(backup_name+1-backup),L"minima.bpe.previous");
    if(GetFileAttributesW(source)!=INVALID_FILE_ATTRIBUTES) {
        DeleteFileW(backup);
        if(!CopyFileW(source,backup,TRUE)) {
            MessageBoxW(g_window,L"Could not back up the existing tokenizer. Rebuild was cancelled to protect the current files.",L"ProjectMinima",MB_ICONERROR);
            return;
        }
        append_log(L"Previous tokenizer backed up to minima.bpe.previous.\r\n");
    }
    start_engine(FALSE,TRUE);
}

static BOOL request_cooperative_stop(void) {
    wchar_t path[MAX_PATH];
    wcscpy_s(path,_countof(path),g_engine_path);
    wchar_t* slash=wcsrchr(path,L'\\');
    if (slash) wcscpy_s(slash+1,_countof(path)-(size_t)(slash+1-path),L"minima.stop");
    HANDLE file=CreateFileW(path,GENERIC_WRITE,0,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
    if(file==INVALID_HANDLE_VALUE){append_log(L"Could not create stop signal file.\r\n");return FALSE;}
    CloseHandle(file);
    append_log(L"Stop requested. The engine will finish the current step and save a checkpoint.\r\n");
    return TRUE;
}

static void send_chat_message(void) {
    if(!g_stdin_write || !g_chat_mode)return;
    wchar_t wide[4096];GetWindowTextW(g_chat_input,wide,_countof(wide));
    if(!wide[0])return;
    int needed=WideCharToMultiByte(CP_UTF8,0,wide,-1,NULL,0,NULL,NULL);
    if(needed<=1)return;
    char* utf8=(char*)malloc((size_t)needed+2);if(!utf8)return;
    WideCharToMultiByte(CP_UTF8,0,wide,-1,utf8,needed,NULL,NULL);
    size_t len=strlen(utf8);utf8[len++]='\n';
    DWORD written=0;
    if(!WriteFile(g_stdin_write,utf8,(DWORD)len,&written,NULL) || written!=(DWORD)len){
        append_log(L"\r\nCould not send the complete chat message to the engine.\r\n");
        EnableWindow(GetDlgItem(g_window,IDC_CHAT_SEND),TRUE);
        free(utf8);return;
    }
    append_log(L"\r\nYou: ");append_log(wide);append_log(L"\r\n");
    free(utf8);SetWindowTextW(g_chat_input,L"");
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_SEND),FALSE);
    EnableWindow(g_chat_test,FALSE);
    SetWindowTextW(g_status,L"Waiting for Minima");
}

static BOOL read_float_control(HWND control, float minimum, float maximum, float* value) {
    wchar_t text[64], *end = NULL;
    GetWindowTextW(control, text, _countof(text));
    if (!text[0]) return FALSE;
    const float parsed = wcstof(text, &end);
    if (!end || *end || !isfinite((double)parsed) ||
        parsed < minimum || parsed > maximum) return FALSE;
    *value = parsed;
    return TRUE;
}

static BOOL read_integer_control(HWND control, long minimum, long maximum, long* value) {
    wchar_t text[64], *end = NULL;
    GetWindowTextW(control, text, _countof(text));
    if (!text[0]) return FALSE;
    const long parsed = wcstol(text, &end, 10);
    if (!end || *end || parsed < minimum || parsed > maximum) return FALSE;
    *value = parsed;
    return TRUE;
}

static void test_generation_settings(void) {
    if (!g_chat_mode || !g_stdin_write ||
        !InterlockedCompareExchange(&g_running, 0, 0)) return;

    wchar_t prompt[4096];
    GetWindowTextW(g_chat_input, prompt, _countof(prompt));
    if (!prompt[0]) {
        MessageBoxW(g_window, L"Enter a prompt or choose one from the prompt list first.",
            L"Generation Lab", MB_ICONINFORMATION);
        return;
    }

    float temperature, top_p, repetition;
    long top_k, max_tokens;
    if (!read_float_control(g_temperature, 0.05f, 5.0f, &temperature) ||
        !read_integer_control(g_top_k, 1, 256, &top_k) ||
        !read_float_control(g_top_p, 0.01f, 1.0f, &top_p) ||
        !read_float_control(g_repetition, 1.0f, 3.0f, &repetition) ||
        !read_integer_control(g_max_tokens, 1, 512, &max_tokens)) {
        MessageBoxW(g_window,
            L"Sampling settings are out of range. Temperature: 0.05-5.0, top-k: 1-256, top-p: 0.01-1.0, repetition: 1.0-3.0, max tokens: 1-512.",
            L"Generation Lab", MB_ICONWARNING);
        return;
    }

    wchar_t message[8192];
    const int chars = _snwprintf_s(message, _countof(message), _TRUNCATE,
        L"/test %.6g %ld %.6g %.6g %ld\n%s\n",
        (double)temperature, top_k, (double)top_p, (double)repetition,
        max_tokens, prompt);
    if (chars <= 0) {
        append_log(L"Generation test prompt is too long.\r\n");
        return;
    }

    const int bytes = WideCharToMultiByte(CP_UTF8, 0, message, chars,
        NULL, 0, NULL, NULL);
    if (bytes <= 0) {
        append_log(L"Could not encode the generation test prompt as UTF-8.\r\n");
        return;
    }
    char* utf8 = (char*)malloc((size_t)bytes);
    if (!utf8) return;
    WideCharToMultiByte(CP_UTF8, 0, message, chars, utf8, bytes, NULL, NULL);
    DWORD written = 0;
    const BOOL sent = WriteFile(g_stdin_write, utf8, (DWORD)bytes, &written, NULL) &&
        written == (DWORD)bytes;
    free(utf8);
    if (!sent) {
        append_log(L"Could not send the generation test to Minima.\r\n");
        return;
    }

    wchar_t header[256];
    _snwprintf_s(header, _countof(header), _TRUNCATE,
        L"\r\n--- Generation Lab | T=%.3g, top-k=%ld, top-p=%.3g, repetition=%.3g, max=%ld ---\r\nPrompt: ",
        (double)temperature, top_k, (double)top_p, (double)repetition, max_tokens);
    append_log(header);
    append_log(prompt);
    append_log(L"\r\n");
    SetWindowTextW(g_chat_input, L"");
    EnableWindow(GetDlgItem(g_window, IDC_CHAT_SEND), FALSE);
    EnableWindow(g_chat_test, FALSE);
    SetWindowTextW(g_status, L"Testing");
}

static void stop_generation(void) {
    if(g_process && g_chat_mode && InterlockedCompareExchange(&g_running,0,0)){
        append_log(L"\r\nStopping generation...\r\n");
        EnableWindow(g_chat_input,FALSE);
        EnableWindow(GetDlgItem(g_window,IDC_CHAT_SEND),FALSE);
        EnableWindow(g_chat_test,FALSE);
        EnableWindow(GetDlgItem(g_window,IDC_CHAT_STOP),FALSE);
        SetWindowTextW(g_status,L"Stopping chat");
        TerminateProcess(g_process,3);
    }
}

static void request_checkpoint_save(void) {
    if(!g_process || g_chat_mode)return;
    wchar_t path[MAX_PATH];wcscpy_s(path,_countof(path),g_engine_path);
    wchar_t* slash=wcsrchr(path,L'\\');
    if(slash)wcscpy_s(slash+1,_countof(path)-(size_t)(slash+1-path),L"minima.save");
    HANDLE file=CreateFileW(path,GENERIC_WRITE,0,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
    if(file!=INVALID_HANDLE_VALUE){CloseHandle(file);append_log(L"Checkpoint save requested.\r\n");}
    else append_log(L"Could not create checkpoint save signal file.\r\n");
}

static void stop_engine(void) {
    if (g_process && InterlockedCompareExchange(&g_running,0,0)) {
        if (MessageBoxW(g_window,L"Request a graceful stop after the current training step?",
            L"Stop training",MB_YESNO|MB_ICONQUESTION)==IDYES)
            request_cooperative_stop();
    }
}

static void initialize_settings_context(HWND hwnd) {
    ZeroMemory(&g_settings_context, sizeof(g_settings_context));
    g_settings_context.path = g_settings_path;
    g_settings_context.window = hwnd;
    g_settings_context.use_saved_tokenizer = &g_use_saved_tokenizer;
    g_settings_context.advanced_expanded = &g_advanced_expanded;
    g_settings_context.prebuilt_vocab = g_prebuilt_vocab;
    g_settings_context.prebuilt_tokenizer_bytes = g_prebuilt_tokenizer_bytes;
    g_settings_context.prebuilt_sampling = g_prebuilt_sampling;
    g_settings_context.set_advanced_expanded = set_advanced_expanded;
    g_settings_context.update_fresh_only_controls = update_fresh_only_controls;

    HWND edits[24] = {
        g_steps, g_vocab, g_dim, g_layers, g_heads, g_ffn, g_max_seq,
        g_batch_size, g_sequence_length, g_lr, g_warmup, g_total_steps,
        g_log_interval, g_validation_interval, g_validation_batches,
        g_checkpoint_interval, g_seed, g_tokenizer_bytes, g_temperature,
        g_top_k, g_top_p, g_repetition, g_max_tokens, g_grad_clip
    };
    HWND combos[3] = { g_lr_schedule, g_tokenizer_sampling, g_corpus_mix };
    CopyMemory(g_settings_context.edit_controls, edits, sizeof(edits));
    CopyMemory(g_settings_context.combo_controls, combos, sizeof(combos));
    g_settings_context.resume_control = g_resume;
    g_settings_context.corpus_control = g_corpus;
}

static void create_controls(HWND hwnd) {
    g_dpi = GetDpiForWindow(hwnd);
    if (!g_dpi) g_dpi = 96;
    rebuild_ui_font(hwnd);

    gui_pages_create_shell(hwnd, &g_main_tabs, &g_group_corpus,
        &g_group_model, &g_group_training, &g_group_advanced,
        &g_group_monitor, &g_group_output);
    HFONT font = g_ui_font ? g_ui_font : (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    GuiPageControls page_controls = {
        &g_corpus,
        &g_steps,
        &g_resume,
        &g_vocab,
        &g_dim,
        &g_layers,
        &g_heads,
        &g_ffn,
        &g_max_seq,
        &g_batch_size,
        &g_sequence_length,
        &g_lr,
        &g_warmup,
        &g_log_interval,
        &g_validation_interval,
        &g_validation_batches,
        &g_checkpoint_interval,
        &g_seed,
        &g_tokenizer_bytes,
        &g_temperature,
        &g_top_k,
        &g_top_p,
        &g_repetition,
        &g_max_tokens,
        &g_lr_schedule,
        &g_tokenizer_sampling,
        &g_corpus_mix,
        &g_grad_clip,
        &g_total_steps,
        &g_status,
        &g_graph,
        &g_output_label,
        &g_log,
        &g_chat_input,
        &g_chat_test,
    };
    gui_pages_create_content(hwnd, &page_controls, register_advanced_control);


    HWND controls[] = { g_corpus,g_steps,g_resume,g_log,g_lr,g_warmup,g_log_interval,
        g_validation_batches,g_validation_interval,g_checkpoint_interval,g_seed,g_tokenizer_bytes,g_graph,
        g_temperature,g_top_k,g_top_p,g_repetition,g_max_tokens,g_chat_input,
        g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,g_lr_schedule,g_tokenizer_sampling,g_corpus_mix,g_grad_clip,
        g_batch_size,g_sequence_length,g_total_steps };
    for (int i = 0; i < (int)_countof(controls); ++i)
        SendMessageW(controls[i], WM_SETFONT, (WPARAM)font, TRUE);
    EnumChildWindows(hwnd, set_child_font, (LPARAM)font);

    g_tooltip = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, NULL,
        WS_POPUP | TTS_ALWAYSTIP | TTS_NOPREFIX, CW_USEDEFAULT, CW_USEDEFAULT,
        CW_USEDEFAULT, CW_USEDEFAULT, hwnd, NULL, GetModuleHandleW(NULL), NULL);
    if (g_tooltip) {
        SendMessageW(g_tooltip, TTM_SETMAXTIPWIDTH, 0, scale_px(320));
        SendMessageW(g_tooltip, TTM_SETDELAYTIME, TTDT_INITIAL, 500);
        add_tooltip(g_steps, L"Number of additional optimizer updates in this run.");
        add_tooltip(g_total_steps, L"Global learning-rate schedule horizon for a fresh run. Resume restores the horizon stored in the checkpoint.");
        add_tooltip(g_log_interval, L"Optimizer steps between progress summaries. Long runs default to 1,000 steps; you can change this value at any time before training.");
        add_tooltip(g_batch_size, L"Number of independent training sequences in each update.");
        add_tooltip(g_sequence_length, L"Number of tokens processed per sequence. Must not exceed model context length.");
        add_tooltip(g_lr, L"Base learning rate. The cosine or constant schedule is applied to this value.");
        add_tooltip(g_warmup, L"Number of initial optimizer steps used for linear learning-rate warmup.");
        add_tooltip(g_tokenizer_bytes, L"Maximum sample bytes used for BPE training. The sample is split evenly across each selected source file without loading the full files.");
        add_tooltip(g_lr_schedule, L"Cosine decay reaches zero at the saved global schedule horizon. Constant keeps the base rate.");
        add_tooltip(g_tokenizer_sampling, L"Spread samples across each selected source file; prefix samples from the beginning of each source.");
        add_tooltip(g_corpus_mix, L"Choose how training examples are divided among selected files. Sqrt is a middle ground: large files still contribute more while smaller sources are no longer nearly ignored.");
        add_tooltip(g_grad_clip, L"Maximum global gradient length. 1.0 is a common stability setting; use 0 to disable clipping. The run log reports average and peak unclipped lengths and how often clipping was applied.");
        add_tooltip(GetDlgItem(hwnd, IDC_REBUILD_TOKENIZER), L"Rebuild minima.bpe without training model weights. The next fresh run can reuse it.");
        add_tooltip(GetDlgItem(hwnd, IDC_AUTO_SETTINGS), L"Estimate training and tokenizer settings from the selected corpus size.");
        add_tooltip(GetDlgItem(hwnd, IDC_BASIC_DEFAULTS), L"Restore the standard model and training settings without changing the selected corpus files.");
        add_tooltip(GetDlgItem(hwnd, IDC_SAVE_NOW), L"Request a checkpoint at the next optimizer update.");
        add_tooltip(g_chat_input, L"Type a message and press Enter to start chat and send it. While chatting, Enter sends each message. Use the dropdown for sample prompts.");
        add_tooltip(GetDlgItem(hwnd, IDC_CHAT_START), L"Start the checkpoint chat. You can also type a message below and press Enter.");
        add_tooltip(g_chat_test, L"Run one isolated generation with the current sampling settings. It does not change the active chat defaults or model checkpoint.");
    }

    gui_layout_scale_children(hwnd, 96, g_dpi);
    g_initial_rect_count=0;
    EnumChildWindows(hwnd,capture_initial_rect,0);
    gui_layout_layout_controls(hwnd);
    gui_layout_select_main_page(hwnd, 0);

    /* Make every interactive control reachable through normal keyboard tab navigation. */
    HWND tab_controls[] = {
        g_corpus, GetDlgItem(hwnd,IDC_ADD), GetDlgItem(hwnd,IDC_REMOVE),
        GetDlgItem(hwnd,IDC_BASIC_DEFAULTS),
        g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,g_steps,g_resume,
        g_batch_size,g_sequence_length,g_lr,g_warmup,g_log_interval,
        g_validation_interval,g_validation_batches,g_checkpoint_interval,
        g_seed,g_tokenizer_bytes,g_temperature,g_top_k,g_top_p,g_repetition,
        g_max_tokens,g_lr_schedule,g_tokenizer_sampling,g_corpus_mix,g_grad_clip,g_total_steps,
        GetDlgItem(hwnd,IDC_START),GetDlgItem(hwnd,IDC_STOP),
        GetDlgItem(hwnd,IDC_CHAT_START),g_chat_test,GetDlgItem(hwnd,IDC_SAVE_NOW),
        GetDlgItem(hwnd,IDC_REBUILD_TOKENIZER),GetDlgItem(hwnd,IDC_AUTO_SETTINGS),
        g_group_advanced,g_chat_input,GetDlgItem(hwnd,IDC_CHAT_SEND),GetDlgItem(hwnd,IDC_CHAT_STOP)
    };
    g_tab_control_count=(int)_countof(tab_controls);
    for(int i=0;i<g_tab_control_count;i++) {
        g_tab_controls[i]=tab_controls[i];
        LONG_PTR style=GetWindowLongPtrW(g_tab_controls[i],GWL_STYLE);
        SetWindowLongPtrW(g_tab_controls[i],GWL_STYLE,style|WS_TABSTOP);
    }

    initialize_settings_context(hwnd);
    minima_settings_load(&g_settings_context);
    gui_layout_select_main_page(hwnd, 0);
    wchar_t history_path[MAX_PATH];wcscpy_s(history_path,_countof(history_path),g_engine_path);
    wchar_t* history_slash=wcsrchr(history_path,L'\\');
    if(history_slash)wcscpy_s(history_slash+1,_countof(history_path)-(size_t)(history_slash+1-history_path),L"minima_chat.txt");
    FILE* history_file=_wfopen(history_path,L"rt");
    if(history_file){
        char line[65536];
        if(fseek(history_file,0,SEEK_END)==0){
            long end=ftell(history_file),start=end>1024*1024?end-1024*1024:0;
            if(start>0){fseek(history_file,start,SEEK_SET);(void)fgets(line,sizeof(line),history_file);}
            else rewind(history_file);
        }
        while(fgets(line,sizeof(line),history_file)){
            int count=MultiByteToWideChar(CP_UTF8,0,line,-1,NULL,0);
            if(count>0){wchar_t* wide=(wchar_t*)calloc((size_t)count,sizeof(wchar_t));
                if(wide){MultiByteToWideChar(CP_UTF8,0,line,-1,wide,count);append_log(wide);free(wide);}}
        }
        fclose(history_file);
    }
}

static void action_add_corpus(HWND hwnd) {
    (void)hwnd;
    g_use_saved_tokenizer=FALSE;
    WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"0",g_settings_path);
    add_corpus_files();
}
static void action_remove_corpus(HWND hwnd) {
    (void)hwnd;
    g_use_saved_tokenizer=FALSE;
    WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"0",g_settings_path);
    int selected[32];
    LRESULT n = SendMessageW(g_corpus, LB_GETSELITEMS, 32, (LPARAM)selected);
    for (int i = (int)n - 1; i >= 0; --i)
        SendMessageW(g_corpus, LB_DELETESTRING, selected[i], 0);
}
static void action_resume_changed(HWND hwnd) { (void)hwnd; update_fresh_only_controls(); }
static void action_start_training(HWND hwnd) { (void)hwnd; start_engine(FALSE,FALSE); }
static void action_stop_training(HWND hwnd) { (void)hwnd; stop_engine(); }
static void action_start_chat(HWND hwnd) { (void)hwnd; start_engine(TRUE,FALSE); }
static void action_rebuild_tokenizer(HWND hwnd) { (void)hwnd; rebuild_tokenizer(); }
static void action_auto_settings(HWND hwnd) { (void)hwnd; apply_auto_settings(); }
static void action_basic_defaults(HWND hwnd) {
    if (!InterlockedCompareExchange(&g_running,0,0) &&
        MessageBoxW(hwnd,L"Restore all model, training, tokenizer, and generation settings to their basic defaults? The selected corpus files will be kept.",
            L"Basic defaults",MB_YESNO|MB_ICONQUESTION)==IDYES) {
        minima_settings_reset_defaults(&g_settings_context);
        append_log(L"Basic defaults restored. Selected corpus files were left unchanged.\\r\\n");
    }
}
static void action_advanced_toggle(HWND hwnd) {
    set_advanced_expanded(hwnd,
        SendMessageW(g_group_advanced, BM_GETCHECK, 0, 0) == BST_CHECKED);
}
static void action_chat_send(HWND hwnd) { (void)hwnd; send_chat_message(); }
static void action_chat_test(HWND hwnd) { (void)hwnd; test_generation_settings(); }
static void action_chat_stop(HWND hwnd) { (void)hwnd; stop_generation(); }
static void action_save_checkpoint(HWND hwnd) { (void)hwnd; request_checkpoint_save(); }

static const GuiActionHandlers g_action_handlers = {
    action_add_corpus, action_remove_corpus, action_resume_changed,
    action_start_training, action_stop_training, action_start_chat,
    action_rebuild_tokenizer, action_auto_settings, action_basic_defaults,
    action_advanced_toggle, action_chat_send, action_chat_test,
    action_chat_stop, action_save_checkpoint
};

static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        g_window = hwnd;
        create_controls(hwnd);
        minima_graph_refresh();
        SetTimer(hwnd,IDT_GRAPH_REFRESH,1000,NULL);
        return 0;
    case WM_TIMER:
        if(wp==IDT_GRAPH_REFRESH && g_graph) {
            minima_graph_refresh();
            InvalidateRect(g_graph,NULL,FALSE);
        }
        return 0;
    case WM_GETMINMAXINFO: {
        MINMAXINFO* info=(MINMAXINFO*)lp;
        info->ptMinTrackSize.x=scale_px(960);
        info->ptMinTrackSize.y=scale_px(780);
        return 0;
    }
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) gui_layout_layout_controls(hwnd);
        return 0;
    case WM_DPICHANGED: {
        const UINT old_dpi = g_dpi ? g_dpi : 96;
        UINT new_dpi = HIWORD(wp);
        if (!new_dpi) new_dpi = old_dpi;
        RECT* suggested = (RECT*)lp;
        SetWindowPos(hwnd, NULL, suggested->left, suggested->top,
            suggested->right - suggested->left,
            suggested->bottom - suggested->top,
            SWP_NOZORDER | SWP_NOACTIVATE);
        gui_layout_scale_children(hwnd, old_dpi, new_dpi);
        g_dpi = new_dpi;
        minima_graph_set_dpi(g_dpi);
        rebuild_ui_font(hwnd);
        gui_layout_layout_controls(hwnd);
        minima_graph_refresh();
        if (g_graph) InvalidateRect(g_graph, NULL, FALSE);
        return 0;
    }
    case WM_NOTIFY:
        if (((NMHDR*)lp)->hwndFrom == g_main_tabs &&
            ((NMHDR*)lp)->code == TCN_SELCHANGE) {
            gui_layout_select_main_page(hwnd, TabCtrl_GetCurSel(g_main_tabs));
            return 0;
        }
        break;
    case WM_COMMAND:
        if (gui_actions_dispatch(hwnd, LOWORD(wp), &g_action_handlers))
            return 0;
        break;
    case WM_CHAT_READY:
        minima_gui_set_chat_send_ready(GetDlgItem(hwnd,IDC_CHAT_SEND),
            g_chat_mode, InterlockedCompareExchange(&g_running,0,0) != 0);
        EnableWindow(g_chat_test,
            g_chat_mode && InterlockedCompareExchange(&g_running,0,0) != 0);
        if (g_chat_mode) {
            SetWindowTextW(g_status, L"Chat ready");
            if (g_chat_send_when_ready &&
                IsWindowEnabled(GetDlgItem(hwnd, IDC_CHAT_SEND))) {
                g_chat_send_when_ready = FALSE;
                send_chat_message();
            }
        }
        return 0;
    case WM_APPEND_LOG:
        append_log((const wchar_t*)lp);
        free((void*)lp);
        return 0;
    case WM_ENGINE_EXIT:
        if(wp==0 && g_tokenizer_rebuild_mode) {
            g_use_saved_tokenizer=TRUE; remember_prebuilt_tokenizer_settings();
            SendMessageW(g_resume, BM_SETCHECK, BST_UNCHECKED, 0);
            update_fresh_only_controls();
            append_log(L"Tokenizer rebuilt successfully. Resume was turned off because the existing checkpoint uses the previous tokenizer. Start a fresh training run to use the rebuilt tokenizer; the GUI will reuse it if tokenizer settings and corpus selection remain unchanged.\r\n");
        }
        if(wp==0 && g_fresh_used_saved_tokenizer) {
            g_use_saved_tokenizer=FALSE;
            WritePrivateProfileStringW(L"Training",L"prebuilt_tokenizer",L"0",g_settings_path);
            append_log(L"Fresh model training completed using the prebuilt tokenizer.\r\n");
        }
        g_tokenizer_rebuild_mode=FALSE; g_fresh_used_saved_tokenizer=FALSE;
        if (g_process) { CloseHandle(g_process); g_process = NULL; }
        if (g_thread) { CloseHandle(g_thread); g_thread = NULL; }
        if (g_stdin_write) { CloseHandle(g_stdin_write); g_stdin_write=NULL; }
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_SEND),FALSE);
        EnableWindow(g_chat_test,FALSE);
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_STOP),FALSE);
        g_chat_send_when_ready=FALSE;
        g_chat_mode=FALSE;
        set_running(FALSE);
        append_log(wp == 0 ? L"\r\n--- Engine exited successfully ---\r\n" : L"\r\n--- Engine exited with an error ---\r\n");
        if (g_close_when_done) DestroyWindow(hwnd);
        return 0;
    case WM_CLOSE:
        if (g_process && InterlockedCompareExchange(&g_running,0,0)) {
            if (MessageBoxW(hwnd,L"An engine process is running. Stop it and close?",
                L"ProjectMinima",MB_YESNO|MB_ICONWARNING)!=IDYES) return 0;
            minima_settings_save(&g_settings_context);
            if(g_chat_mode){g_close_when_done=TRUE;TerminateProcess(g_process,3);}
            else {if(!request_cooperative_stop())return 0;g_close_when_done=TRUE;}
            return 0;
        }
        minima_settings_save(&g_settings_context);
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        KillTimer(hwnd, IDT_GRAPH_REFRESH);
        if (g_ui_font && g_ui_font != (HFONT)GetStockObject(DEFAULT_GUI_FONT)) {
            DeleteObject(g_ui_font);
            g_ui_font = NULL;
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command_line, int show) {
    (void)previous; (void)command_line;
    /* Per-Monitor V2 awareness is declared in the embedded application manifest. */
    g_dpi = GetDpiForSystem();
    if (!g_dpi) g_dpi = 96;
    INITCOMMONCONTROLSEX controls = { sizeof(controls), ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS | ICC_TAB_CLASSES };
    InitCommonControlsEx(&controls);

    DWORD n = GetModuleFileNameW(NULL, g_engine_path, _countof(g_engine_path));
    if (!n || n >= _countof(g_engine_path)) return 1;
    wchar_t* slash = wcsrchr(g_engine_path, L'\\');
    if (!slash) return 1;
    wcscpy_s(slash + 1, _countof(g_engine_path) - (size_t)(slash + 1 - g_engine_path), L"Minima.exe");
    wcscpy_s(g_settings_path,_countof(g_settings_path),g_engine_path);
    wchar_t* settings_slash=wcsrchr(g_settings_path,L'\\');
    if(settings_slash)wcscpy_s(settings_slash+1,_countof(g_settings_path)-(size_t)(settings_slash+1-g_settings_path),L"minima_gui.ini");

    WNDCLASSW gwc; ZeroMemory(&gwc,sizeof(gwc));
    gwc.lpfnWndProc=minima_graph_window_proc;gwc.hInstance=instance;gwc.lpszClassName=L"MinimaLossGraph";
    gwc.hCursor=LoadCursorW(NULL,IDC_ARROW);gwc.hbrBackground=(HBRUSH)(COLOR_WINDOW+1);
    if(!RegisterClassW(&gwc))return 1;
    DWORD plen=GetModuleFileNameW(NULL,g_metrics_path,_countof(g_metrics_path));
    if(!plen||plen>=_countof(g_metrics_path))return 1;
    wchar_t* pslash=wcsrchr(g_metrics_path,L'\\');
    if(pslash)wcscpy_s(pslash+1,_countof(g_metrics_path)-(size_t)(pslash+1-g_metrics_path),L"minima_metrics.csv");
    minima_graph_set_metrics_path(g_metrics_path);
    minima_graph_set_dpi(g_dpi);

    WNDCLASSW wc; ZeroMemory(&wc,sizeof(wc));
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"ProjectMinimaWin32";
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    if (!RegisterClassW(&wc)) return 1;

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"ProjectMinima",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        scale_px(960), scale_px(780), NULL, NULL, instance, NULL);
    if (!hwnd) return 1;
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN) {
            HWND focus = GetFocus();
            const BOOL chat_input_focused = focus == g_chat_input ||
                (focus && IsChild(g_chat_input, focus));
            const BOOL prompt_list_open = SendMessageW(g_chat_input,
                CB_GETDROPPEDSTATE, 0, 0) != 0;
            if (chat_input_focused && !prompt_list_open) {
                if (g_chat_mode) {
                    if (IsWindowEnabled(GetDlgItem(hwnd, IDC_CHAT_SEND)))
                        send_chat_message();
                    continue;
                }

                wchar_t draft[4096];
                GetWindowTextW(g_chat_input, draft, _countof(draft));
                if (draft[0] && !InterlockedCompareExchange(&g_running,0,0)) {
                    g_chat_send_when_ready = TRUE;
                    start_engine(TRUE, FALSE);
                    if (!g_chat_mode) g_chat_send_when_ready = FALSE;
                }
                continue;
            }
        }
        if(msg.message==WM_KEYDOWN && msg.wParam==VK_TAB &&
           GetAncestor(GetFocus(),GA_ROOT)==hwnd) {
            BOOL backwards=(GetKeyState(VK_SHIFT)&0x8000)!=0;
            int current=-1;
            for(int i=0;i<g_tab_control_count;i++)
                if(g_tab_controls[i]==GetFocus()){current=i;break;}
            for(int offset=1;offset<=g_tab_control_count;offset++) {
                int index=backwards
                    ? (current-offset+g_tab_control_count*2)%g_tab_control_count
                    : (current+offset)%g_tab_control_count;
                HWND target=g_tab_controls[index];
                if(IsWindowVisible(target) && IsWindowEnabled(target)) {
                    SetFocus(target);
                    break;
                }
            }
            continue;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}

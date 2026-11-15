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
#include <psapi.h>
#include <tlhelp32.h>
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
#pragma comment(lib, "Psapi.lib")
#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#define WM_APPEND_LOG    (WM_APP + 1)
#define WM_ENGINE_EXIT   (WM_APP + 2)
#define WM_CHAT_READY    (WM_APP + 3)
#define IDT_GRAPH_REFRESH 1

HWND g_window, g_corpus, g_log, g_chat_display, g_log_label;
HWND g_clear_chat, g_clear_logs, g_eval_button;
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
static HWND g_optimizer, g_weight_decay, g_threads;
HWND g_cpu_limit, g_ram_limit, g_arena_budget, g_cpu_warning, g_ram_warning, g_resource_profile, g_resource_status;
static wchar_t g_metrics_path[MAX_PATH];
static wchar_t g_settings_path[MAX_PATH];
static HANDLE g_process, g_thread, g_stdin_write, g_eval_job, g_resource_job;
static ULONGLONG g_last_cpu_100ns, g_last_cpu_tick;
static DWORD g_last_cpu_pid;
static double g_peak_cpu_percent;
static SIZE_T g_peak_working_bytes, g_peak_private_bytes;
static DWORD g_cpu_warning_limit, g_ram_warning_limit_gb;
static BOOL g_cpu_warning_issued, g_ram_warning_issued;
static BOOL g_applying_resource_profile;
static BOOL g_training_paused;
static BOOL g_chat_mode;
static BOOL g_evaluation_mode;
static BOOL g_chat_response_active;
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
    if (!g_log || !text) return;
    int end=GetWindowTextLengthW(g_log);
    SendMessageW(g_log,EM_SETSEL,(WPARAM)end,(LPARAM)end);
    SendMessageW(g_log,EM_REPLACESEL,FALSE,(LPARAM)text);
    SendMessageW(g_log,EM_SCROLLCARET,0,0);
}
static void append_chat_text(const wchar_t* text) {
    if (!g_chat_display || !text) return;
    int end=GetWindowTextLengthW(g_chat_display);
    SendMessageW(g_chat_display,EM_SETSEL,(WPARAM)end,(LPARAM)end);
    SendMessageW(g_chat_display,EM_REPLACESEL,FALSE,(LPARAM)text);
    SendMessageW(g_chat_display,EM_SCROLLCARET,0,0);
    save_chat_text(text);
}
static void clear_chat_history(void) {
    if (g_chat_mode && g_stdin_write) {
        static const char clear_command[] = "/clear\n";
        DWORD written = 0;
        (void)WriteFile(g_stdin_write, clear_command,
            (DWORD)(sizeof(clear_command) - 1), &written, NULL);
    }
    if (g_chat_display) SetWindowTextW(g_chat_display, L"");
    if (g_engine_path[0]) {
        wchar_t path[MAX_PATH];
        wcscpy_s(path, _countof(path), g_engine_path);
        wchar_t* slash = wcsrchr(path, L'\\');
        if (slash) wcscpy_s(slash + 1,
            _countof(path) - (size_t)(slash + 1 - path), L"minima_chat.txt");
        DeleteFileW(path);
    }
    append_log(L"Chat transcript cleared.\r\n");
}
static void clear_engine_logs(void) {
    if (g_log) SetWindowTextW(g_log, L"");
}

static BOOL read_bounded_integer(HWND control, DWORD maximum, DWORD* value_out) {
    wchar_t text[64], *end = NULL;
    GetWindowTextW(control, text, (int)_countof(text));
    if (!text[0]) return FALSE;
    unsigned long value = wcstoul(text, &end, 10);
    if (!end || *end || value > maximum) return FALSE;
    *value_out = (DWORD)value;
    return TRUE;
}

static BOOL read_resource_limits(DWORD* cpu_percent, DWORD* ram_gb) {
    return read_bounded_integer(g_cpu_limit, 100, cpu_percent) &&
        read_bounded_integer(g_ram_limit, 64, ram_gb);
}

static DWORD selected_resource_priority(void) {
    wchar_t profile[32];
    GetWindowTextW(g_resource_profile, profile, (int)_countof(profile));
    if (wcscmp(profile, L"Eco") == 0) return BELOW_NORMAL_PRIORITY_CLASS;
    if (wcscmp(profile, L"Performance") == 0 || wcscmp(profile, L"Training") == 0)
        return ABOVE_NORMAL_PRIORITY_CLASS;
    return NORMAL_PRIORITY_CLASS;
}

static void apply_resource_profile(void) {
    if (!g_resource_profile || g_applying_resource_profile) return;
    wchar_t profile[32];
    GetWindowTextW(g_resource_profile, profile, (int)_countof(profile));
    g_applying_resource_profile = TRUE;
    if (wcscmp(profile, L"Eco") == 0) {
        SetWindowTextW(g_cpu_limit, L"35");
        SetWindowTextW(g_ram_limit, L"4");
        SetWindowTextW(g_arena_budget, L"3");
        SetWindowTextW(g_threads, L"2");
    } else if (wcscmp(profile, L"Balanced") == 0) {
        SetWindowTextW(g_cpu_limit, L"65");
        SetWindowTextW(g_ram_limit, L"8");
        SetWindowTextW(g_arena_budget, L"6");
        SetWindowTextW(g_threads, L"4");
    } else if (wcscmp(profile, L"Performance") == 0) {
        SetWindowTextW(g_cpu_limit, L"100");
        SetWindowTextW(g_ram_limit, L"12");
        SetWindowTextW(g_arena_budget, L"10");
        SetWindowTextW(g_threads, L"16");
    } else if (wcscmp(profile, L"Training") == 0) {
        SetWindowTextW(g_cpu_limit, L"85");
        SetWindowTextW(g_ram_limit, L"10");
        SetWindowTextW(g_arena_budget, L"8");
        SetWindowTextW(g_threads, L"8");
    } else if (wcscmp(profile, L"Chat") == 0) {
        SetWindowTextW(g_cpu_limit, L"50");
        SetWindowTextW(g_ram_limit, L"4");
        SetWindowTextW(g_arena_budget, L"3");
        SetWindowTextW(g_threads, L"4");
    }
    g_applying_resource_profile = FALSE;
}

static BOOL configure_resource_job(HANDLE job, DWORD cpu_percent, DWORD ram_gb,
    DWORD priority_class) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
    ZeroMemory(&limits, sizeof(limits));
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
        JOB_OBJECT_LIMIT_PRIORITY_CLASS;
    limits.BasicLimitInformation.PriorityClass = priority_class;
    if (ram_gb > 0) {
        limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
        limits.JobMemoryLimit = (SIZE_T)ram_gb * (SIZE_T)1024 * 1024 * 1024;
    }
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
        &limits, sizeof(limits))) return FALSE;

    if (cpu_percent > 0) {
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION cpu;
        ZeroMemory(&cpu, sizeof(cpu));
        cpu.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE |
            JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
        cpu.CpuRate = cpu_percent * 100;
        if (!SetInformationJobObject(job, JobObjectCpuRateControlInformation,
            &cpu, sizeof(cpu))) return FALSE;
    }
    return TRUE;
}

static BOOL query_job_memory_usage(HANDLE job, SIZE_T* working_bytes,
    SIZE_T* private_bytes) {
    if (!job || !working_bytes || !private_bytes) return FALSE;
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return FALSE;

    SIZE_T working_total = 0, private_total = 0;
    BOOL found = FALSE;
    PROCESSENTRY32W entry;
    ZeroMemory(&entry, sizeof(entry));
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snapshot, &entry)) {
        do {
            HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                FALSE, entry.th32ProcessID);
            if (!process) continue;
            BOOL in_job = FALSE;
            if (IsProcessInJob(process, job, &in_job) && in_job) {
                PROCESS_MEMORY_COUNTERS_EX counters;
                ZeroMemory(&counters, sizeof(counters));
                counters.cb = sizeof(counters);
                if (GetProcessMemoryInfo(process,
                    (PROCESS_MEMORY_COUNTERS*)&counters, sizeof(counters))) {
                    working_total += counters.WorkingSetSize;
                    private_total += counters.PrivateUsage;
                    found = TRUE;
                }
            }
            CloseHandle(process);
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    if (!found) return FALSE;
    *working_bytes = working_total;
    *private_bytes = private_total;
    return TRUE;
}

static void reset_resource_peaks(void) {
    g_peak_cpu_percent = 0.0;
    g_peak_working_bytes = 0;
    g_peak_private_bytes = 0;
    g_cpu_warning_issued = FALSE;
    g_ram_warning_issued = FALSE;
}

static void update_resource_monitor(void) {
    if (!g_resource_status) return;
    if (!g_process || !InterlockedCompareExchange(&g_running, 0, 0)) {
        wchar_t idle_status[160];
        _snwprintf_s(idle_status, _countof(idle_status), _TRUNCATE,
            L"Last peak: CPU %.0f%% | RAM %.2f GB",
            g_peak_cpu_percent,
            (double)g_peak_working_bytes / (1024.0 * 1024.0 * 1024.0));
        SetWindowTextW(g_resource_status, idle_status);
        g_last_cpu_pid = 0;
        g_last_cpu_100ns = g_last_cpu_tick = 0;
        return;
    }

    PROCESS_MEMORY_COUNTERS_EX memory;
    ZeroMemory(&memory, sizeof(memory));
    memory.cb = sizeof(memory);
    double cpu_percent = 0.0;
    HANDLE active_job = g_resource_job ? g_resource_job : g_eval_job;
    ULONGLONG total = 0;
    BOOL have_cpu_time = FALSE;
    if (active_job) {
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting;
        ZeroMemory(&accounting, sizeof(accounting));
        if (QueryInformationJobObject(active_job, JobObjectBasicAccountingInformation,
            &accounting, sizeof(accounting), NULL)) {
            total = (ULONGLONG)accounting.TotalKernelTime.QuadPart +
                (ULONGLONG)accounting.TotalUserTime.QuadPart;
            have_cpu_time = TRUE;
        }
    }
    if (!have_cpu_time) {
        FILETIME created, exited, kernel, user;
        if (GetProcessTimes(g_process, &created, &exited, &kernel, &user)) {
            ULARGE_INTEGER k, u;
            k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
            u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
            total = k.QuadPart + u.QuadPart;
            have_cpu_time = TRUE;
        }
    }
    ULONGLONG now = GetTickCount64();
    DWORD pid = GetProcessId(g_process);
    if (have_cpu_time) {
        if (pid == g_last_cpu_pid && g_last_cpu_tick && now > g_last_cpu_tick &&
            total >= g_last_cpu_100ns) {
            SYSTEM_INFO si; GetSystemInfo(&si);
            DWORD cores = si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1;
            double elapsed = (double)(now - g_last_cpu_tick) / 1000.0;
            cpu_percent = ((double)(total - g_last_cpu_100ns) / 10000000.0) /
                (elapsed * (double)cores) * 100.0;
            if (cpu_percent > 100.0) cpu_percent = 100.0;
        }
        g_last_cpu_pid = pid;
        g_last_cpu_tick = now;
        g_last_cpu_100ns = total;
    }
    if (cpu_percent > g_peak_cpu_percent) g_peak_cpu_percent = cpu_percent;
    SIZE_T working_bytes = 0, private_bytes = 0;
    BOOL have_memory = active_job &&
        query_job_memory_usage(active_job, &working_bytes, &private_bytes);
    if (!have_memory && GetProcessMemoryInfo(g_process,
        (PROCESS_MEMORY_COUNTERS*)&memory, sizeof(memory))) {
        working_bytes = memory.WorkingSetSize;
        private_bytes = memory.PrivateUsage;
        have_memory = TRUE;
    }
    if (have_memory) {
        if (working_bytes > g_peak_working_bytes) g_peak_working_bytes = working_bytes;
        if (private_bytes > g_peak_private_bytes) g_peak_private_bytes = private_bytes;
        wchar_t status[256];
        double working = (double)working_bytes / (1024.0 * 1024.0 * 1024.0);
        double peak_working = (double)g_peak_working_bytes / (1024.0 * 1024.0 * 1024.0);
        double private_gb = (double)private_bytes / (1024.0 * 1024.0 * 1024.0);
        double peak_private = (double)g_peak_private_bytes / (1024.0 * 1024.0 * 1024.0);
        _snwprintf_s(status, _countof(status), _TRUNCATE,
            L"CPU %.0f/%.0f%% | WS %.2f/%.2f GB | Commit %.2f/%.2f GB",
            cpu_percent, g_peak_cpu_percent, working, peak_working,
            private_gb, peak_private);
        SetWindowTextW(g_resource_status, status);

        if (!g_cpu_warning_issued && g_cpu_warning_limit > 0 &&
            cpu_percent >= (double)g_cpu_warning_limit) {
            wchar_t warning[160];
            _snwprintf_s(warning, _countof(warning), _TRUNCATE,
                L"Resource warning: CPU utilization reached %.0f%% (configured threshold %lu%%).\r\n",
                cpu_percent, (unsigned long)g_cpu_warning_limit);
            append_log(warning);
            g_cpu_warning_issued = TRUE;
        }
        const SIZE_T ram_warning_bytes =
            (SIZE_T)g_ram_warning_limit_gb * (SIZE_T)1024 * 1024 * 1024;
        if (!g_ram_warning_issued && ram_warning_bytes > 0 &&
            working_bytes >= ram_warning_bytes) {
            wchar_t warning[160];
            _snwprintf_s(warning, _countof(warning), _TRUNCATE,
                L"Resource warning: process-group working set reached %.2f GiB (configured threshold %lu GiB).\r\n",
                working, (unsigned long)g_ram_warning_limit_gb);
            append_log(warning);
            g_ram_warning_issued = TRUE;
        }
    } else {
        SetWindowTextW(g_resource_status, L"CPU monitoring | memory unavailable");
    }
}

static void update_fresh_only_controls(void) {
    BOOL fresh=SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED;
    BOOL enabled=!InterlockedCompareExchange(&g_running,0,0) && fresh;
    EnableWindow(g_seed,enabled);
    EnableWindow(g_tokenizer_bytes,enabled);
    EnableWindow(g_tokenizer_sampling,enabled);
    EnableWindow(g_optimizer,enabled);
}

static BOOL engine_sidecar_path(const wchar_t* filename, wchar_t* path, size_t capacity) {
    if (!filename || !path || capacity == 0 || !g_engine_path[0]) return FALSE;
    if (wcscpy_s(path, capacity, g_engine_path) != 0) return FALSE;
    wchar_t* slash = wcsrchr(path, L'\\');
    if (!slash) return FALSE;
    return wcscpy_s(slash + 1, capacity - (size_t)(slash + 1 - path), filename) == 0;
}

static void save_configuration_profile(void) {
    wchar_t filename[MAX_PATH] = L"MinimaProfile.ini";
    static const wchar_t filter[] =
        L"Minima GUI profile (*.ini)\0*.ini\0All files (*.*)\0*.*\0";
    OPENFILENAMEW dialog;
    ZeroMemory(&dialog, sizeof(dialog));
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_window;
    dialog.lpstrFilter = filter;
    dialog.lpstrFile = filename;
    dialog.nMaxFile = (DWORD)_countof(filename);
    dialog.lpstrDefExt = L"ini";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!GetSaveFileNameW(&dialog)) return;

    minima_settings_save(&g_settings_context);
    if (_wcsicmp(filename, g_settings_path) != 0 &&
        !CopyFileW(g_settings_path, filename, FALSE)) {
        MessageBoxW(g_window, L"Could not export the current configuration profile.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    MessageBoxW(g_window, L"Configuration profile saved.", L"ProjectMinima", MB_ICONINFORMATION);
}

static void load_configuration_profile(void) {
    wchar_t filename[MAX_PATH] = L"";
    static const wchar_t filter[] =
        L"Minima GUI profile (*.ini)\0*.ini\0All files (*.*)\0*.*\0";
    OPENFILENAMEW dialog;
    ZeroMemory(&dialog, sizeof(dialog));
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_window;
    dialog.lpstrFilter = filter;
    dialog.lpstrFile = filename;
    dialog.nMaxFile = (DWORD)_countof(filename);
    dialog.lpstrDefExt = L"ini";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    if (!GetOpenFileNameW(&dialog)) return;

    if (_wcsicmp(filename, g_settings_path) != 0 &&
        !CopyFileW(filename, g_settings_path, FALSE)) {
        MessageBoxW(g_window, L"Could not import the selected configuration profile.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    minima_settings_load(&g_settings_context);
    apply_resource_profile();
    update_fresh_only_controls();
    MessageBoxW(g_window, L"Configuration profile loaded.", L"ProjectMinima", MB_ICONINFORMATION);
}

static void toggle_training_pause(void) {
    if (!InterlockedCompareExchange(&g_running, 0, 0) ||
        g_chat_mode || g_evaluation_mode || g_tokenizer_rebuild_mode) return;
    wchar_t path[MAX_PATH];
    if (!engine_sidecar_path(L"minima.pause", path, _countof(path))) {
        MessageBoxW(g_window, L"Could not locate the training pause signal file.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    if (!g_training_paused) {
        HANDLE file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file == INVALID_HANDLE_VALUE) {
            MessageBoxW(g_window, L"Could not create the training pause signal.", L"ProjectMinima", MB_ICONERROR);
            return;
        }
        CloseHandle(file);
        g_training_paused = TRUE;
        SetWindowTextW(GetDlgItem(g_window, IDC_PAUSE), L"Resume training");
        SetWindowTextW(g_status, L"Pausing after current step");
        append_log(L"Training pause requested; Minima will pause between optimizer steps.\r\n");
    } else {
        if (!DeleteFileW(path) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            MessageBoxW(g_window, L"Could not remove the training pause signal.", L"ProjectMinima", MB_ICONERROR);
            return;
        }
        g_training_paused = FALSE;
        SetWindowTextW(GetDlgItem(g_window, IDC_PAUSE), L"Pause training");
        SetWindowTextW(g_status, L"Resuming training");
        append_log(L"Training resume requested.\r\n");
    }
}

static void set_running(BOOL running) {
    InterlockedExchange(&g_running, running ? 1 : 0);
    EnableWindow(GetDlgItem(g_window, IDC_START), !running);
    EnableWindow(GetDlgItem(g_window, IDC_STOP), running);
    EnableWindow(GetDlgItem(g_window, IDC_SAVE_PROFILE), !running);
    EnableWindow(GetDlgItem(g_window, IDC_LOAD_PROFILE), !running);
    EnableWindow(GetDlgItem(g_window, IDC_PAUSE), running && !g_chat_mode && !g_evaluation_mode && !g_tokenizer_rebuild_mode);
    if (!running) {
        g_training_paused = FALSE;
        SetWindowTextW(GetDlgItem(g_window, IDC_PAUSE), L"Pause training");
        wchar_t pause_path[MAX_PATH];
        if (engine_sidecar_path(L"minima.pause", pause_path, _countof(pause_path)))
            DeleteFileW(pause_path);
    } else {
        SetWindowTextW(GetDlgItem(g_window, IDC_PAUSE), L"Pause training");
    }
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
        g_batch_size,g_sequence_length,g_lr_schedule,g_tokenizer_sampling,g_corpus_mix,g_grad_clip,
        g_optimizer,g_weight_decay,g_threads,g_cpu_limit,g_ram_limit,g_arena_budget,g_cpu_warning,g_ram_warning,g_resource_profile};
    for(int i=0;i<(int)_countof(fields);i++)if(fields[i])EnableWindow(fields[i],!running);
    EnableWindow(g_chat_input, !running || g_chat_mode);
    /* Sampling controls remain editable during chat; each test uses a one-shot override. */
    EnableWindow(g_temperature,!running || g_chat_mode);
    EnableWindow(g_top_k,!running || g_chat_mode);
    EnableWindow(g_top_p,!running || g_chat_mode);
    EnableWindow(g_repetition,!running || g_chat_mode);
    EnableWindow(g_max_tokens,!running || g_chat_mode);
    EnableWindow(g_chat_test,FALSE);
    EnableWindow(g_eval_button,!running);
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
           !append_option(command,_countof(command),L"--weight-decay",g_weight_decay) ||
           !append_option(command,_countof(command),L"--threads",g_threads) ||
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
           !append_combo_option(command,_countof(command),L"--tokenizer-sampling",g_tokenizer_sampling) ||
           (SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED &&
            !append_combo_option(command,_countof(command),L"--optimizer",g_optimizer))){
            MessageBoxW(g_window,L"Check the training settings.",L"ProjectMinima",MB_ICONWARNING);return;
        }
        if(SendMessageW(g_resume,BM_GETCHECK,0,0)!=BST_CHECKED &&
           !append_option(command,_countof(command),L"--total-steps",g_total_steps)){
            MessageBoxW(g_window,L"Check the total training-step horizon.",L"ProjectMinima",MB_ICONWARNING);return;
        }
        if(use_prebuilt_tokenizer)wcscat_s(command,_countof(command),L" --use-existing-tokenizer");
        wcscat_s(command,_countof(command),L" --metrics \"minima_metrics.csv\" --stop-file \"minima.stop\" --pause-file \"minima.pause\" --save-file \"minima.save\"");
        wchar_t stop_path[MAX_PATH];wcscpy_s(stop_path,_countof(stop_path),g_engine_path);
        wchar_t* stop_slash=wcsrchr(stop_path,L'\\');
        if(stop_slash)wcscpy_s(stop_slash+1,_countof(stop_path)-(size_t)(stop_slash+1-stop_path),L"minima.stop");
        DeleteFileW(stop_path);
        wchar_t pause_path[MAX_PATH];
        if (engine_sidecar_path(L"minima.pause", pause_path, _countof(pause_path))) DeleteFileW(pause_path);
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

    DWORD cpu_limit_percent = 0, ram_limit_gb = 0, arena_budget_gb = 0;
    DWORD cpu_warning_percent = 0, ram_warning_gb = 0;
    if (!read_resource_limits(&cpu_limit_percent, &ram_limit_gb) ||
        !read_bounded_integer(g_arena_budget, 64, &arena_budget_gb) ||
        !read_bounded_integer(g_cpu_warning, 100, &cpu_warning_percent) ||
        !read_bounded_integer(g_ram_warning, 64, &ram_warning_gb)) {
        MessageBoxW(g_window, L"CPU cap/warning must be 0-100%, process commit cap/arena budget/RAM warning must be 0-64 GB. Use 0 to disable each setting.", L"Resource limits", MB_ICONWARNING);
        return;
    }
    g_cpu_warning_limit = cpu_warning_percent;
    g_ram_warning_limit_gb = ram_warning_gb;
    if (arena_budget_gb > 0 &&
        !append_option(command, _countof(command), L"--memory-budget-gb", g_arena_budget)) {
        MessageBoxW(g_window, L"Could not add the arena memory budget to the process command.", L"Resource limits", MB_ICONERROR);
        return;
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

    g_resource_job = CreateJobObjectW(NULL, NULL);
    if (!g_resource_job || !configure_resource_job(g_resource_job,
        cpu_limit_percent, ram_limit_gb, selected_resource_priority())) {
        if (g_resource_job) { CloseHandle(g_resource_job); g_resource_job = NULL; }
        if(null_input!=INVALID_HANDLE_VALUE)CloseHandle(null_input);
        CloseHandle(read_pipe);CloseHandle(write_pipe);
        if(child_stdin)CloseHandle(child_stdin);
        if(g_stdin_write){CloseHandle(g_stdin_write);g_stdin_write=NULL;}
        MessageBoxW(g_window,L"Could not configure Windows resource limits. The engine was not started.",L"Resource limits",MB_ICONERROR);
        return;
    }

    if (!CreateProcessW(g_engine_path, command, NULL, NULL, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, working_dir, &si, &pi)) {
        if(null_input!=INVALID_HANDLE_VALUE)CloseHandle(null_input);
        CloseHandle(read_pipe);CloseHandle(write_pipe);
        if(child_stdin)CloseHandle(child_stdin);
        if(g_stdin_write){CloseHandle(g_stdin_write);g_stdin_write=NULL;}
        if(g_resource_job){CloseHandle(g_resource_job);g_resource_job=NULL;}
        MessageBoxW(g_window,L"Could not start Minima.exe. Make sure it is beside MinimaGUI.exe.",L"ProjectMinima",MB_ICONERROR);
        return;
    }
    CloseHandle(write_pipe);
    if(null_input!=INVALID_HANDLE_VALUE)CloseHandle(null_input);
    if(child_stdin)CloseHandle(child_stdin);
    if (!AssignProcessToJobObject(g_resource_job, pi.hProcess)) {
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        CloseHandle(read_pipe); CloseHandle(g_resource_job); g_resource_job = NULL;
        if (g_stdin_write) { CloseHandle(g_stdin_write); g_stdin_write = NULL; }
        MessageBoxW(g_window,L"Windows refused to apply the resource-control job. The engine was not started.",L"Resource limits",MB_ICONERROR);
        return;
    }
    if (ResumeThread(pi.hThread) == (DWORD)-1) {
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        CloseHandle(read_pipe); CloseHandle(g_resource_job); g_resource_job = NULL;
        if (g_stdin_write) { CloseHandle(g_stdin_write); g_stdin_write = NULL; }
        MessageBoxW(g_window,L"Could not resume Minima after applying resource limits.",L"ProjectMinima",MB_ICONERROR);
        return;
    }
    g_process = pi.hProcess;
    g_last_cpu_pid = 0; g_last_cpu_tick = g_last_cpu_100ns = 0;
    reset_resource_peaks();
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
        if (g_resource_job) { CloseHandle(g_resource_job); g_resource_job = NULL; }
        CloseHandle(read_pipe);
        g_chat_mode=FALSE;
        g_tokenizer_rebuild_mode=FALSE;
        g_fresh_used_saved_tokenizer=FALSE;
        set_running(FALSE);
        MessageBoxW(g_window, L"Could not create the output-reader thread.", L"ProjectMinima", MB_ICONERROR);
    }
}

static void start_logic_evaluation(void) {
    if (InterlockedCompareExchange(&g_running, 0, 0)) return;
    if (!engine_sidecar_exists(L"minima.model") || !engine_sidecar_exists(L"minima.bpe")) {
        MessageBoxW(g_window, L"Train a model and create minima.bpe before running the evaluation.", L"24-question evaluation", MB_ICONWARNING);
        return;
    }
    wchar_t engine_dir[MAX_PATH], root[MAX_PATH];
    wcscpy_s(engine_dir, _countof(engine_dir), g_engine_path);
    wchar_t* p = wcsrchr(engine_dir, L'\\'); if (!p) return; *p = 0;
    wcscpy_s(root, _countof(root), engine_dir);
    p = wcsrchr(root, L'\\'); if (!p) return; *p = 0;
    p = wcsrchr(root, L'\\'); if (!p) return; *p = 0;
    wchar_t script[MAX_PATH], dataset[MAX_PATH], output[MAX_PATH];
    _snwprintf_s(script, _countof(script), _TRUNCATE, L"%s\\tools\\evaluate_logic.py", root);
    _snwprintf_s(dataset, _countof(dataset), _TRUNCATE, L"%s\\data\\evaluation\\logic.jsonl", root);
    _snwprintf_s(output, _countof(output), _TRUNCATE, L"%s\\logic_results.jsonl", engine_dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES || GetFileAttributesW(dataset) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(g_window, L"Could not find the evaluation script or dataset. Keep the repository folders beside the x64 build folder.", L"24-question evaluation", MB_ICONERROR);
        return;
    }
    wchar_t command[8192] = L"py -3";
    if (!append_argument(command, _countof(command), script) ||
        !append_argument(command, _countof(command), L"--exe") ||
        !append_argument(command, _countof(command), g_engine_path) ||
        !append_argument(command, _countof(command), L"--workdir") ||
        !append_argument(command, _countof(command), engine_dir) ||
        !append_argument(command, _countof(command), L"--dataset") ||
        !append_argument(command, _countof(command), dataset) ||
        !append_argument(command, _countof(command), L"--output") ||
        !append_argument(command, _countof(command), output) ||
        !append_option(command, _countof(command), L"--vocab-size", g_vocab) ||
        !append_option(command, _countof(command), L"--dim", g_dim) ||
        !append_option(command, _countof(command), L"--layers", g_layers) ||
        !append_option(command, _countof(command), L"--heads", g_heads) ||
        !append_option(command, _countof(command), L"--ffn-dim", g_ffn) ||
        !append_option(command, _countof(command), L"--max-seq", g_max_seq)) {
        MessageBoxW(g_window, L"Check the model settings and evaluation command length.", L"24-question evaluation", MB_ICONWARNING);
        return;
    }
    wcscat_s(command, _countof(command), L" --max-new-tokens 64");
    DWORD cpu_limit_percent = 0, ram_limit_gb = 0, arena_budget_gb = 0;
    DWORD cpu_warning_percent = 0, ram_warning_gb = 0;
    if (!read_resource_limits(&cpu_limit_percent, &ram_limit_gb) ||
        !read_bounded_integer(g_arena_budget, 64, &arena_budget_gb) ||
        !read_bounded_integer(g_cpu_warning, 100, &cpu_warning_percent) ||
        !read_bounded_integer(g_ram_warning, 64, &ram_warning_gb)) {
        MessageBoxW(g_window, L"CPU cap/warning must be 0-100%, process commit cap/arena budget/RAM warning must be 0-64 GB. Use 0 to disable each setting.", L"Resource limits", MB_ICONWARNING);
        return;
    }
    g_cpu_warning_limit = cpu_warning_percent;
    g_ram_warning_limit_gb = ram_warning_gb;
    if (arena_budget_gb > 0 &&
        !append_option(command, _countof(command), L"--memory-budget-gb", g_arena_budget)) {
        MessageBoxW(g_window, L"Could not add the arena memory budget to the process command.", L"Resource limits", MB_ICONERROR);
        return;
    }
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { MessageBoxW(g_window, L"Could not create evaluation output pipe.", L"ProjectMinima", MB_ICONERROR); return; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (nul == INVALID_HANDLE_VALUE) { CloseHandle(rd); CloseHandle(wr); MessageBoxW(g_window, L"Could not open NUL.", L"ProjectMinima", MB_ICONERROR); return; }
    HANDLE eval_job = CreateJobObjectW(NULL, NULL);
    if (!eval_job) { CloseHandle(nul); CloseHandle(rd); CloseHandle(wr); MessageBoxW(g_window, L"Could not create an evaluation process group.", L"ProjectMinima", MB_ICONERROR); return; }
    if (!configure_resource_job(eval_job, cpu_limit_percent, ram_limit_gb, selected_resource_priority())) {
        CloseHandle(eval_job); CloseHandle(nul); CloseHandle(rd); CloseHandle(wr);
        MessageBoxW(g_window, L"Could not configure evaluation process cleanup.", L"ProjectMinima", MB_ICONERROR); return;
    }
    STARTUPINFOW si; PROCESS_INFORMATION pi; ZeroMemory(&si, sizeof(si)); ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si); si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    si.hStdInput = nul; si.hStdOutput = wr; si.hStdError = wr;
    if (!CreateProcessW(NULL, command, NULL, NULL, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, root, &si, &pi)) {
        CloseHandle(eval_job); CloseHandle(nul); CloseHandle(rd); CloseHandle(wr);
        MessageBoxW(g_window, L"Could not launch Python. Install Python 3 and ensure the 'py' launcher is available.", L"24-question evaluation", MB_ICONERROR);
        return;
    }
    CloseHandle(nul); CloseHandle(wr);
    if (!AssignProcessToJobObject(eval_job, pi.hProcess)) {
        TerminateProcess(pi.hProcess, 1); CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        CloseHandle(eval_job); CloseHandle(rd);
        MessageBoxW(g_window, L"Could not attach the evaluation process to its cleanup group.", L"ProjectMinima", MB_ICONERROR);
        return;
    }
    ResumeThread(pi.hThread); CloseHandle(pi.hThread);
    g_eval_job = eval_job; g_process = pi.hProcess;
    g_last_cpu_pid = 0; g_last_cpu_tick = g_last_cpu_100ns = 0;
    reset_resource_peaks();
    update_resource_monitor();
    g_chat_mode = FALSE; g_evaluation_mode = TRUE;
    gui_layout_select_main_page(g_window, 2); set_running(TRUE);
    EnableWindow(GetDlgItem(g_window, IDC_STOP), FALSE);
    EnableWindow(GetDlgItem(g_window, IDC_CHAT_STOP), TRUE);
    SetWindowTextW(GetDlgItem(g_window, IDC_CHAT_STOP), L"Stop evaluation");
    SetWindowTextW(g_status, L"Evaluation running");
    append_log(L"\r\n--- Starting 24-question logic evaluation (greedy decoding) ---\r\n");
    g_thread = minima_engine_start_reader(rd, g_window, g_process, WM_APPEND_LOG, WM_ENGINE_EXIT, WM_CHAT_READY);
    if (!g_thread) {
        if (g_eval_job) { CloseHandle(g_eval_job); g_eval_job = NULL; }
        else TerminateProcess(g_process, 1);
        CloseHandle(g_process); g_process = NULL; CloseHandle(rd);
        g_evaluation_mode = FALSE; set_running(FALSE);
        MessageBoxW(g_window, L"Could not create evaluation output-reader thread.", L"ProjectMinima", MB_ICONERROR);
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
    append_chat_text(L"\r\nYou: ");append_chat_text(wide);append_chat_text(L"\r\n");
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

    HWND edits[31] = {
        g_steps, g_vocab, g_dim, g_layers, g_heads, g_ffn, g_max_seq,
        g_batch_size, g_sequence_length, g_lr, g_warmup, g_total_steps,
        g_log_interval, g_validation_interval, g_validation_batches,
        g_checkpoint_interval, g_seed, g_tokenizer_bytes, g_temperature,
        g_top_k, g_top_p, g_repetition, g_max_tokens, g_grad_clip,
        g_weight_decay, g_threads, g_cpu_limit, g_ram_limit, g_arena_budget,
        g_cpu_warning, g_ram_warning
    };
    HWND combos[5] = { g_lr_schedule, g_tokenizer_sampling, g_corpus_mix, g_optimizer, g_resource_profile };
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
        &g_chat_display,
        &g_log_label,
        &g_clear_chat,
        &g_clear_logs,
        &g_chat_input,
        &g_chat_test,
        &g_optimizer,
        &g_weight_decay,
        &g_threads,
        &g_arena_budget,
        &g_cpu_warning,
        &g_ram_warning
    };
    gui_pages_create_content(hwnd, &page_controls, register_advanced_control);
    CreateWindowW(L"BUTTON", L"Pause training", WS_CHILD | WS_VISIBLE | WS_DISABLED,
        142, 340, 84, 30, hwnd, (HMENU)(INT_PTR)IDC_PAUSE, NULL, NULL);
    g_resource_status = CreateWindowW(L"STATIC", L"CPU: --   RAM: --", WS_CHILD | WS_VISIBLE,
        600, 430, 300, 20, hwnd, NULL, NULL, NULL);
    CreateWindowW(L"STATIC", L"CPU cap %", WS_CHILD | WS_VISIBLE,
        600, 455, 76, 18, hwnd, (HMENU)(INT_PTR)IDC_CPU_LIMIT_LABEL, NULL, NULL);
    g_cpu_limit = CreateWindowW(L"EDIT", L"0", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        680, 452, 48, 23, hwnd, (HMENU)(INT_PTR)IDC_CPU_LIMIT, NULL, NULL);
    CreateWindowW(L"STATIC", L"Commit cap GB", WS_CHILD | WS_VISIBLE,
        750, 455, 88, 18, hwnd, (HMENU)(INT_PTR)IDC_RAM_LIMIT_LABEL, NULL, NULL);
    g_ram_limit = CreateWindowW(L"EDIT", L"0", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        842, 452, 48, 23, hwnd, (HMENU)(INT_PTR)IDC_RAM_LIMIT, NULL, NULL);
    CreateWindowW(L"STATIC", L"Arena budget GB", WS_CHILD | WS_VISIBLE,
        600, 482, 104, 18, hwnd, (HMENU)(INT_PTR)IDC_ARENA_BUDGET_LABEL, NULL, NULL);
    g_arena_budget = CreateWindowW(L"EDIT", L"0", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        708, 479, 48, 23, hwnd, (HMENU)(INT_PTR)IDC_ARENA_BUDGET, NULL, NULL);
    CreateWindowW(L"STATIC", L"Warn CPU %", WS_CHILD | WS_VISIBLE,
        600, 512, 78, 18, hwnd, (HMENU)(INT_PTR)IDC_CPU_WARNING_LABEL, NULL, NULL);
    g_cpu_warning = CreateWindowW(L"EDIT", L"0", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        680, 509, 48, 23, hwnd, (HMENU)(INT_PTR)IDC_CPU_WARNING, NULL, NULL);
    CreateWindowW(L"STATIC", L"Warn RAM GB", WS_CHILD | WS_VISIBLE,
        750, 512, 88, 18, hwnd, (HMENU)(INT_PTR)IDC_RAM_WARNING_LABEL, NULL, NULL);
    g_ram_warning = CreateWindowW(L"EDIT", L"0", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER,
        842, 509, 48, 23, hwnd, (HMENU)(INT_PTR)IDC_RAM_WARNING, NULL, NULL);
    CreateWindowW(L"STATIC", L"Profile", WS_CHILD | WS_VISIBLE,
        600, 542, 76, 18, hwnd, (HMENU)(INT_PTR)IDC_RESOURCE_PROFILE_LABEL, NULL, NULL);
    g_resource_profile = CreateWindowW(L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | CBS_DROPDOWNLIST | WS_VSCROLL,
        680, 539, 140, 100, hwnd, (HMENU)(INT_PTR)IDC_RESOURCE_PROFILE, NULL, NULL);
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Custom");
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Eco");
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Balanced");
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Performance");
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Training");
    SendMessageW(g_resource_profile, CB_ADDSTRING, 0, (LPARAM)L"Chat");
    SendMessageW(g_resource_profile, CB_SETCURSEL, 0, 0);
    CreateWindowW(L"BUTTON", L"Save profile", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
        600, 570, 112, 24, hwnd, (HMENU)(INT_PTR)IDC_SAVE_PROFILE, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Load profile", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
        720, 570, 112, 24, hwnd, (HMENU)(INT_PTR)IDC_LOAD_PROFILE, NULL, NULL);
    g_eval_button = CreateWindowW(L"BUTTON", L"Run 24-question test", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
        12, 665, 150, 26, hwnd, (HMENU)(INT_PTR)IDC_RUN_EVAL, NULL, NULL);

    HWND controls[] = { g_corpus,g_steps,g_resume,g_log,g_chat_display,g_clear_chat,g_clear_logs,g_eval_button,g_log_label,g_lr,g_warmup,g_log_interval,
        g_validation_batches,g_validation_interval,g_checkpoint_interval,g_seed,g_tokenizer_bytes,g_graph,
        g_temperature,g_top_k,g_top_p,g_repetition,g_max_tokens,g_chat_input,
        g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,g_lr_schedule,g_tokenizer_sampling,g_corpus_mix,g_grad_clip,
        g_batch_size,g_sequence_length,g_total_steps,g_optimizer,g_weight_decay,g_threads,g_cpu_limit,g_ram_limit,g_arena_budget,g_cpu_warning,g_ram_warning,g_resource_profile,g_resource_status,GetDlgItem(hwnd,IDC_PAUSE),GetDlgItem(hwnd,IDC_SAVE_PROFILE),GetDlgItem(hwnd,IDC_LOAD_PROFILE) };
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
        add_tooltip(g_optimizer, L"Fresh-run optimizer. Resume automatically uses the optimizer stored in the checkpoint. Lion uses engine defaults of learning rate 0.00003 and weight decay 0.1; adjust those fields when selecting Lion.");
        add_tooltip(g_weight_decay, L"AdamW/Lion weight decay, from 0 to 1. The GUI passes this value explicitly, including when resuming.");
        add_tooltip(g_threads, L"OpenMP worker threads. 0 lets the runtime choose its default; 1-256 explicitly sets the worker count.");
        add_tooltip(g_cpu_limit, L"Windows Job Object hard CPU cap. 0 disables the cap; 1-100 limits total CPU capacity across all logical processors, not the percentage of one core.");
        add_tooltip(g_ram_limit, L"Windows Job Object hard committed-memory limit for the Minima process group. This is a commit ceiling, not a physical working-set cap. 0 disables it; 1-64 sets GiB. Too low can make allocations fail.");
        add_tooltip(g_arena_budget, L"Engine-level shared budget for pages committed by Minima's permanent, temporary, and corpus arenas. 0 disables it; 1-64 sets GiB. This budget rejects arena allocations cleanly before Windows runs out of commit.");
        add_tooltip(g_cpu_warning, L"Log one warning per run when measured process-group CPU utilization reaches this percentage. 0 disables the warning; 1-100 is normalized to total logical-processor capacity.");
        add_tooltip(g_ram_warning, L"Log one warning per run when the process-group working set reaches this GiB threshold. 0 disables the warning; 1-64 is allowed.");
        add_tooltip(g_resource_profile, L"Eco, Balanced, and Performance presets set CPU cap, process commit cap, arena budget, OpenMP threads, and Windows process priority. Any manual edit to those controls switches the selection back to Custom.");
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
        g_optimizer,g_weight_decay,g_threads,g_cpu_limit,g_ram_limit,g_arena_budget,g_cpu_warning,g_ram_warning,g_resource_profile,
        GetDlgItem(hwnd,IDC_START),GetDlgItem(hwnd,IDC_STOP),
        GetDlgItem(hwnd,IDC_CHAT_START),GetDlgItem(hwnd,IDC_PAUSE),GetDlgItem(hwnd,IDC_SAVE_PROFILE),GetDlgItem(hwnd,IDC_LOAD_PROFILE),g_chat_test,GetDlgItem(hwnd,IDC_SAVE_NOW),
        GetDlgItem(hwnd,IDC_REBUILD_TOKENIZER),GetDlgItem(hwnd,IDC_AUTO_SETTINGS),
        g_group_advanced,g_chat_input,GetDlgItem(hwnd,IDC_CHAT_SEND),GetDlgItem(hwnd,IDC_CHAT_STOP),g_eval_button
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
                if(wide){MultiByteToWideChar(CP_UTF8,0,line,-1,wide,count);append_chat_text(wide);free(wide);}}
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
static void action_chat_stop(HWND hwnd) { (void)hwnd; if (g_evaluation_mode && g_process) {
        append_log(L"\r\nEvaluation stop requested.\r\n");
        if (g_eval_job) { CloseHandle(g_eval_job); g_eval_job = NULL; }
        else TerminateProcess(g_process, 3);
    } else stop_generation(); }
static void action_save_checkpoint(HWND hwnd) { (void)hwnd; request_checkpoint_save(); }
static void action_clear_chat(HWND hwnd) { (void)hwnd; clear_chat_history(); }
static void action_clear_logs(HWND hwnd) { (void)hwnd; clear_engine_logs(); }
static void action_run_evaluation(HWND hwnd) { (void)hwnd; start_logic_evaluation(); }

static const GuiActionHandlers g_action_handlers = {
    action_add_corpus, action_remove_corpus, action_resume_changed,
    action_start_training, action_stop_training, action_start_chat,
    action_rebuild_tokenizer, action_auto_settings, action_basic_defaults,
    action_advanced_toggle, action_chat_send, action_chat_test,
    action_chat_stop, action_save_checkpoint, action_clear_chat, action_clear_logs, action_run_evaluation
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
        if(wp==IDT_GRAPH_REFRESH) {
            if (g_graph) { minima_graph_refresh(); InvalidateRect(g_graph,NULL,FALSE); }
            update_resource_monitor();
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
        if (LOWORD(wp) == IDC_SAVE_PROFILE && HIWORD(wp) == BN_CLICKED) {
            save_configuration_profile();
            return 0;
        }
        if (LOWORD(wp) == IDC_LOAD_PROFILE && HIWORD(wp) == BN_CLICKED) {
            load_configuration_profile();
            return 0;
        }
        if (LOWORD(wp) == IDC_PAUSE && HIWORD(wp) == BN_CLICKED) {
            toggle_training_pause();
            return 0;
        }
        if (LOWORD(wp) == IDC_RESOURCE_PROFILE && HIWORD(wp) == CBN_SELCHANGE) {
            apply_resource_profile();
            return 0;
        }
        if (HIWORD(wp) == EN_CHANGE && !g_applying_resource_profile &&
            ((HWND)lp == g_cpu_limit || (HWND)lp == g_ram_limit ||
             (HWND)lp == g_arena_budget || (HWND)lp == g_threads)) {
            LRESULT custom = SendMessageW(g_resource_profile, CB_FINDSTRINGEXACT,
                (WPARAM)-1, (LPARAM)L"Custom");
            if (custom >= 0) SendMessageW(g_resource_profile, CB_SETCURSEL, (WPARAM)custom, 0);
        }
        if (gui_actions_dispatch(hwnd, LOWORD(wp), &g_action_handlers))
            return 0;
        break;
    case WM_CHAT_READY:
        g_chat_response_active = FALSE;
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
    case WM_APPEND_LOG: {
        const wchar_t* line = (const wchar_t*)lp;
        append_log(line);
        if (line && wcsstr(line, L"TRAIN_PAUSED")) {
            g_training_paused = TRUE;
            SetWindowTextW(GetDlgItem(hwnd, IDC_PAUSE), L"Resume training");
            SetWindowTextW(g_status, L"Training paused");
        } else if (line && wcsstr(line, L"TRAIN_RESUMED")) {
            g_training_paused = FALSE;
            SetWindowTextW(GetDlgItem(hwnd, IDC_PAUSE), L"Pause training");
            SetWindowTextW(g_status, L"Training running");
        }
        if (g_chat_mode && line) {
            if (wcsncmp(line, L"BOT: ", 5) == 0) {
                g_chat_response_active = TRUE;
                append_chat_text(L"Minima: ");
                append_chat_text(line + 5);
            } else if (g_chat_response_active) {
                append_chat_text(line);
            }
        }
        free((void*)lp);
        return 0;
    }
    case WM_ENGINE_EXIT: {
        const BOOL was_evaluation = g_evaluation_mode;
        g_evaluation_mode = FALSE;
        SetWindowTextW(GetDlgItem(hwnd, IDC_CHAT_STOP), L"Stop generation");
        update_resource_monitor();
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
        if (g_eval_job) { CloseHandle(g_eval_job); g_eval_job = NULL; }
        if (g_resource_job) { CloseHandle(g_resource_job); g_resource_job = NULL; }
        if (g_process) { CloseHandle(g_process); g_process = NULL; }
        if (g_thread) { CloseHandle(g_thread); g_thread = NULL; }
        if (g_stdin_write) { CloseHandle(g_stdin_write); g_stdin_write=NULL; }
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_SEND),FALSE);
        EnableWindow(g_chat_test,FALSE);
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_STOP),FALSE);
        g_chat_send_when_ready=FALSE;
        g_chat_mode=FALSE;
        set_running(FALSE);
        append_log(was_evaluation
            ? (wp == 0 ? L"\r\n--- Evaluation completed ---\r\n" : L"\r\n--- Evaluation stopped or failed ---\r\n")
            : (wp == 0 ? L"\r\n--- Engine exited successfully ---\r\n" : L"\r\n--- Engine exited with an error ---\r\n"));
        if (g_close_when_done) DestroyWindow(hwnd);
        return 0;
    }
    case WM_CLOSE:
        if (g_process && InterlockedCompareExchange(&g_running,0,0)) {
            if (MessageBoxW(hwnd,L"An engine process is running. Stop it and close?",
                L"ProjectMinima",MB_YESNO|MB_ICONWARNING)!=IDYES) return 0;
            minima_settings_save(&g_settings_context);
            if (g_evaluation_mode) {
                g_close_when_done = TRUE;
                if (g_eval_job) { CloseHandle(g_eval_job); g_eval_job = NULL; }
                else TerminateProcess(g_process, 3);
            } else if(g_chat_mode){g_close_when_done=TRUE;TerminateProcess(g_process,3);}
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

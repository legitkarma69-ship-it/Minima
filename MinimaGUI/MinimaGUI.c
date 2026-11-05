#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Comdlg32.lib")

#define IDC_CORPUS       1001
#define IDC_ADD          1002
#define IDC_REMOVE       1003
#define IDC_STEPS        1004
#define IDC_RESUME       1005
#define IDC_START        1006
#define IDC_STOP         1007
#define IDC_LOG          1008
#define IDC_STATUS       1009
#define IDC_CHAT_START   1010
#define IDC_CHAT_INPUT   1011
#define IDC_CHAT_SEND    1012
#define IDC_CHAT_STOP    1013
#define IDC_SAVE_NOW     1014
#define WM_APPEND_LOG    (WM_APP + 1)
#define WM_ENGINE_EXIT   (WM_APP + 2)
#define IDT_GRAPH_REFRESH 1

static HWND g_window, g_corpus, g_steps, g_resume, g_log, g_status;
static HWND g_lr, g_warmup, g_log_interval, g_validation_batches;
static HWND g_checkpoint_interval, g_seed, g_tokenizer_bytes, g_graph, g_validation_interval;
static HWND g_temperature, g_top_k, g_top_p, g_repetition, g_max_tokens, g_chat_input;
static HWND g_vocab, g_dim, g_layers, g_heads, g_ffn, g_max_seq;
static HWND g_lr_schedule, g_tokenizer_sampling, g_batch_size, g_sequence_length;
static wchar_t g_metrics_path[MAX_PATH];
static HANDLE g_process, g_thread, g_stdin_write;
static BOOL g_chat_mode;
static volatile LONG g_running;
static BOOL g_close_when_done;
static wchar_t g_engine_path[MAX_PATH];

static LRESULT CALLBACK graph_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    (void)wp; (void)lp;
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps; HDC dc=BeginPaint(hwnd,&ps);
        RECT r; GetClientRect(hwnd,&r); FillRect(dc,&r,(HBRUSH)(COLOR_WINDOW+1));
        SetBkMode(dc,TRANSPARENT); SetTextColor(dc,RGB(40,40,40));
        TextOutW(dc,8,6,L"Training / validation loss",26);
        int left=42,top=30,right=r.right-12,bottom=r.bottom-24;
        HPEN grid=CreatePen(PS_SOLID,1,RGB(220,220,220)); HPEN old=(HPEN)SelectObject(dc,grid);
        for(int i=0;i<=4;i++){int y=top+(bottom-top)*i/4;MoveToEx(dc,left,y,NULL);LineTo(dc,right,y);}
        SelectObject(dc,old);DeleteObject(grid);
        float tr[512],va[512];int n=0;FILE* f=_wfopen(g_metrics_path,L"rt");
        if(f){
            char line[512];
            if(fseek(f,0,SEEK_END)==0){
                long end=ftell(f),start=end>262144?end-262144:0;
                if(start>0){fseek(f,start,SEEK_SET);(void)fgets(line,sizeof(line),f);}
                else {rewind(f);(void)fgets(line,sizeof(line),f);}
            }
            while(fgets(line,sizeof(line),f)){
            double step,a,b,lr,tp,vp,elapsed;
            if(sscanf_s(line,"%lf,%lf,%lf,%lf,%lf,%lf,%lf",&step,&a,&b,&lr,&tp,&vp,&elapsed)==7){
                if(n==512){memmove(tr,tr+1,511*sizeof(float));memmove(va,va+1,511*sizeof(float));n=511;}
                tr[n]=(float)a;va[n]=(float)b;n++;
            }}fclose(f);}
        if(n>1){float lo=tr[0],hi=tr[0];for(int i=0;i<n;i++){if(va[i]<lo)lo=va[i];if(tr[i]<lo)lo=tr[i];if(va[i]>hi)hi=va[i];if(tr[i]>hi)hi=tr[i];}
            if(hi-lo<0.001f)hi=lo+0.001f;
            HPEN p1=CreatePen(PS_SOLID,2,RGB(35,105,190)),p2=CreatePen(PS_SOLID,2,RGB(190,85,45));
            for(int s=0;s<2;s++){SelectObject(dc,s?p2:p1);for(int i=0;i<n;i++){float v=s?va[i]:tr[i];int x=left+(right-left)*i/(n-1);int y=bottom-(int)((v-lo)/(hi-lo)*(bottom-top));if(i==0)MoveToEx(dc,x,y,NULL);else LineTo(dc,x,y);}}
            SelectObject(dc,GetStockObject(BLACK_PEN));DeleteObject(p1);DeleteObject(p2);
            wchar_t legend[96];swprintf_s(legend,_countof(legend),L"Train %.3f   Valid %.3f",tr[n-1],va[n-1]);TextOutW(dc,left,bottom+5,legend,(int)wcslen(legend));
        }else TextOutW(dc,left,top+35,L"Waiting for training metrics...",31);
        EndPaint(hwnd,&ps);return 0;
    }
    return DefWindowProcW(hwnd,msg,wp,lp);
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
    HWND fields[]={g_lr,g_warmup,g_log_interval,g_validation_batches,g_checkpoint_interval,
        g_seed,g_tokenizer_bytes,g_validation_interval,g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,
        g_temperature,g_top_k,g_top_p,g_repetition,g_max_tokens,g_batch_size,g_sequence_length,g_lr_schedule,g_tokenizer_sampling};
    for(int i=0;i<(int)_countof(fields);i++)if(fields[i])EnableWindow(fields[i],!running);
    SetWindowTextW(g_status,running?(g_chat_mode?L"Chat running":L"Training running"):L"Idle");
}

static DWORD WINAPI reader_thread(void* parameter) {
    HANDLE pipe = (HANDLE)parameter;
    char buffer[2048];
    DWORD got;
    char* carry = (char*)calloc(1, 8192);
    size_t used = 0;
    if (!carry) {
        CloseHandle(pipe);
        if(g_process)TerminateProcess(g_process,1);
        PostMessageW(g_window,WM_ENGINE_EXIT,1,0);
        return 1;
    }

    while (ReadFile(pipe, buffer, sizeof(buffer), &got, NULL) && got) {
        for (DWORD i = 0; i < got; ++i) {
            if (used + 2 >= 8192) used = 0;
            carry[used++] = buffer[i];
            if (buffer[i] == '\n' || used >= 8000) {
                carry[used] = 0;
                if((used==11 && memcmp(carry,"CHAT_READY\n",11)==0) ||
                   (used==12 && memcmp(carry,"CHAT_CLOSED\n",12)==0)){used=0;continue;}
                UINT code_page=CP_UTF8;
                int needed=MultiByteToWideChar(code_page,0,carry,(int)used,NULL,0);
                if(needed<=0){code_page=CP_ACP;needed=MultiByteToWideChar(code_page,0,carry,(int)used,NULL,0);}
                if (needed > 0) {
                    wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
                    if (wide) {
                        MultiByteToWideChar(code_page, 0, carry, (int)used, wide, needed);
                        PostMessageW(g_window, WM_APPEND_LOG, 0, (LPARAM)wide);
                    }
                }
                used = 0;
            }
        }
    }
    if (used) {
        carry[used] = 0;
        UINT code_page = CP_UTF8;
        int needed = MultiByteToWideChar(code_page, 0, carry, (int)used, NULL, 0);
        if (needed <= 0) {
            code_page = CP_ACP;
            needed = MultiByteToWideChar(code_page, 0, carry, (int)used, NULL, 0);
        }
        if (needed > 0) {
            wchar_t* wide = (wchar_t*)calloc((size_t)needed + 1, sizeof(wchar_t));
            if (wide) {
                MultiByteToWideChar(code_page, 0, carry, (int)used, wide, needed);
                PostMessageW(g_window, WM_APPEND_LOG, 0, (LPARAM)wide);
            }
        }
    }
    free(carry);
    CloseHandle(pipe);

    if (g_process) {
        WaitForSingleObject(g_process, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(g_process, &code);
        PostMessageW(g_window, WM_ENGINE_EXIT, (WPARAM)code, 0);
    }
    return 0;
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

    wchar_t* first = file_buffer;
    wchar_t* next = first + wcslen(first) + 1;
    if (!*next) {
        SendMessageW(g_corpus, LB_ADDSTRING, 0, (LPARAM)first);
        return;
    }
    while (*next) {
        wchar_t full[MAX_PATH * 2];
        _snwprintf_s(full, _countof(full), _TRUNCATE, L"%s\\%s", first, next);
        SendMessageW(g_corpus, LB_ADDSTRING, 0, (LPARAM)full);
        next += wcslen(next) + 1;
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

static void start_engine(BOOL chat_mode) {
    if (InterlockedCompareExchange(&g_running, 0, 0)) return;

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
    } else {
        wchar_t steps[32];GetWindowTextW(g_steps,steps,_countof(steps));
        wchar_t* end=NULL;long count=wcstol(steps,&end,10);
        if(!end||*end||count<1||count>2147483647L){MessageBoxW(g_window,L"Enter a positive training-step count.",L"ProjectMinima",MB_ICONWARNING);return;}
        _snwprintf_s(command,_countof(command),_TRUNCATE,L"\"%s\" --steps %ld",g_engine_path,count);
        if(SendMessageW(g_resume,BM_GETCHECK,0,0)==BST_CHECKED)wcscat_s(command,_countof(command),L" --resume");
        if(!append_option(command,_countof(command),L"--lr",g_lr) ||
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
           !append_combo_option(command,_countof(command),L"--tokenizer-sampling",g_tokenizer_sampling)){
            MessageBoxW(g_window,L"Check the training settings.",L"ProjectMinima",MB_ICONWARNING);return;
        }
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
    int corpus_count = (int)SendMessageW(g_corpus, LB_GETCOUNT, 0, 0);
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
    CloseHandle(pi.hThread);
    g_chat_mode=chat_mode;
    set_running(TRUE);
    EnableWindow(GetDlgItem(g_window,IDC_STOP),!chat_mode);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_STOP),chat_mode);
    EnableWindow(g_chat_input,chat_mode);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_SEND),chat_mode);
    EnableWindow(GetDlgItem(g_window,IDC_CHAT_STOP),chat_mode);
    append_log(chat_mode?L"\r\n--- Starting checkpoint chat ---\r\n":L"\r\n--- Starting Minima training engine ---\r\n");
    g_thread = CreateThread(NULL, 0, reader_thread, read_pipe, 0, NULL);
    if (!g_thread) {
        TerminateProcess(g_process, 1);
        if(g_stdin_write){CloseHandle(g_stdin_write);g_stdin_write=NULL;}
        CloseHandle(g_process);
        g_process = NULL;
        CloseHandle(read_pipe);
        set_running(FALSE);
        MessageBoxW(g_window, L"Could not create the output-reader thread.", L"ProjectMinima", MB_ICONERROR);
    }
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
    append_log(L"\r\nYou: ");append_log(wide);append_log(L"\r\n");
    WriteFile(g_stdin_write,utf8,(DWORD)len,&written,NULL);
    free(utf8);SetWindowTextW(g_chat_input,L"");
}

static void stop_generation(void) {
    if(g_process && g_chat_mode && InterlockedCompareExchange(&g_running,0,0)){
        append_log(L"\r\nStopping generation...\r\n");
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

static void create_controls(HWND hwnd) {
    HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    CreateWindowW(L"STATIC", L"Corpus files", WS_CHILD | WS_VISIBLE, 12, 12, 130, 20, hwnd, NULL, NULL, NULL);
    g_corpus = CreateWindowW(L"LISTBOX", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | LBS_EXTENDEDSEL | WS_VSCROLL,
        12, 34, 520, 105, hwnd, (HMENU)IDC_CORPUS, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Add files...", WS_CHILD | WS_VISIBLE, 12, 146, 100, 26, hwnd, (HMENU)IDC_ADD, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Remove selected", WS_CHILD | WS_VISIBLE, 120, 146, 130, 26, hwnd, (HMENU)IDC_REMOVE, NULL, NULL);

    CreateWindowW(L"STATIC",L"Model: vocab",WS_CHILD|WS_VISIBLE,12,176,78,18,hwnd,NULL,NULL,NULL);
    g_vocab=CreateWindowW(L"EDIT",L"2048",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,92,173,55,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"dim",WS_CHILD|WS_VISIBLE,155,176,28,18,hwnd,NULL,NULL,NULL);
    g_dim=CreateWindowW(L"EDIT",L"64",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,184,173,42,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"layers",WS_CHILD|WS_VISIBLE,235,176,42,18,hwnd,NULL,NULL,NULL);
    g_layers=CreateWindowW(L"EDIT",L"2",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,278,173,35,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"heads",WS_CHILD|WS_VISIBLE,320,176,40,18,hwnd,NULL,NULL,NULL);
    g_heads=CreateWindowW(L"EDIT",L"2",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,362,173,35,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"FFN",WS_CHILD|WS_VISIBLE,404,176,30,18,hwnd,NULL,NULL,NULL);
    g_ffn=CreateWindowW(L"EDIT",L"128",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,435,173,48,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"context",WS_CHILD|WS_VISIBLE,490,176,48,18,hwnd,NULL,NULL,NULL);
    g_max_seq=CreateWindowW(L"EDIT",L"64",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,540,173,42,23,hwnd,NULL,NULL,NULL);

    CreateWindowW(L"STATIC",L"Steps",WS_CHILD|WS_VISIBLE,12,210,42,18,hwnd,NULL,NULL,NULL);
    g_steps=CreateWindowW(L"EDIT",L"2000",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,56,207,58,23,hwnd,(HMENU)IDC_STEPS,NULL,NULL);
    g_resume=CreateWindowW(L"BUTTON",L"Resume",WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX,118,208,75,23,hwnd,(HMENU)IDC_RESUME,NULL,NULL);
    CreateWindowW(L"STATIC",L"Batch",WS_CHILD|WS_VISIBLE,198,210,38,18,hwnd,NULL,NULL,NULL);
    g_batch_size=CreateWindowW(L"EDIT",L"4",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,237,207,35,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Seq",WS_CHILD|WS_VISIBLE,278,210,30,18,hwnd,NULL,NULL,NULL);
    g_sequence_length=CreateWindowW(L"EDIT",L"32",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,309,207,40,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"LR",WS_CHILD|WS_VISIBLE,355,210,22,18,hwnd,NULL,NULL,NULL);
    g_lr=CreateWindowW(L"EDIT",L"0.0003",WS_CHILD|WS_VISIBLE|WS_BORDER,379,207,65,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Warmup",WS_CHILD|WS_VISIBLE,450,210,52,18,hwnd,NULL,NULL,NULL);
    g_warmup=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,503,207,45,23,hwnd,NULL,NULL,NULL);

    CreateWindowW(L"STATIC",L"Log every",WS_CHILD|WS_VISIBLE,12,244,60,18,hwnd,NULL,NULL,NULL);
    g_log_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,74,241,48,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Validate every",WS_CHILD|WS_VISIBLE,132,244,90,18,hwnd,NULL,NULL,NULL);
    g_validation_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,224,241,45,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Val batches",WS_CHILD|WS_VISIBLE,276,244,68,18,hwnd,NULL,NULL,NULL);
    g_validation_batches=CreateWindowW(L"EDIT",L"16",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,346,241,40,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Save every",WS_CHILD|WS_VISIBLE,392,244,65,18,hwnd,NULL,NULL,NULL);
    g_checkpoint_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,459,241,42,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Seed",WS_CHILD|WS_VISIBLE,507,244,35,18,hwnd,NULL,NULL,NULL);
    g_seed=CreateWindowW(L"EDIT",L"1234",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,542,241,55,23,hwnd,NULL,NULL,NULL);

    CreateWindowW(L"STATIC",L"Tokenizer bytes",WS_CHILD|WS_VISIBLE,12,278,92,18,hwnd,NULL,NULL,NULL);
    g_tokenizer_bytes=CreateWindowW(L"EDIT",L"262144",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,106,275,75,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Temp",WS_CHILD|WS_VISIBLE,195,278,38,18,hwnd,NULL,NULL,NULL);
    g_temperature=CreateWindowW(L"EDIT",L"0.8",WS_CHILD|WS_VISIBLE|WS_BORDER,234,275,45,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Top-k",WS_CHILD|WS_VISIBLE,285,278,38,18,hwnd,NULL,NULL,NULL);
    g_top_k=CreateWindowW(L"EDIT",L"40",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,324,275,40,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Top-p",WS_CHILD|WS_VISIBLE,370,278,38,18,hwnd,NULL,NULL,NULL);
    g_top_p=CreateWindowW(L"EDIT",L"0.9",WS_CHILD|WS_VISIBLE|WS_BORDER,409,275,45,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Repeat",WS_CHILD|WS_VISIBLE,460,278,45,18,hwnd,NULL,NULL,NULL);
    g_repetition=CreateWindowW(L"EDIT",L"1.1",WS_CHILD|WS_VISIBLE|WS_BORDER,506,275,45,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Max tokens",WS_CHILD|WS_VISIBLE,12,309,75,18,hwnd,NULL,NULL,NULL);
    g_max_tokens=CreateWindowW(L"EDIT",L"32",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,90,306,48,23,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"LR schedule",WS_CHILD|WS_VISIBLE,155,309,68,18,hwnd,NULL,NULL,NULL);
    g_lr_schedule=CreateWindowW(L"COMBOBOX",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|CBS_DROPDOWNLIST|WS_VSCROLL,225,306,95,100,hwnd,NULL,NULL,NULL);
    SendMessageW(g_lr_schedule,CB_ADDSTRING,0,(LPARAM)L"cosine");SendMessageW(g_lr_schedule,CB_ADDSTRING,0,(LPARAM)L"constant");SendMessageW(g_lr_schedule,CB_SETCURSEL,0,0);
    CreateWindowW(L"STATIC",L"Tokenizer sample",WS_CHILD|WS_VISIBLE,330,309,95,18,hwnd,NULL,NULL,NULL);
    g_tokenizer_sampling=CreateWindowW(L"COMBOBOX",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|CBS_DROPDOWNLIST|WS_VSCROLL,427,306,95,100,hwnd,NULL,NULL,NULL);
    SendMessageW(g_tokenizer_sampling,CB_ADDSTRING,0,(LPARAM)L"spread");SendMessageW(g_tokenizer_sampling,CB_ADDSTRING,0,(LPARAM)L"prefix");SendMessageW(g_tokenizer_sampling,CB_SETCURSEL,0,0);

    CreateWindowW(L"BUTTON",L"Start training",WS_CHILD|WS_VISIBLE,12,340,120,30,hwnd,(HMENU)IDC_START,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Stop training",WS_CHILD|WS_VISIBLE|WS_DISABLED,142,340,105,30,hwnd,(HMENU)IDC_STOP,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Start chat",WS_CHILD|WS_VISIBLE,257,340,90,30,hwnd,(HMENU)IDC_CHAT_START,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Save checkpoint",WS_CHILD|WS_VISIBLE|WS_DISABLED,355,340,105,30,hwnd,(HMENU)IDC_SAVE_NOW,NULL,NULL);
    g_status=CreateWindowW(L"STATIC",L"Idle",WS_CHILD|WS_VISIBLE,470,346,110,20,hwnd,(HMENU)IDC_STATUS,NULL,NULL,NULL);

    g_graph=CreateWindowExW(WS_EX_CLIENTEDGE,L"MinimaLossGraph",L"",WS_CHILD|WS_VISIBLE,600,170,285,175,hwnd,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Engine output / chat history",WS_CHILD|WS_VISIBLE,12,380,300,20,hwnd,NULL,NULL,NULL);
    g_log=CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",L"",
        WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_AUTOVSCROLL|ES_READONLY,
        12,402,850,250,hwnd,(HMENU)IDC_LOG,NULL,NULL);
    g_chat_input=CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL,
        12,665,700,26,hwnd,(HMENU)IDC_CHAT_INPUT,NULL,NULL);
    SendMessageW(g_chat_input,EM_SETLIMITTEXT,2048,0);
    EnableWindow(g_chat_input,FALSE);
    CreateWindowW(L"BUTTON",L"Generate / Send",WS_CHILD|WS_VISIBLE|WS_DISABLED,720,665,110,26,hwnd,(HMENU)IDC_CHAT_SEND,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Stop generation",WS_CHILD|WS_VISIBLE|WS_DISABLED,835,665,100,26,hwnd,(HMENU)IDC_CHAT_STOP,NULL,NULL);

    HWND controls[] = { g_corpus,g_steps,g_resume,g_log,g_lr,g_warmup,g_log_interval,
        g_validation_batches,g_validation_interval,g_checkpoint_interval,g_seed,g_tokenizer_bytes,g_graph,
        g_temperature,g_top_k,g_top_p,g_repetition,g_max_tokens,g_chat_input,
        g_vocab,g_dim,g_layers,g_heads,g_ffn,g_max_seq,g_lr_schedule,g_tokenizer_sampling,
        g_batch_size,g_sequence_length };
    for (int i = 0; i < (int)_countof(controls); ++i)
        SendMessageW(controls[i], WM_SETFONT, (WPARAM)font, TRUE);

    wchar_t history_path[MAX_PATH];wcscpy_s(history_path,_countof(history_path),g_engine_path);
    wchar_t* history_slash=wcsrchr(history_path,L'\\');
    if(history_slash)wcscpy_s(history_slash+1,_countof(history_path)-(size_t)(history_slash+1-history_path),L"minima_chat.txt");
    FILE* history_file=_wfopen(history_path,L"rt");
    if(history_file){
        char line[4096];
        while(fgets(line,sizeof(line),history_file)){
            int count=MultiByteToWideChar(CP_UTF8,0,line,-1,NULL,0);
            if(count>0){wchar_t* wide=(wchar_t*)calloc((size_t)count,sizeof(wchar_t));
                if(wide){MultiByteToWideChar(CP_UTF8,0,line,-1,wide,count);append_log(wide);free(wide);}}
        }
        fclose(history_file);
    }
}

static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        g_window = hwnd; create_controls(hwnd); SetTimer(hwnd,IDT_GRAPH_REFRESH,1000,NULL); return 0;
    case WM_TIMER:
        if(wp==IDT_GRAPH_REFRESH && g_graph) InvalidateRect(g_graph,NULL,FALSE);
        return 0;
    case WM_SIZE: {
        RECT r;
        GetClientRect(hwnd, &r);
        MoveWindow(g_corpus,12,34,r.right-24,95,TRUE);
        MoveWindow(g_graph,r.right-320,170,285,175,TRUE);
        MoveWindow(g_log,12,402,r.right-24,r.bottom-510,TRUE);
        MoveWindow(g_chat_input,12,r.bottom-75,r.right-240,26,TRUE);
        MoveWindow(GetDlgItem(hwnd,IDC_CHAT_SEND),r.right-225,r.bottom-75,110,26,TRUE);
        MoveWindow(GetDlgItem(hwnd,IDC_CHAT_STOP),r.right-110,r.bottom-75,100,26,TRUE);
        return 0;
    }
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_ADD: add_corpus_files(); return 0;
        case IDC_REMOVE: {
            int selected[128];
            LRESULT n = SendMessageW(g_corpus, LB_GETSELITEMS, 128, (LPARAM)selected);
            for (int i = (int)n - 1; i >= 0; --i)
                SendMessageW(g_corpus, LB_DELETESTRING, selected[i], 0);
            return 0;
        }
        case IDC_START: start_engine(FALSE); return 0;
        case IDC_STOP: stop_engine(); return 0;
        case IDC_CHAT_START: start_engine(TRUE); return 0;
        case IDC_CHAT_SEND: send_chat_message(); return 0;
        case IDC_CHAT_STOP: stop_generation(); return 0;
        case IDC_SAVE_NOW: request_checkpoint_save(); return 0;
        }
        break;
    case WM_APPEND_LOG:
        append_log((const wchar_t*)lp);
        free((void*)lp);
        return 0;
    case WM_ENGINE_EXIT:
        if (g_process) { CloseHandle(g_process); g_process = NULL; }
        if (g_thread) { CloseHandle(g_thread); g_thread = NULL; }
        if (g_stdin_write) { CloseHandle(g_stdin_write); g_stdin_write=NULL; }
        EnableWindow(g_chat_input,FALSE);
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_SEND),FALSE);
        EnableWindow(GetDlgItem(hwnd,IDC_CHAT_STOP),FALSE);
        set_running(FALSE);
        append_log(wp == 0 ? L"\r\n--- Engine exited successfully ---\r\n" : L"\r\n--- Engine exited with an error ---\r\n");
        g_chat_mode=FALSE;
        if (g_close_when_done) DestroyWindow(hwnd);
        return 0;
    case WM_CLOSE:
        if (g_process && InterlockedCompareExchange(&g_running,0,0)) {
            if (MessageBoxW(hwnd,L"An engine process is running. Stop it and close?",
                L"ProjectMinima",MB_YESNO|MB_ICONWARNING)!=IDYES) return 0;
            if(g_chat_mode){g_close_when_done=TRUE;TerminateProcess(g_process,3);}
            else {if(!request_cooperative_stop())return 0;g_close_when_done=TRUE;}
            return 0;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command_line, int show) {
    (void)previous; (void)command_line;
    INITCOMMONCONTROLSEX controls = { sizeof(controls), ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS };
    InitCommonControlsEx(&controls);

    DWORD n = GetModuleFileNameW(NULL, g_engine_path, _countof(g_engine_path));
    if (!n || n >= _countof(g_engine_path)) return 1;
    wchar_t* slash = wcsrchr(g_engine_path, L'\\');
    if (!slash) return 1;
    wcscpy_s(slash + 1, _countof(g_engine_path) - (size_t)(slash + 1 - g_engine_path), L"Minima.exe");

    WNDCLASSW gwc; ZeroMemory(&gwc,sizeof(gwc));
    gwc.lpfnWndProc=graph_proc;gwc.hInstance=instance;gwc.lpszClassName=L"MinimaLossGraph";
    gwc.hCursor=LoadCursorW(NULL,IDC_ARROW);gwc.hbrBackground=(HBRUSH)(COLOR_WINDOW+1);
    if(!RegisterClassW(&gwc))return 1;
    DWORD plen=GetModuleFileNameW(NULL,g_metrics_path,_countof(g_metrics_path));
    if(!plen||plen>=_countof(g_metrics_path))return 1;
    wchar_t* pslash=wcsrchr(g_metrics_path,L'\\');
    if(pslash)wcscpy_s(pslash+1,_countof(g_metrics_path)-(size_t)(pslash+1-g_metrics_path),L"minima_metrics.csv");

    WNDCLASSW wc; ZeroMemory(&wc,sizeof(wc));
    wc.lpfnWndProc = window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = L"ProjectMinimaWin32";
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    if (!RegisterClassW(&wc)) return 1;

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"ProjectMinima",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 960, 780,
        NULL, NULL, instance, NULL);
    if (!hwnd) return 1;
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}

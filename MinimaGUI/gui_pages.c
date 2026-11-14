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
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | TCS_FIXEDWIDTH,
            12, 6, 920, 28, parent, (HMENU)(INT_PTR)IDC_MAIN_TABS,
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
            8, 236, 220, 24, parent, (HMENU)(INT_PTR)IDC_ADVANCED_TOGGLE,
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

void gui_pages_create_content(HWND parent, GuiPageControls* page_controls,
    void (*register_advanced)(HWND)) {
    if (!page_controls) return;
#define REGISTER_ADVANCED(control) do { if (register_advanced) register_advanced(control); } while (0)
#define g_corpus (*page_controls->corpus)
#define g_steps (*page_controls->steps)
#define g_resume (*page_controls->resume)
#define g_vocab (*page_controls->vocab)
#define g_dim (*page_controls->dim)
#define g_layers (*page_controls->layers)
#define g_heads (*page_controls->heads)
#define g_ffn (*page_controls->ffn)
#define g_max_seq (*page_controls->max_seq)
#define g_batch_size (*page_controls->batch_size)
#define g_sequence_length (*page_controls->sequence_length)
#define g_lr (*page_controls->lr)
#define g_warmup (*page_controls->warmup)
#define g_log_interval (*page_controls->log_interval)
#define g_validation_interval (*page_controls->validation_interval)
#define g_validation_batches (*page_controls->validation_batches)
#define g_checkpoint_interval (*page_controls->checkpoint_interval)
#define g_seed (*page_controls->seed)
#define g_tokenizer_bytes (*page_controls->tokenizer_bytes)
#define g_temperature (*page_controls->temperature)
#define g_top_k (*page_controls->top_k)
#define g_top_p (*page_controls->top_p)
#define g_repetition (*page_controls->repetition)
#define g_max_tokens (*page_controls->max_tokens)
#define g_lr_schedule (*page_controls->lr_schedule)
#define g_tokenizer_sampling (*page_controls->tokenizer_sampling)
#define g_total_steps (*page_controls->total_steps)
#define g_status (*page_controls->status)
#define g_graph (*page_controls->graph)
#define g_output_label (*page_controls->output_label)
#define g_log (*page_controls->log)
#define g_chat_input (*page_controls->chat_input)
#define g_chat_test (*page_controls->chat_test)
    CreateWindowW(L"STATIC", L"Corpus files", WS_CHILD | WS_VISIBLE, 12, 12, 130, 20, parent, NULL, NULL, NULL);
    g_corpus = CreateWindowW(L"LISTBOX", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | LBS_EXTENDEDSEL | WS_VSCROLL,
        12, 34, 520, 105, parent, (HMENU)(INT_PTR)IDC_CORPUS, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Add files...", WS_CHILD | WS_VISIBLE, 12, 146, 100, 26, parent, (HMENU)(INT_PTR)IDC_ADD, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Remove selected", WS_CHILD | WS_VISIBLE, 120, 146, 130, 26, parent, (HMENU)(INT_PTR)IDC_REMOVE, NULL, NULL);
    CreateWindowW(L"BUTTON", L"Basic defaults", WS_CHILD | WS_VISIBLE, 260, 146, 132, 26, parent, (HMENU)(INT_PTR)IDC_BASIC_DEFAULTS, NULL, NULL);

    CreateWindowW(L"STATIC",L"Model: vocab",WS_CHILD|WS_VISIBLE,12,176,78,18,parent,NULL,NULL,NULL);
    g_vocab=CreateWindowW(L"EDIT",L"2048",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,92,173,55,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"dim",WS_CHILD|WS_VISIBLE,155,176,28,18,parent,NULL,NULL,NULL);
    g_dim=CreateWindowW(L"EDIT",L"64",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,184,173,42,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"layers",WS_CHILD|WS_VISIBLE,235,176,42,18,parent,NULL,NULL,NULL);
    g_layers=CreateWindowW(L"EDIT",L"2",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,278,173,35,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"heads",WS_CHILD|WS_VISIBLE,320,176,40,18,parent,NULL,NULL,NULL);
    g_heads=CreateWindowW(L"EDIT",L"2",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,362,173,35,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"FFN",WS_CHILD|WS_VISIBLE,404,176,30,18,parent,NULL,NULL,NULL);
    g_ffn=CreateWindowW(L"EDIT",L"128",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,435,173,48,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"context",WS_CHILD|WS_VISIBLE,490,176,48,18,parent,NULL,NULL,NULL);
    g_max_seq=CreateWindowW(L"EDIT",L"64",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,540,173,42,23,parent,NULL,NULL,NULL);

    CreateWindowW(L"STATIC",L"Steps",WS_CHILD|WS_VISIBLE,12,210,42,18,parent,NULL,NULL,NULL);
    g_steps=CreateWindowW(L"EDIT",L"2000",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,56,207,58,23,parent,(HMENU)(INT_PTR)IDC_STEPS,NULL,NULL);
    g_resume=CreateWindowW(L"BUTTON",L"Resume",WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX,118,208,75,23,parent,(HMENU)(INT_PTR)IDC_RESUME,NULL,NULL);
    CreateWindowW(L"STATIC",L"Batch",WS_CHILD|WS_VISIBLE,198,210,38,18,parent,NULL,NULL,NULL);
    g_batch_size=CreateWindowW(L"EDIT",L"4",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,237,207,35,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Seq",WS_CHILD|WS_VISIBLE,278,210,30,18,parent,NULL,NULL,NULL);
    g_sequence_length=CreateWindowW(L"EDIT",L"32",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,309,207,40,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"LR",WS_CHILD|WS_VISIBLE,355,210,22,18,parent,NULL,NULL,NULL);
    g_lr=CreateWindowW(L"EDIT",L"0.0003",WS_CHILD|WS_VISIBLE|WS_BORDER,379,207,65,23,parent,NULL,NULL,NULL);
    CreateWindowW(L"STATIC",L"Warmup",WS_CHILD|WS_VISIBLE,450,210,52,18,parent,NULL,NULL,NULL);
    g_warmup=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,503,207,45,23,parent,NULL,NULL,NULL);

    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Log every",WS_CHILD|WS_VISIBLE,12,264,60,18,parent,NULL,NULL,NULL));
    g_log_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,74,261,48,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_log_interval);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Validate every",WS_CHILD|WS_VISIBLE,132,264,90,18,parent,NULL,NULL,NULL));
    g_validation_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,224,261,45,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_validation_interval);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Val batches",WS_CHILD|WS_VISIBLE,276,264,68,18,parent,NULL,NULL,NULL));
    g_validation_batches=CreateWindowW(L"EDIT",L"16",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,346,261,40,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_validation_batches);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Save every",WS_CHILD|WS_VISIBLE,392,264,65,18,parent,NULL,NULL,NULL));
    g_checkpoint_interval=CreateWindowW(L"EDIT",L"100",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,459,261,42,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_checkpoint_interval);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Seed (fresh)",WS_CHILD|WS_VISIBLE,507,264,55,18,parent,NULL,NULL,NULL));
    g_seed=CreateWindowW(L"EDIT",L"1234",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,542,261,55,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_seed);

    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Tokenizer bytes",WS_CHILD|WS_VISIBLE,12,298,92,18,parent,NULL,NULL,NULL));
    g_tokenizer_bytes=CreateWindowW(L"EDIT",L"1048576",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,106,295,75,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_tokenizer_bytes);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Temp",WS_CHILD|WS_VISIBLE,195,298,38,18,parent,NULL,NULL,NULL));
    g_temperature=CreateWindowW(L"EDIT",L"0.8",WS_CHILD|WS_VISIBLE|WS_BORDER,234,295,45,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_temperature);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Top-k",WS_CHILD|WS_VISIBLE,285,298,38,18,parent,NULL,NULL,NULL));
    g_top_k=CreateWindowW(L"EDIT",L"40",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,324,295,40,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_top_k);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Top-p",WS_CHILD|WS_VISIBLE,370,298,38,18,parent,NULL,NULL,NULL));
    g_top_p=CreateWindowW(L"EDIT",L"0.9",WS_CHILD|WS_VISIBLE|WS_BORDER,409,295,45,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_top_p);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Repeat",WS_CHILD|WS_VISIBLE,460,298,45,18,parent,NULL,NULL,NULL));
    g_repetition=CreateWindowW(L"EDIT",L"1.1",WS_CHILD|WS_VISIBLE|WS_BORDER,506,295,45,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_repetition);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Max tokens",WS_CHILD|WS_VISIBLE,12,329,75,18,parent,NULL,NULL,NULL));
    g_max_tokens=CreateWindowW(L"EDIT",L"32",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,90,326,48,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_max_tokens);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"LR schedule",WS_CHILD|WS_VISIBLE,155,329,68,18,parent,NULL,NULL,NULL));
    g_lr_schedule=CreateWindowW(L"COMBOBOX",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|CBS_DROPDOWNLIST|WS_VSCROLL,225,326,95,100,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_lr_schedule);
    SendMessageW(g_lr_schedule,CB_ADDSTRING,0,(LPARAM)L"cosine");SendMessageW(g_lr_schedule,CB_ADDSTRING,0,(LPARAM)L"constant");SendMessageW(g_lr_schedule,CB_SETCURSEL,0,0);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Tokenizer sample",WS_CHILD|WS_VISIBLE,330,329,95,18,parent,NULL,NULL,NULL));
    g_tokenizer_sampling=CreateWindowW(L"COMBOBOX",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|CBS_DROPDOWNLIST|WS_VSCROLL,427,326,95,100,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_tokenizer_sampling);
    SendMessageW(g_tokenizer_sampling,CB_ADDSTRING,0,(LPARAM)L"spread");SendMessageW(g_tokenizer_sampling,CB_ADDSTRING,0,(LPARAM)L"prefix");SendMessageW(g_tokenizer_sampling,CB_SETCURSEL,0,0);
    REGISTER_ADVANCED(CreateWindowW(L"STATIC",L"Total steps",WS_CHILD|WS_VISIBLE,535,329,65,18,parent,NULL,NULL,NULL));
    g_total_steps=CreateWindowW(L"EDIT",L"2000",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_NUMBER,600,326,70,23,parent,NULL,NULL,NULL); REGISTER_ADVANCED(g_total_steps);

    CreateWindowW(L"BUTTON",L"Start training",WS_CHILD|WS_VISIBLE|BS_DEFPUSHBUTTON,12,340,120,30,parent,(HMENU)(INT_PTR)IDC_START,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Stop training",WS_CHILD|WS_VISIBLE|WS_DISABLED,142,340,105,30,parent,(HMENU)(INT_PTR)IDC_STOP,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Start chat",WS_CHILD|WS_VISIBLE,257,340,90,30,parent,(HMENU)(INT_PTR)IDC_CHAT_START,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Save checkpoint",WS_CHILD|WS_VISIBLE|WS_DISABLED,355,340,105,30,parent,(HMENU)(INT_PTR)IDC_SAVE_NOW,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Rebuild tokenizer",WS_CHILD|WS_VISIBLE,470,340,135,30,parent,(HMENU)(INT_PTR)IDC_REBUILD_TOKENIZER,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Auto settings",WS_CHILD|WS_VISIBLE,612,340,115,30,parent,(HMENU)(INT_PTR)IDC_AUTO_SETTINGS,NULL,NULL);
    g_status=CreateWindowW(L"STATIC",L"Idle",WS_CHILD|WS_VISIBLE,738,346,110,20,parent,(HMENU)(INT_PTR)IDC_STATUS,NULL,NULL);

    g_graph=CreateWindowExW(WS_EX_CLIENTEDGE,L"MinimaLossGraph",L"",WS_CHILD|WS_VISIBLE,600,170,285,175,parent,NULL,NULL,NULL);
    g_output_label=CreateWindowW(L"STATIC",L"Engine output / chat history",WS_CHILD|WS_VISIBLE,12,380,300,20,parent,NULL,NULL,NULL);
    g_log=CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",L"",
        WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_AUTOVSCROLL|ES_READONLY,
        12,402,850,250,parent,(HMENU)(INT_PTR)IDC_LOG,NULL,NULL);
    SendMessageW(g_log,EM_SETLIMITTEXT,1024*1024,0);
    g_chat_input=CreateWindowExW(WS_EX_CLIENTEDGE,L"COMBOBOX",L"",
        WS_CHILD|WS_VISIBLE|WS_BORDER|CBS_DROPDOWN|CBS_AUTOHSCROLL|WS_VSCROLL,
        12,665,600,180,parent,(HMENU)(INT_PTR)IDC_CHAT_INPUT,NULL,NULL);
    SendMessageW(g_chat_input,CB_LIMITTEXT,1024,0);
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"The quick brown fox");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"Once upon a time");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"In the beginning");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"The meaning of life");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"It was a dark");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"Hello there, how do you do?");
    SendMessageW(g_chat_input,CB_ADDSTRING,0,(LPARAM)L"Explain how a small language model learns patterns.");
    COMBOBOXINFO chat_combo_info = { sizeof(chat_combo_info) };
    if (GetComboBoxInfo(g_chat_input, &chat_combo_info) && chat_combo_info.hwndItem)
        SendMessageW(chat_combo_info.hwndItem, EM_SETCUEBANNER, TRUE,
            (LPARAM)L"Type a message and press Enter to start or send; the arrow shows examples.");
    g_chat_test=CreateWindowW(L"BUTTON",L"Test settings",WS_CHILD|WS_VISIBLE|WS_DISABLED,620,665,100,26,parent,(HMENU)(INT_PTR)IDC_CHAT_TEST,NULL,NULL);
    REGISTER_ADVANCED(g_chat_test);
    CreateWindowW(L"BUTTON",L"Send",WS_CHILD|WS_VISIBLE|WS_DISABLED,730,665,110,26,parent,(HMENU)(INT_PTR)IDC_CHAT_SEND,NULL,NULL);
    CreateWindowW(L"BUTTON",L"Stop generation",WS_CHILD|WS_VISIBLE|WS_DISABLED,850,665,100,26,parent,(HMENU)(INT_PTR)IDC_CHAT_STOP,NULL,NULL);
#undef g_corpus
#undef g_steps
#undef g_resume
#undef g_vocab
#undef g_dim
#undef g_layers
#undef g_heads
#undef g_ffn
#undef g_max_seq
#undef g_batch_size
#undef g_sequence_length
#undef g_lr
#undef g_warmup
#undef g_log_interval
#undef g_validation_interval
#undef g_validation_batches
#undef g_checkpoint_interval
#undef g_seed
#undef g_tokenizer_bytes
#undef g_temperature
#undef g_top_k
#undef g_top_p
#undef g_repetition
#undef g_max_tokens
#undef g_lr_schedule
#undef g_tokenizer_sampling
#undef g_total_steps
#undef g_status
#undef g_graph
#undef g_output_label
#undef g_log
#undef g_chat_input
#undef g_chat_test
#undef REGISTER_ADVANCED
}

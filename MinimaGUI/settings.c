#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <wchar.h>
#include "settings.h"

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

static const wchar_t* const g_edit_keys[24] = {
    L"steps", L"vocab", L"dim", L"layers", L"heads", L"ffn", L"context",
    L"batch", L"sequence", L"learning_rate", L"warmup", L"total_steps",
    L"log_interval", L"validation_interval", L"validation_batches",
    L"checkpoint_interval", L"seed", L"tokenizer_bytes", L"temperature",
    L"top_k", L"top_p", L"repetition", L"max_tokens", L"grad_clip"
};
static const wchar_t* const g_edit_defaults[24] = {
    L"10000", L"2048", L"64", L"2", L"2", L"128", L"64", L"4", L"64",
    L"0.0003", L"2500", L"50000", L"1000", L"1000", L"16", L"1000", L"1234",
    L"1048576", L"0.8", L"40", L"0.9", L"1.1", L"32", L"1.0"
};
static const wchar_t* const g_combo_keys[3] = {
    L"lr_schedule", L"tokenizer_sampling", L"corpus_mix"
};
static const wchar_t* const g_combo_defaults[3] = {
    L"cosine", L"spread", L"sqrt"
};

static void load_edit_setting(const wchar_t* path, const wchar_t* key,
    HWND control, const wchar_t* fallback) {
    wchar_t value[128];
    GetPrivateProfileStringW(L"Training", key, fallback, value,
        (DWORD)_countof(value), path);
    SetWindowTextW(control, value);
}

static BOOL read_positive_edit_value(HWND control, long* value_out) {
    wchar_t text[64], *end = NULL;
    GetWindowTextW(control, text, (int)_countof(text));
    if (!text[0]) return FALSE;
    const long value = wcstol(text, &end, 10);
    if (!end || *end || value < 1) return FALSE;
    *value_out = value;
    return TRUE;
}

static void migrate_long_run_intervals(const MinimaSettingsContext* context) {
    long run_steps = 0, schedule_steps = 0;
    (void)read_positive_edit_value(context->edit_controls[0], &run_steps);
    (void)read_positive_edit_value(context->edit_controls[11], &schedule_steps);
    if (run_steps < 50000 && schedule_steps < 50000) return;

    /* Replace the old spammy default, while preserving any custom interval. */
    const int interval_fields[] = { 12, 13, 15 };
    for (int i = 0; i < (int)_countof(interval_fields); ++i) {
        long interval = 0;
        const int field = interval_fields[i];
        if (read_positive_edit_value(context->edit_controls[field], &interval) &&
            interval == 100)
            SetWindowTextW(context->edit_controls[field], L"1000");
    }
}

static void save_edit_setting(const wchar_t* path, const wchar_t* key, HWND control) {
    wchar_t value[128];
    GetWindowTextW(control, value, (int)_countof(value));
    WritePrivateProfileStringW(L"Training", key, value, path);
}

static void load_combo_setting(const wchar_t* path, const wchar_t* key,
    HWND control, const wchar_t* fallback) {
    wchar_t value[64];
    GetPrivateProfileStringW(L"Training", key, fallback, value,
        (DWORD)_countof(value), path);
    LRESULT index = SendMessageW(control, CB_FINDSTRINGEXACT,
        (WPARAM)-1, (LPARAM)value);
    if (index >= 0) SendMessageW(control, CB_SETCURSEL, (WPARAM)index, 0);
}

static void save_combo_setting(const wchar_t* path, const wchar_t* key, HWND control) {
    int index = (int)SendMessageW(control, CB_GETCURSEL, 0, 0);
    if (index < 0) return;
    wchar_t value[64];
    SendMessageW(control, CB_GETLBTEXT, (WPARAM)index, (LPARAM)value);
    WritePrivateProfileStringW(L"Training", key, value, path);
}

void minima_settings_load(const MinimaSettingsContext* context) {
    if (!context || !context->path || !context->path[0]) return;
    for (int i = 0; i < 24; ++i)
        load_edit_setting(context->path, g_edit_keys[i],
            context->edit_controls[i], g_edit_defaults[i]);
    migrate_long_run_intervals(context);
    for (int i = 0; i < 3; ++i)
        load_combo_setting(context->path, g_combo_keys[i],
            context->combo_controls[i], g_combo_defaults[i]);

    const BOOL expanded = GetPrivateProfileIntW(
        L"UI", L"advanced_expanded", 0, context->path) ? TRUE : FALSE;
    if (context->set_advanced_expanded)
        context->set_advanced_expanded(context->window, expanded);

    SendMessageW(context->resume_control, BM_SETCHECK,
        GetPrivateProfileIntW(L"Training", L"resume", 0, context->path)
            ? BST_CHECKED : BST_UNCHECKED, 0);
    if (context->use_saved_tokenizer)
        *context->use_saved_tokenizer = GetPrivateProfileIntW(
            L"Training", L"prebuilt_tokenizer", 0, context->path) ? TRUE : FALSE;
    if (context->prebuilt_vocab)
        GetPrivateProfileStringW(L"Training", L"prebuilt_vocab", L"",
            context->prebuilt_vocab, 64, context->path);
    if (context->prebuilt_tokenizer_bytes)
        GetPrivateProfileStringW(L"Training", L"prebuilt_tokenizer_bytes", L"",
            context->prebuilt_tokenizer_bytes, 64, context->path);
    if (context->prebuilt_sampling)
        GetPrivateProfileStringW(L"Training", L"prebuilt_sampling", L"",
            context->prebuilt_sampling, 64, context->path);

    int count = GetPrivateProfileIntW(L"Corpus", L"count", 0, context->path);
    if (count < 0) count = 0;
    if (count > 32) count = 32;
    for (int i = 0; i < count; ++i) {
        wchar_t key[32], file_path[MAX_PATH * 2];
        _snwprintf_s(key, _countof(key), _TRUNCATE, L"file%d", i);
        GetPrivateProfileStringW(L"Corpus", key, L"", file_path,
            (DWORD)_countof(file_path), context->path);
        if (file_path[0])
            SendMessageW(context->corpus_control, LB_ADDSTRING, 0, (LPARAM)file_path);
    }
    if (context->update_fresh_only_controls)
        context->update_fresh_only_controls();
}

void minima_settings_reset_defaults(const MinimaSettingsContext* context) {
    if (!context) return;
    for (int i = 0; i < 24; ++i)
        SetWindowTextW(context->edit_controls[i], g_edit_defaults[i]);
    for (int i = 0; i < 3; ++i) {
        LRESULT index = SendMessageW(context->combo_controls[i],
            CB_FINDSTRINGEXACT, (WPARAM)-1, (LPARAM)g_combo_defaults[i]);
        if (index >= 0)
            SendMessageW(context->combo_controls[i], CB_SETCURSEL, (WPARAM)index, 0);
    }
    SendMessageW(context->resume_control, BM_SETCHECK, BST_UNCHECKED, 0);
    if (context->use_saved_tokenizer)
        *context->use_saved_tokenizer = FALSE;
    if (context->path && context->path[0])
        WritePrivateProfileStringW(L"Training", L"prebuilt_tokenizer", L"0", context->path);
    if (context->update_fresh_only_controls)
        context->update_fresh_only_controls();

    /* Save the reset configuration while preserving the existing corpus list. */
    minima_settings_save(context);
}

void minima_settings_save(const MinimaSettingsContext* context) {
    if (!context || !context->path || !context->path[0]) return;
    for (int i = 0; i < 24; ++i)
        save_edit_setting(context->path, g_edit_keys[i], context->edit_controls[i]);
    for (int i = 0; i < 3; ++i)
        save_combo_setting(context->path, g_combo_keys[i], context->combo_controls[i]);

    WritePrivateProfileStringW(L"UI", L"advanced_expanded",
        context->advanced_expanded && *context->advanced_expanded ? L"1" : L"0",
        context->path);
    WritePrivateProfileStringW(L"Training", L"resume",
        SendMessageW(context->resume_control, BM_GETCHECK, 0, 0) == BST_CHECKED ? L"1" : L"0",
        context->path);
    WritePrivateProfileStringW(L"Training", L"prebuilt_tokenizer",
        context->use_saved_tokenizer && *context->use_saved_tokenizer ? L"1" : L"0",
        context->path);

    int count = (int)SendMessageW(context->corpus_control, LB_GETCOUNT, 0, 0);
    if (count > 32) count = 32;
    wchar_t number[16];
    _snwprintf_s(number, _countof(number), _TRUNCATE, L"%d", count);
    WritePrivateProfileStringW(L"Corpus", L"count", number, context->path);
    for (int i = 0; i < count; ++i) {
        wchar_t key[32], file_path[MAX_PATH * 2];
        _snwprintf_s(key, _countof(key), _TRUNCATE, L"file%d", i);
        SendMessageW(context->corpus_control, LB_GETTEXT, (WPARAM)i, (LPARAM)file_path);
        WritePrivateProfileStringW(L"Corpus", key, file_path, context->path);
    }
}

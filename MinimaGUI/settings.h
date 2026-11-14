#ifndef MINIMA_GUI_SETTINGS_H
#define MINIMA_GUI_SETTINGS_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct MinimaSettingsContext {
    const wchar_t* path;
    HWND edit_controls[24];
    HWND combo_controls[3];
    HWND resume_control;
    HWND corpus_control;
    HWND window;
    BOOL* use_saved_tokenizer;
    BOOL* advanced_expanded;
    wchar_t* prebuilt_vocab;
    wchar_t* prebuilt_tokenizer_bytes;
    wchar_t* prebuilt_sampling;
    void (*set_advanced_expanded)(HWND, BOOL);
    void (*update_fresh_only_controls)(void);
} MinimaSettingsContext;

void minima_settings_load(const MinimaSettingsContext* context);
void minima_settings_reset_defaults(const MinimaSettingsContext* context);
void minima_settings_save(const MinimaSettingsContext* context);

#endif

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <stdio.h>
#include <wchar.h>
#include <string.h>
#include "graph.h"

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

static wchar_t g_metrics_path[MAX_PATH];
static UINT g_graph_dpi = 96;
static HPEN g_graph_grid_pen, g_graph_train_pen, g_graph_valid_pen;
static HDC g_graph_back_dc;
static HBITMAP g_graph_back_bitmap;
static HGDIOBJ g_graph_back_old_bitmap;
static int g_graph_back_width, g_graph_back_height;
static float g_graph_train[512], g_graph_valid[512];
static int g_graph_count;

static int graph_scale_px(int value) {
    return MulDiv(value, (int)(g_graph_dpi ? g_graph_dpi : 96), 96);
}

void minima_graph_set_metrics_path(const wchar_t* path) {
    if (!path) { g_metrics_path[0] = L'\0'; return; }
    wcsncpy_s(g_metrics_path, _countof(g_metrics_path), path, _TRUNCATE);
}

void minima_graph_set_dpi(UINT dpi) {
    g_graph_dpi = dpi ? dpi : 96;
}

void minima_graph_refresh(void) {
    g_graph_count = 0;
    FILE* file = _wfopen(g_metrics_path, L"rt");
    if (!file) return;

    char line[512];
    if (fseek(file, 0, SEEK_END) == 0) {
        long end = ftell(file);
        long start = end > 262144 ? end - 262144 : 0;
        if (start > 0) {
            fseek(file, start, SEEK_SET);
            (void)fgets(line, sizeof(line), file);
        } else {
            rewind(file);
            (void)fgets(line, sizeof(line), file);
        }
    }
    while (fgets(line, sizeof(line), file)) {
        double step, train, valid, lr, train_ppl, valid_ppl, elapsed;
        if (sscanf_s(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &step, &train, &valid, &lr, &train_ppl, &valid_ppl, &elapsed) == 7) {
            if (g_graph_count == 512) {
                memmove(g_graph_train, g_graph_train + 1, 511 * sizeof(float));
                memmove(g_graph_valid, g_graph_valid + 1, 511 * sizeof(float));
                g_graph_count = 511;
            }
            g_graph_train[g_graph_count] = (float)train;
            g_graph_valid[g_graph_count] = (float)valid;
            ++g_graph_count;
        }
    }
    fclose(file);
}

static void release_graph_backbuffer(void) {
    if (g_graph_back_dc && g_graph_back_old_bitmap)
        SelectObject(g_graph_back_dc, g_graph_back_old_bitmap);
    if (g_graph_back_bitmap) {
        DeleteObject(g_graph_back_bitmap);
        g_graph_back_bitmap = NULL;
    }
    if (g_graph_back_dc) {
        DeleteDC(g_graph_back_dc);
        g_graph_back_dc = NULL;
    }
    g_graph_back_old_bitmap = NULL;
    g_graph_back_width = 0;
    g_graph_back_height = 0;
}

static BOOL ensure_graph_backbuffer(HDC reference, int width, int height) {
    if (width <= 0 || height <= 0) return FALSE;
    if (!g_graph_back_dc) {
        g_graph_back_dc = CreateCompatibleDC(reference);
        if (!g_graph_back_dc) return FALSE;
    }
    if (g_graph_back_bitmap && width == g_graph_back_width &&
        height == g_graph_back_height)
        return TRUE;

    if (g_graph_back_bitmap) {
        SelectObject(g_graph_back_dc, g_graph_back_old_bitmap);
        DeleteObject(g_graph_back_bitmap);
        g_graph_back_bitmap = NULL;
        g_graph_back_old_bitmap = NULL;
    }

    HBITMAP bitmap = CreateCompatibleBitmap(reference, width, height);
    if (!bitmap) return FALSE;
    HGDIOBJ previous = SelectObject(g_graph_back_dc, bitmap);
    if (!previous || previous == HGDI_ERROR) {
        DeleteObject(bitmap);
        return FALSE;
    }
    g_graph_back_bitmap = bitmap;
    g_graph_back_old_bitmap = previous;
    g_graph_back_width = width;
    g_graph_back_height = height;
    return TRUE;
}

LRESULT CALLBACK minima_graph_window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    (void)wp; (void)lp;
    if (msg == WM_CREATE) {
        g_graph_grid_pen = CreatePen(PS_SOLID, 1, RGB(220,220,220));
        g_graph_train_pen = CreatePen(PS_SOLID, 2, RGB(35,105,190));
        g_graph_valid_pen = CreatePen(PS_SOLID, 2, RGB(190,85,45));
        return 0;
    }
    if (msg == WM_DESTROY) {
        release_graph_backbuffer();
        if (g_graph_grid_pen) { DeleteObject(g_graph_grid_pen); g_graph_grid_pen = NULL; }
        if (g_graph_train_pen) { DeleteObject(g_graph_train_pen); g_graph_train_pen = NULL; }
        if (g_graph_valid_pen) { DeleteObject(g_graph_valid_pen); g_graph_valid_pen = NULL; }
        return 0;
    }
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        RECT rect;
        GetClientRect(hwnd, &rect);
        const int width = rect.right - rect.left;
        const int height = rect.bottom - rect.top;
        if (ensure_graph_backbuffer(dc, width, height)) {
            HDC back = g_graph_back_dc;
            FillRect(back, &rect, (HBRUSH)(COLOR_WINDOW + 1));
            SetBkMode(back, TRANSPARENT);
            SetTextColor(back, RGB(40,40,40));
            TextOutW(back, graph_scale_px(8), graph_scale_px(5), L"Training / validation loss", 26);

            const int left = graph_scale_px(58), top = graph_scale_px(30);
            const int right = width - graph_scale_px(12), bottom = height - graph_scale_px(27);
            float low = 0.0f, high = 1.0f;
            if (g_graph_count > 0) {
                low = g_graph_train[0];
                high = g_graph_train[0];
                for (int i = 0; i < g_graph_count; ++i) {
                    if (g_graph_train[i] < low) low = g_graph_train[i];
                    if (g_graph_valid[i] < low) low = g_graph_valid[i];
                    if (g_graph_train[i] > high) high = g_graph_train[i];
                    if (g_graph_valid[i] > high) high = g_graph_valid[i];
                }
                if (high - low < 0.001f) high = low + 0.001f;
            }

            if (g_graph_grid_pen) SelectObject(back, g_graph_grid_pen);
            for (int i = 0; i <= 4; ++i) {
                const int y = top + (bottom - top) * i / 4;
                MoveToEx(back, left, y, NULL);
                LineTo(back, right, y);
                wchar_t label[24];
                const float value = high - (high - low) * (float)i / 4.0f;
                swprintf_s(label, _countof(label), L"%.2f", value);
                TextOutW(back, graph_scale_px(3), y - graph_scale_px(7), label, (int)wcslen(label));
            }

            if (g_graph_count > 1) {
                for (int series = 0; series < 2; ++series) {
                    SelectObject(back, series ? g_graph_valid_pen : g_graph_train_pen);
                    for (int i = 0; i < g_graph_count; ++i) {
                        const float value = series ? g_graph_valid[i] : g_graph_train[i];
                        const int x = left + (right - left) * i / (g_graph_count - 1);
                        const int y = bottom - (int)((value - low) / (high - low) * (bottom - top));
                        if (i == 0) MoveToEx(back, x, y, NULL);
                        else LineTo(back, x, y);
                    }
                }
                wchar_t legend[96];
                swprintf_s(legend, _countof(legend), L"Train %.3f   Valid %.3f",
                    g_graph_train[g_graph_count - 1], g_graph_valid[g_graph_count - 1]);
                SetTextColor(back, RGB(35,105,190));
                TextOutW(back, left, bottom + graph_scale_px(5), legend, (int)wcslen(legend));
            } else {
                TextOutW(back, left, top + graph_scale_px(35), L"Waiting for training metrics...", 31);
            }
            BitBlt(dc, 0, 0, width, height, back, 0, 0, SRCCOPY);
        } else {
            FillRect(dc, &rect, (HBRUSH)(COLOR_WINDOW + 1));
        }
        EndPaint(hwnd, &ps);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}


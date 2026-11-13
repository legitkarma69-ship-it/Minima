#include "util.h"
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// -------------------- RNG --------------------

void rng_seed(Rng* r, uint64_t seed) {
    r->state = seed ? seed : 0x123456789abcdef0ULL;
}

uint64_t rng_u64(Rng* r) {
    uint64_t x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float rng_f32(Rng* r) {
    // 24 bits of randomness → [0, 1)
    return (rng_u64(r) >> 40) * (1.0f / 16777216.0f);
}

int rng_int(Rng* r, int lo, int hi) {
    if (lo >= hi) return lo;
    uint64_t range = (uint64_t)(hi - lo + 1);
    return lo + (int)(rng_u64(r) % range);
}

// -------------------- File --------------------

bool file_exists(const char* path) {
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES &&
        !(attr & FILE_ATTRIBUTE_DIRECTORY));
}

size_t file_size(const char* path) {
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad))
        return 0;
    ULARGE_INTEGER ul;
    ul.HighPart = fad.nFileSizeHigh;
    ul.LowPart = fad.nFileSizeLow;
    return (size_t)ul.QuadPart;
}

bool read_entire_file(const char* path, void* buf, size_t buf_size, size_t* out_read) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(buf, 1, buf_size, f);
    fclose(f);
    if (out_read) *out_read = n;
    return true;
}

// -------------------- Timing --------------------

double now_seconds(void) {
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

// -------------------- Fatal --------------------

void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

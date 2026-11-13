#include "mapped_buffer.h"

#include <string.h>
#include <stdint.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

bool mapped_buffer_create(MappedBuffer* buffer, size_t size) {
    if (!buffer || size == 0) return false;
    memset(buffer, 0, sizeof(*buffer));

    char directory[MAX_PATH];
    char path[MAX_PATH];
    const DWORD directory_length = GetTempPathA(MAX_PATH, directory);
    if (directory_length == 0 || directory_length >= MAX_PATH ||
        !GetTempFileNameA(directory, "min", 0, path))
        return false;

    HANDLE file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        DeleteFileA(path);
        return false;
    }

    LARGE_INTEGER length;
    length.QuadPart = (LONGLONG)size;
    if (length.QuadPart <= 0 ||
        !SetFilePointerEx(file, length, NULL, FILE_BEGIN) ||
        !SetEndOfFile(file)) {
        CloseHandle(file);
        DeleteFileA(path);
        return false;
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READWRITE,
        (DWORD)(((uint64_t)size) >> 32), (DWORD)((uint64_t)size), NULL);
    if (!mapping) {
        CloseHandle(file);
        DeleteFileA(path);
        return false;
    }

    void* view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!view) {
        CloseHandle(mapping);
        CloseHandle(file);
        DeleteFileA(path);
        return false;
    }
    buffer->data = view;
    buffer->size = size;
    buffer->file_handle = file;
    buffer->mapping_handle = mapping;
    return true;
}

void mapped_buffer_destroy(MappedBuffer* buffer) {
    if (!buffer) return;
    if (buffer->data) UnmapViewOfFile(buffer->data);
    if (buffer->mapping_handle)
        CloseHandle((HANDLE)buffer->mapping_handle);
    if (buffer->file_handle)
        CloseHandle((HANDLE)buffer->file_handle);
    memset(buffer, 0, sizeof(*buffer));
}

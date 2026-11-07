#include "mapped_buffer.h"

#include <string.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <stdlib.h>
#endif

bool mapped_buffer_create(MappedBuffer* buffer, size_t size) {
    if (!buffer || size == 0) return false;
    memset(buffer, 0, sizeof(*buffer));
#ifndef _WIN32
    buffer->file_descriptor = -1;
#endif

#ifdef _WIN32
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
#else
    if (size > (size_t)LLONG_MAX) return false;
    char path[] = "/tmp/minima-token-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return false;
    (void)unlink(path);
    if (ftruncate(fd, (off_t)size) != 0) {
        close(fd);
        return false;
    }
    void* view = mmap(NULL, size, PROT_READ | PROT_WRITE,
        MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) {
        close(fd);
        return false;
    }
    buffer->data = view;
    buffer->size = size;
    buffer->file_descriptor = fd;
    return true;
#endif
}

void mapped_buffer_destroy(MappedBuffer* buffer) {
    if (!buffer) return;
#ifdef _WIN32
    if (buffer->data) UnmapViewOfFile(buffer->data);
    if (buffer->mapping_handle)
        CloseHandle((HANDLE)buffer->mapping_handle);
    if (buffer->file_handle)
        CloseHandle((HANDLE)buffer->file_handle);
#else
    if (buffer->data && buffer->size)
        (void)munmap(buffer->data, buffer->size);
    if (buffer->file_descriptor >= 0)
        (void)close(buffer->file_descriptor);
#endif
    memset(buffer, 0, sizeof(*buffer));
#ifndef _WIN32
    buffer->file_descriptor = -1;
#endif
}

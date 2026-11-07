#ifndef MAPPED_BUFFER_H
#define MAPPED_BUFFER_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    void* data;
    size_t size;
#ifdef _WIN32
    void* file_handle;
    void* mapping_handle;
#else
    int file_descriptor;
#endif
} MappedBuffer;

/* Create a temporary, disk-backed memory mapping. The OS pages data in/out as
   needed, allowing large token arrays without permanently pinning all pages. */
bool mapped_buffer_create(MappedBuffer* buffer, size_t size);
void mapped_buffer_destroy(MappedBuffer* buffer);

#endif

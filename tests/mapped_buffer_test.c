#include "mapped_buffer.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const size_t size = 1024u * 1024u + 37u;
    MappedBuffer buffer;
    if (!mapped_buffer_create(&buffer, size)) {
        fprintf(stderr, "mapped buffer creation failed\n");
        return 1;
    }
    uint8_t* bytes = (uint8_t*)buffer.data;
    const size_t positions[] = { 0, 4095, 65536, size - 1 };
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++)
        bytes[positions[i]] = (uint8_t)(i * 53u + 7u);
    int failed = buffer.size != size;
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++)
        if (bytes[positions[i]] != (uint8_t)(i * 53u + 7u))
            failed = 1;
    mapped_buffer_destroy(&buffer);
    if (buffer.data != NULL || buffer.size != 0) failed = 1;
    if (failed) {
        fprintf(stderr, "mapped buffer regression failed\n");
        return 1;
    }
    printf("Disk-backed mapped buffer: PASS\n");
    return 0;
}

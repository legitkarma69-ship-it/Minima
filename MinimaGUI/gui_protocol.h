#ifndef MINIMA_GUI_PROTOCOL_H
#define MINIMA_GUI_PROTOCOL_H

#include <stddef.h>
#include <string.h>

enum {
    MINIMA_PROTOCOL_NONE = 0,
    MINIMA_PROTOCOL_CHAT_READY = 1,
    MINIMA_PROTOCOL_CHAT_CLOSED = 2
};

/* Recognize protocol lines with either LF or Windows CRLF termination. */
static int minima_protocol_marker(const char* line, size_t length) {
    if (!line) return MINIMA_PROTOCOL_NONE;
    if (length && line[length - 1] == '\n') --length;
    if (length && line[length - 1] == '\r') --length;

    if (length == sizeof("CHAT_READY") - 1 &&
        memcmp(line, "CHAT_READY", sizeof("CHAT_READY") - 1) == 0)
        return MINIMA_PROTOCOL_CHAT_READY;
    if (length == sizeof("CHAT_CLOSED") - 1 &&
        memcmp(line, "CHAT_CLOSED", sizeof("CHAT_CLOSED") - 1) == 0)
        return MINIMA_PROTOCOL_CHAT_CLOSED;
    return MINIMA_PROTOCOL_NONE;
}

typedef void (*MinimaProtocolReadyCallback)(void* context);

/* Dispatch the readiness marker through the callback used by the GUI reader. */
static int minima_protocol_dispatch(const char* line, size_t length,
    MinimaProtocolReadyCallback ready_callback, void* context) {
    const int marker = minima_protocol_marker(line, length);
    if (marker == MINIMA_PROTOCOL_CHAT_READY && ready_callback)
        ready_callback(context);
    return marker;
}

#endif

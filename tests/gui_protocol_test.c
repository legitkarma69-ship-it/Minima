#include "../MinimaGUI/gui_protocol.h"

#include <stdio.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL: %s (line %d)\n", #expr, __LINE__); \
    return 1; \
} } while (0)

int main(void) {
    CHECK(minima_protocol_marker("CHAT_READY\n", sizeof("CHAT_READY\n") - 1) ==
        MINIMA_PROTOCOL_CHAT_READY);
    CHECK(minima_protocol_marker("CHAT_READY\r\n", sizeof("CHAT_READY\r\n") - 1) ==
        MINIMA_PROTOCOL_CHAT_READY);
    CHECK(minima_protocol_marker("CHAT_CLOSED\n", sizeof("CHAT_CLOSED\n") - 1) ==
        MINIMA_PROTOCOL_CHAT_CLOSED);
    CHECK(minima_protocol_marker("CHAT_CLOSED\r\n", sizeof("CHAT_CLOSED\r\n") - 1) ==
        MINIMA_PROTOCOL_CHAT_CLOSED);
    CHECK(minima_protocol_marker("CHAT_READY_EXTRA\r\n",
        sizeof("CHAT_READY_EXTRA\r\n") - 1) == MINIMA_PROTOCOL_NONE);
    CHECK(minima_protocol_marker("CHAT_READY", sizeof("CHAT_READY") - 1) ==
        MINIMA_PROTOCOL_CHAT_READY);
    puts("GUI protocol LF/CRLF tests: PASS");
    return 0;
}

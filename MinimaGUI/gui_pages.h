#ifndef MINIMA_GUI_PAGES_H
#define MINIMA_GUI_PAGES_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef struct {
    HWND* corpus;
    HWND* steps;
    HWND* resume;
    HWND* vocab;
    HWND* dim;
    HWND* layers;
    HWND* heads;
    HWND* ffn;
    HWND* max_seq;
    HWND* batch_size;
    HWND* sequence_length;
    HWND* lr;
    HWND* warmup;
    HWND* log_interval;
    HWND* validation_interval;
    HWND* validation_batches;
    HWND* checkpoint_interval;
    HWND* seed;
    HWND* tokenizer_bytes;
    HWND* temperature;
    HWND* top_k;
    HWND* top_p;
    HWND* repetition;
    HWND* max_tokens;
    HWND* lr_schedule;
    HWND* tokenizer_sampling;
    HWND* total_steps;
    HWND* status;
    HWND* graph;
    HWND* output_label;
    HWND* log;
    HWND* chat_input;
    HWND* chat_test;
} GuiPageControls;

void gui_pages_create_content(HWND parent, GuiPageControls* page_controls,
    void (*register_advanced)(HWND));
void gui_pages_create_shell(HWND parent, HWND* tabs, HWND* corpus_group,
    HWND* model_group, HWND* training_group, HWND* advanced_toggle,
    HWND* monitor_group, HWND* output_group);
#endif

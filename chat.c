#include "chat.h"

#include "generate.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

int chat_run_console(Model* model, Tokenizer* tok, KVCache* cache,
    const ChatOptions* options) {
    uint16_t history[512], input_ids[8192], prompt[512], output[512];
    int history_len=0;
    char line[8192], decoded[512* TOK_MAX_TOKEN_LEN+1];
    puts("CHAT_READY");
    puts("ProjectMinima chat. Type /quit to exit.");
    for(;;){
        if(!fgets(line,sizeof(line),stdin))break;
        size_t bytes=strlen(line);while(bytes && (line[bytes-1]=='\n'||line[bytes-1]=='\r'))line[--bytes]=0;
        if(strcmp(line,"/quit")==0||strcmp(line,"/exit")==0)break;
        if(!bytes){puts("CHAT_READY");continue;}
        int input_count=tokenizer_encode(tok,(const uint8_t*)line,bytes,input_ids,(int)(sizeof(input_ids)/sizeof(input_ids[0])));
        if(input_count<=0){puts("BOT: [input encoding failed]");puts("CHAT_READY");continue;}
        if(input_count>=model->max_seq){memmove(input_ids,input_ids+(input_count-(model->max_seq-1)),(size_t)(model->max_seq-1)*sizeof(uint16_t));input_count=model->max_seq-1;}
        uint16_t separator[4];
        int separator_count=history_len
            ? tokenizer_encode(tok,(const uint8_t*)"\n",1,separator,4) : 0;
        int prompt_len=history_len+separator_count+input_count;
        if(prompt_len>model->max_seq-1){
            int drop=prompt_len-(model->max_seq-1);
            if(drop<history_len){memmove(history,history+drop,(size_t)(history_len-drop)*sizeof(uint16_t));history_len-=drop;}
            else {history_len=0;separator_count=0;}
            prompt_len=history_len+separator_count+input_count;
        }
        memcpy(prompt,history,(size_t)history_len*sizeof(uint16_t));
        if(separator_count)memcpy(prompt+history_len,separator,(size_t)separator_count*sizeof(uint16_t));
        memcpy(prompt+history_len+separator_count,input_ids,(size_t)input_count*sizeof(uint16_t));
        Rng rng; rng_seed(&rng,((uint64_t)clock()<<17)^(uint64_t)options->seed^(uint64_t)history_len);
        int max_new=options->max_new_tokens;
        if(max_new>model->max_seq-prompt_len)max_new=model->max_seq-prompt_len;
        int total=generate_ex(model,tok,cache,prompt,prompt_len,output,512,max_new,
            options->temperature,options->top_k,options->top_p,options->repetition_penalty,&rng);
        if(total<=prompt_len){puts("BOT: [generation failed]");puts("CHAT_READY");continue;}
        int decoded_bytes=tokenizer_decode(tok,output+prompt_len,total-prompt_len,decoded,(int)sizeof(decoded));
        if(decoded_bytes<0)decoded[0]=0;
        printf("BOT: %s\n",decoded);
        history_len=total;
        if(history_len>model->max_seq-1){int drop=history_len-(model->max_seq-1);memmove(history,output+drop,(size_t)(history_len-drop)*sizeof(uint16_t));history_len-=drop;}
        else memcpy(history,output,(size_t)history_len*sizeof(uint16_t));
        puts("CHAT_READY");
    }
    puts("CHAT_CLOSED");
    return 0;
}

static void print_repetition_stats(const uint16_t* ids,
    int prompt_len, int total_len) {
    const int n = total_len - prompt_len;
    if (!ids || n <= 0) return;

    int unique = 0;
    int adjacent_repeats = 0;
    for (int i = 0; i < n; i++) {
        bool seen = false;
        for (int j = 0; j < i; j++) {
            if (ids[prompt_len + i] == ids[prompt_len + j]) {
                seen = true;
                break;
            }
        }
        if (!seen) unique++;
        if (i > 0 && ids[prompt_len + i] == ids[prompt_len + i - 1])
            adjacent_repeats++;
    }

    int frequent_ids[5] = { -1, -1, -1, -1, -1 };
    int frequent_counts[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        const uint16_t id = ids[prompt_len + i];
        bool counted = false;
        for (int j = 0; j < i; j++) {
            if (ids[prompt_len + j] == id) {
                counted = true;
                break;
            }
        }
        if (counted) continue;

        int count = 0;
        for (int j = i; j < n; j++)
            if (ids[prompt_len + j] == id) count++;

        for (int slot = 0; slot < 5; slot++) {
            if (count > frequent_counts[slot]) {
                for (int move = 4; move > slot; move--) {
                    frequent_counts[move] = frequent_counts[move - 1];
                    frequent_ids[move] = frequent_ids[move - 1];
                }
                frequent_counts[slot] = count;
                frequent_ids[slot] = id;
                break;
            }
        }
    }

    printf("      most frequent token IDs:");
    for (int i = 0; i < 5 && frequent_ids[i] >= 0; i++) {
        printf(" %d=%d(%.1f%%)", frequent_ids[i], frequent_counts[i],
            100.0 * (double)frequent_counts[i] / (double)n);
    }
    putchar('\n');

    int repeated_ngram_len = 0;
    int repeated_ngram_count = 1;
    for (int width = 2; width <= 4 && width <= n; width++) {
        for (int i = 0; i + width <= n; i++) {
            int count = 1;
            for (int j = i + 1; j + width <= n; j++) {
                bool equal = true;
                for (int k = 0; k < width; k++) {
                    if (ids[prompt_len + i + k] !=
                        ids[prompt_len + j + k]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) count++;
            }
            if (count > repeated_ngram_count) {
                repeated_ngram_count = count;
                repeated_ngram_len = width;
            }
        }
    }

    printf("      repetition: unique=%d/%d, adjacent-identical=%d, ",
        unique, n, adjacent_repeats);
    if (repeated_ngram_len > 0)
        printf("most-repeated %d-token pattern occurs %d times\n",
            repeated_ngram_len, repeated_ngram_count);
    else
        printf("no repeated 2-4 token pattern\n");
}

bool chat_run_generation_benchmark(Model* model, Tokenizer* tok,
    KVCache* cache) {
    static const char* prompts[] = {
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "The meaning of life",
        "It was a dark"
    };

    printf("\nGeneration diagnostics\n");
    printf("  Greedy decoding is tested for repeatability; sampled modes use fixed seeds.\n");
    uint16_t prompt_ids[64];
    uint16_t greedy_ids[64];
    uint16_t verify_ids[64];
    uint16_t sample_ids[64];
    char output_text[64 * TOK_MAX_TOKEN_LEN + 1];

    for (size_t i = 0; i < sizeof(prompts) / sizeof(prompts[0]); i++) {
        const size_t prompt_bytes = strlen(prompts[i]);
        const int prompt_len = tokenizer_encode(tok,
            (const uint8_t*)prompts[i], prompt_bytes,
            prompt_ids, (int)(sizeof(prompt_ids) / sizeof(prompt_ids[0])));
        if (prompt_len <= 0) {
            printf("  [%zu] prompt encoding failed\n", i + 1);
            return false;
        }

        printf("\n  [%zu] Prompt: \"%s\"\n", i + 1, prompts[i]);
        if (!generation_inspect_next_tokens(model, tok, cache,
            prompt_ids, prompt_len, 10)) {
            printf("  next-token inspection failed\n");
            return false;
        }

        // Greedy run and exact deterministic repeat.
        Rng greedy_rng;
        rng_seed(&greedy_rng, 5678u + (uint64_t)i);
        const int greedy_len = generate(model, tok, cache,
            prompt_ids, prompt_len, greedy_ids, 64,
            32, 1.0f, 1, &greedy_rng);
        if (greedy_len <= prompt_len) return false;

        Rng verify_rng;
        rng_seed(&verify_rng, 5678u + (uint64_t)i);
        const int verify_len = generate(model, tok, cache,
            prompt_ids, prompt_len, verify_ids, 64,
            32, 1.0f, 1, &verify_rng);
        if (verify_len != greedy_len ||
            memcmp(greedy_ids, verify_ids,
                (size_t)greedy_len * sizeof(uint16_t)) != 0) {
            printf("  deterministic greedy check: FAIL\n");
            return false;
        }

        int bytes = tokenizer_decode(tok, greedy_ids,
            greedy_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    greedy (T=1.0, k=1): %s\n", output_text);
        print_repetition_stats(greedy_ids, prompt_len, greedy_len);

        // Lower-temperature top-k sampling.
        Rng sample_rng;
        rng_seed(&sample_rng, 9100u + (uint64_t)i);
        int sample_len = generate(model, tok, cache,
            prompt_ids, prompt_len, sample_ids, 64,
            32, 0.8f, 40, &sample_rng);
        if (sample_len <= prompt_len) return false;
        bytes = tokenizer_decode(tok, sample_ids,
            sample_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    sampled (T=0.8, k=40): %s\n", output_text);
        print_repetition_stats(sample_ids, prompt_len, sample_len);

        // Slightly warmer sampling shows whether diversity improves or
        // merely turns repetition into noisy token selection.
        rng_seed(&sample_rng, 12100u + (uint64_t)i);
        sample_len = generate(model, tok, cache,
            prompt_ids, prompt_len, sample_ids, 64,
            32, 1.0f, 40, &sample_rng);
        if (sample_len <= prompt_len) return false;
        bytes = tokenizer_decode(tok, sample_ids,
            sample_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    sampled (T=1.0, k=40): %s\n", output_text);
        print_repetition_stats(sample_ids, prompt_len, sample_len);

        rng_seed(&sample_rng,15100u+(uint64_t)i);
        sample_len=generate_ex(model,tok,cache,prompt_ids,prompt_len,sample_ids,64,
            32,0.8f,40,0.9f,1.1f,&sample_rng);
        if(sample_len<=prompt_len)return false;
        bytes=tokenizer_decode(tok,sample_ids,sample_len,output_text,(int)sizeof(output_text));
        if(bytes<=0)return false;
        printf("    sampled (T=0.8, k=40, p=0.9, repetition=1.1): %s\n",output_text);
        print_repetition_stats(sample_ids,prompt_len,sample_len);
    }

    printf("\nDeterministic greedy check: PASS (all prompts reproduced exactly)\n");
    return true;
}

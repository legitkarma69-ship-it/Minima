#include "diagnostics.h"

#include "forward.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIAG_TOP_LIMIT 100
#define DIAG_EVAL_WINDOWS 16
#define DIAG_BUCKETS 5
#define DIAG_GENERATION_TOKENS 32

typedef struct {
    uint16_t id;
    uint64_t count;
} TokenCount;

typedef struct {
    const char* name;
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
} GenerationMode;

typedef struct {
    double entropy_bits;
    double distinct[3];
    double adjacent_repeat_rate;
    int unique_tokens;
    int max_repeated_ngram;
} GenerationStats;

static const char* const frequency_bucket_names[DIAG_BUCKETS] = {
    "top_1pct", "top_10pct", "top_50pct", "tail", "unseen"
};

static int compare_token_count(const void* a, const void* b) {
    const TokenCount* left = (const TokenCount*)a;
    const TokenCount* right = (const TokenCount*)b;
    if (left->count < right->count) return 1;
    if (left->count > right->count) return -1;
    return (int)left->id - (int)right->id;
}

static void write_csv_string(FILE* file, const char* text) {
    fputc('"', file);
    for (const unsigned char* p = (const unsigned char*)text; *p; ++p) {
        if (*p == '"') fputs("\"\"", file);
        else if (*p == '\\') fputs("\\\\", file);
        else if (*p == '\n') fputs("\\n", file);
        else if (*p == '\r') fputs("\\r", file);
        else if (*p == '\t') fputs("\\t", file);
        else if (*p >= 32 && *p <= 126) fputc(*p, file);
        else fprintf(file, "\\x%02X", (unsigned int)*p);
    }
    fputc('"', file);
}

static void write_csv_token(FILE* file, const char* token, int length) {
    fputc('"', file);
    for (int i = 0; i < length; ++i) {
        const unsigned char byte = (unsigned char)token[i];
        if (byte == '"') fputs("\"\"", file);
        else if (byte == '\\') fputs("\\\\", file);
        else if (byte == '\n') fputs("\\n", file);
        else if (byte == '\r') fputs("\\r", file);
        else if (byte == '\t') fputs("\\t", file);
        else if (byte >= 32 && byte <= 126) fputc(byte, file);
        else fprintf(file, "\\x%02X", (unsigned int)byte);
    }
    fputc('"', file);
}

static bool token_has_whitespace(const Tokenizer* tok, int id) {
    const unsigned char* s = (const unsigned char*)tok->token_str[id];
    for (int i = 0; i < tok->token_len[id]; ++i) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
            s[i] == '\r' || s[i] == '\v' || s[i] == '\f')
            return true;
    }
    return false;
}

static const char* token_category(int id) {
    if (id < 4) return "special";
    if (id < 260) return "byte";
    return "merge";
}

static int frequency_bucket(uint16_t id, const uint64_t* train_counts,
    const int* rank_by_id, int active_tokens) {
    if (id >= TOK_MAX_VOCAB || train_counts[id] == 0)
        return DIAG_BUCKETS - 1;

    const int rank = rank_by_id[id];
    const int top_one = (active_tokens + 99) / 100;
    const int top_ten = (active_tokens + 9) / 10;
    const int top_fifty = (active_tokens + 1) / 2;
    if (rank < top_one) return 0;
    if (rank < top_ten) return 1;
    if (rank < top_fifty) return 2;
    return 3;
}

static bool tokenizer_report(const Tokenizer* tok, Arena* arena,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    uint64_t** out_train_counts, int** out_rank_by_id, int* out_active) {
    if (!tok || !arena || !all_ids || train_tokens <= 0 ||
        validation_tokens <= 0 || tok->vocab_size < 260 ||
        tok->vocab_size > TOK_MAX_VOCAB)
        return false;

    const size_t vocab = (size_t)tok->vocab_size;
    uint64_t* train_counts = (uint64_t*)arena_alloc_zero(arena,
        vocab * sizeof(uint64_t));
    uint64_t* validation_counts = (uint64_t*)arena_alloc_zero(arena,
        vocab * sizeof(uint64_t));
    int* rank_by_id = (int*)arena_alloc(arena, vocab * sizeof(int));
    TokenCount* ranked = (TokenCount*)arena_alloc(arena,
        vocab * sizeof(TokenCount));
    if (!train_counts || !validation_counts || !rank_by_id || !ranked)
        return false;

    for (int i = 0; i < train_tokens; ++i) {
        const uint16_t id = all_ids[i];
        if (id >= tok->vocab_size) return false;
        train_counts[id]++;
    }
    for (int i = 0; i < validation_tokens; ++i) {
        const uint16_t id = all_ids[train_tokens + i];
        if (id >= tok->vocab_size) return false;
        validation_counts[id]++;
    }

    int active = 0;
    for (int id = 0; id < tok->vocab_size; ++id) {
        rank_by_id[id] = -1;
        if (train_counts[id] > 0) {
            ranked[active].id = (uint16_t)id;
            ranked[active].count = train_counts[id];
            active++;
        }
    }
    qsort(ranked, (size_t)active, sizeof(TokenCount), compare_token_count);
    for (int rank = 0; rank < active; ++rank)
        rank_by_id[ranked[rank].id] = rank;

    uint64_t train_total = (uint64_t)train_tokens;
    uint64_t validation_total = (uint64_t)validation_tokens;
    double js = 0.0;
    for (int id = 0; id < tok->vocab_size; ++id) {
        const double p = (double)train_counts[id] / (double)train_total;
        const double q = (double)validation_counts[id] / (double)validation_total;
        const double m = 0.5 * (p + q);
        if (p > 0.0) js += 0.5 * p * log(p / m) / log(2.0);
        if (q > 0.0) js += 0.5 * q * log(q / m) / log(2.0);
    }

    uint64_t raw_byte_count = 0;
    uint64_t used_merge_count = 0;
    uint64_t whitespace_token_count = 0;
    uint64_t leading_space_count = 0;
    uint64_t newline_token_count = 0;
    uint64_t represented_bytes = 0;
    int unused_merge_tokens = 0;
    int max_token_bytes = 0;
    for (int id = 0; id < tok->vocab_size; ++id) {
        if (tok->token_len[id] > max_token_bytes)
            max_token_bytes = tok->token_len[id];
        represented_bytes += (uint64_t)tok->token_len[id] * train_counts[id];
        if (id >= 4 && id < 260) raw_byte_count += train_counts[id];
        if (id >= 260) {
            if (train_counts[id]) used_merge_count++;
            else unused_merge_tokens++;
        }
        if (id >= 4 && token_has_whitespace(tok, id)) {
            whitespace_token_count += train_counts[id];
            const unsigned char first = (unsigned char)tok->token_str[id][0];
            if (first == ' ' || first == '\t') leading_space_count += train_counts[id];
            for (int j = 0; j < tok->token_len[id]; ++j)
                if (tok->token_str[id][j] == '\n') {
                    newline_token_count += train_counts[id];
                    break;
                }
        }
    }

    uint64_t top_mass[3] = {0, 0, 0};
    const int limits[3] = {10, 50, 100};
    for (int rank = 0; rank < active; ++rank) {
        for (int k = 0; k < 3; ++k)
            if (rank < limits[k]) top_mass[k] += ranked[rank].count;
    }

    printf("\nTokenizer and corpus diagnostics\n");
    printf("  vocabulary / merges        : %d / %d\n", tok->vocab_size, tok->n_merges);
    printf("  active tokens in train     : %d / %d\n", active, tok->vocab_size);
    printf("  unused merge tokens        : %d\n", unused_merge_tokens);
    printf("  mean bytes per train token : %.3f\n",
        (double)represented_bytes / (double)train_tokens);
    printf("  maximum token byte length : %d\n", max_token_bytes);
    printf("  raw-byte fallback share    : %.2f%%\n",
        100.0 * (double)raw_byte_count / (double)train_tokens);
    printf("  whitespace token share    : %.2f%% (leading-space %.2f%%, newline-containing %.2f%%)\n",
        100.0 * (double)whitespace_token_count / (double)train_tokens,
        100.0 * (double)leading_space_count / (double)train_tokens,
        100.0 * (double)newline_token_count / (double)train_tokens);
    printf("  train/validation JS div.   : %.6f bits (0 = identical distributions)\n", js);
    printf("  top 10 / 50 / 100 token mass: %.2f%% / %.2f%% / %.2f%%\n",
        100.0 * (double)top_mass[0] / (double)train_tokens,
        100.0 * (double)top_mass[1] / (double)train_tokens,
        100.0 * (double)top_mass[2] / (double)train_tokens);

    FILE* csv = fopen("minima_token_frequency.csv", "wb");
    if (!csv) return false;
    fputs("rank,token_id,token,byte_length,category,train_count,train_percent,validation_count,validation_percent\n", csv);
    for (int rank = 0; rank < active; ++rank) {
        const int id = ranked[rank].id;
        fprintf(csv, "%d,%d,", rank + 1, id);
        write_csv_string(csv, tok->token_str[id]);
        fprintf(csv, ",%d,%s,%llu,%.8f,%llu,%.8f\n",
            tok->token_len[id], token_category(id),
            (unsigned long long)train_counts[id],
            100.0 * (double)train_counts[id] / (double)train_tokens,
            (unsigned long long)validation_counts[id],
            100.0 * (double)validation_counts[id] / (double)validation_tokens);
    }
    if (fclose(csv) != 0) return false;
    printf("  token frequency CSV       : minima_token_frequency.csv (%d active rows)\n", active);

    *out_train_counts = train_counts;
    *out_rank_by_id = rank_by_id;
    *out_active = active;
    return true;
}

static bool frequency_loss_report(Model* model, Arena* temp,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    int seq_len, const uint64_t* train_counts, const int* rank_by_id,
    int active_tokens) {
    if (!model || !temp || !all_ids || seq_len <= 0 || seq_len > model->max_seq ||
        train_tokens < seq_len + 1 || validation_tokens < seq_len + 1)
        return false;

    arena_reset(temp);
    Activations act;
    if (!activations_init(&act, temp, 1, seq_len, model))
        return false;

    double loss_sum[2][DIAG_BUCKETS] = {{0}};
    uint64_t target_count[2][DIAG_BUCKETS] = {{0}};
    uint16_t input[512];
    const int region_starts[2] = {0, train_tokens};
    const int region_lengths[2] = {train_tokens, validation_tokens};

    for (int split = 0; split < 2; ++split) {
        const int span = region_lengths[split] - seq_len - 1;
        const int windows = span > 0 ? DIAG_EVAL_WINDOWS : 1;
        for (int w = 0; w < windows; ++w) {
            const int offset = windows > 1
                ? (int)(((int64_t)span * w) / (windows - 1)) : 0;
            const int start = region_starts[split] + offset;
            for (int t = 0; t < seq_len; ++t)
                input[t] = all_ids[start + t];

            float* logits = model_forward(model, &act, input, 1, seq_len);
            if (!logits) return false;

            for (int t = 0; t < seq_len; ++t) {
                const uint16_t target = all_ids[start + t + 1];
                if (target >= model->vocab_size) return false;
                const float* row = logits + (size_t)t * model->vocab_size;
                float max_logit = row[0];
                for (int v = 1; v < model->vocab_size; ++v)
                    if (row[v] > max_logit) max_logit = row[v];
                double sum_exp = 0.0;
                for (int v = 0; v < model->vocab_size; ++v)
                    sum_exp += exp((double)row[v] - (double)max_logit);
                const double loss = log(sum_exp) -
                    ((double)row[target] - (double)max_logit);
                const int bucket = frequency_bucket(target, train_counts,
                    rank_by_id, active_tokens);
                loss_sum[split][bucket] += loss;
                target_count[split][bucket]++;
            }
        }
    }

    FILE* csv = fopen("minima_frequency_loss.csv", "wb");
    if (!csv) return false;
    fputs("split,frequency_bucket,target_count,mean_cross_entropy,perplexity\n", csv);
    printf("\nLoss by training-token-frequency bucket\n");
    printf("  Evaluation uses %d deterministic, evenly spaced windows per split.\n",
        DIAG_EVAL_WINDOWS);
    for (int split = 0; split < 2; ++split) {
        const char* split_name = split == 0 ? "train_sample" : "validation";
        for (int bucket = 0; bucket < DIAG_BUCKETS; ++bucket) {
            const uint64_t count = target_count[split][bucket];
            const double mean = count ? loss_sum[split][bucket] / (double)count : NAN;
            const double ppl = count ? exp(mean) : NAN;
            fprintf(csv, "%s,%s,%llu,%.8f,%.8f\n",
                split_name, frequency_bucket_names[bucket],
                (unsigned long long)count, mean, ppl);
            if (count)
                printf("  %-14s %-9s n=%5llu loss=%8.4f ppl=%9.2f\n",
                    split_name, frequency_bucket_names[bucket],
                    (unsigned long long)count, mean, ppl);
            else
                printf("  %-14s %-9s n=0 (no sampled targets)\n",
                    split_name, frequency_bucket_names[bucket]);
        }
    }
    if (fclose(csv) != 0) return false;
    printf("  frequency loss CSV        : minima_frequency_loss.csv\n");
    return true;
}

static bool ngram_equal(const uint16_t* ids, int a, int b, int width) {
    for (int i = 0; i < width; ++i)
        if (ids[a + i] != ids[b + i]) return false;
    return true;
}

static GenerationStats generation_stats(const uint16_t* ids, int start, int count) {
    GenerationStats stats;
    memset(&stats, 0, sizeof(stats));
    if (!ids || start < 0 || count <= 0) return stats;

    uint32_t histogram[8192] = {0};
    int unique = 0;
    int adjacent = 0;
    for (int i = 0; i < count; ++i) {
        const uint16_t id = ids[start + i];
        if (id < 8192) {
            if (histogram[id] == 0) unique++;
            histogram[id]++;
        }
        if (i > 0 && id == ids[start + i - 1]) adjacent++;
    }
    stats.unique_tokens = unique;
    stats.adjacent_repeat_rate = count > 1
        ? (double)adjacent / (double)(count - 1) : 0.0;

    for (int id = 0; id < 8192; ++id) {
        if (histogram[id]) {
            const double p = (double)histogram[id] / (double)count;
            stats.entropy_bits -= p * (log(p) / log(2.0));
        }
    }

    for (int width = 1; width <= 3; ++width) {
        const int total = count - width + 1;
        if (total <= 0) continue;
        int distinct = 0;
        for (int i = 0; i < total; ++i) {
            bool seen = false;
            for (int j = 0; j < i; ++j) {
                if (ngram_equal(ids, start + i, start + j, width)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) distinct++;
        }
        stats.distinct[width - 1] = (double)distinct / (double)total;
    }

    for (int width = 2; width <= 4 && width <= count; ++width) {
        const int total = count - width + 1;
        for (int i = 0; i < total; ++i) {
            int occurrences = 1;
            for (int j = i + 1; j < total; ++j)
                if (ngram_equal(ids, start + i, start + j, width))
                    occurrences++;
            if (occurrences > stats.max_repeated_ngram)
                stats.max_repeated_ngram = occurrences;
        }
    }
    return stats;
}

static bool write_generation_report(Model* model, const Tokenizer* tok,
    KVCache* cache) {
    static const char* const prompts[] = {
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "The meaning of life",
        "It was a dark",
        "Hello there, how do you do?",
        "A language model learns",
        "The most important thing is"
    };
    static const GenerationMode modes[] = {
        {"greedy", 1.0f, 1, 1.0f, 1.0f},
        {"cool_topk", 0.7f, 40, 1.0f, 1.0f},
        {"balanced_topk", 0.8f, 40, 1.0f, 1.0f},
        {"warm_topk", 1.0f, 40, 1.0f, 1.0f},
        {"nucleus_repeat_penalty", 0.8f, 40, 0.9f, 1.1f}
    };

    FILE* csv = fopen("minima_generation_diagnostics.csv", "wb");
    if (!csv) return false;
    fputs("prompt,mode,seed,temperature,top_k,top_p,repetition_penalty,new_tokens,unique_tokens,entropy_bits,distinct_1,distinct_2,distinct_3,adjacent_repeat_rate,max_repeated_ngram,text\n", csv);

    printf("\nGeneration diagnostics\n");
    printf("  Fixed prompts and seeds; each mode receives the same prompt.\n");
    uint16_t prompt_ids[512], output_ids[512], verify_ids[512];
    char decoded[512 * TOK_MAX_TOKEN_LEN + 1];

    for (size_t p = 0; p < sizeof(prompts) / sizeof(prompts[0]); ++p) {
        const int prompt_len = tokenizer_encode(tok,
            (const uint8_t*)prompts[p], strlen(prompts[p]),
            prompt_ids, 512);
        if (prompt_len <= 0 || prompt_len >= model->max_seq) {
            fclose(csv);
            return false;
        }

        printf("\n  Prompt: \"%s\"\n", prompts[p]);
        if (!generation_inspect_next_tokens(model, tok, cache,
            prompt_ids, prompt_len, 10)) {
            fclose(csv);
            return false;
        }

        for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m) {
            const uint64_t seed = 0x4D494E494D41ULL +
                (uint64_t)(p * 131u) + (uint64_t)(m * 7919u);
            Rng rng;
            rng_seed(&rng, seed);
            const int max_new = model->max_seq - prompt_len < DIAG_GENERATION_TOKENS
                ? model->max_seq - prompt_len : DIAG_GENERATION_TOKENS;
            const int total = generate_ex(model, (Tokenizer*)tok, cache,
                prompt_ids, prompt_len, output_ids, 512, max_new,
                modes[m].temperature, modes[m].top_k, modes[m].top_p,
                modes[m].repetition_penalty, &rng);
            if (total <= prompt_len) {
                fclose(csv);
                return false;
            }

            if (m == 0) {
                Rng verify_rng;
                rng_seed(&verify_rng, seed);
                const int verify_total = generate_ex(model, (Tokenizer*)tok,
                    cache, prompt_ids, prompt_len, verify_ids, 512, max_new,
                    modes[m].temperature, modes[m].top_k, modes[m].top_p,
                    modes[m].repetition_penalty, &verify_rng);
                if (verify_total != total ||
                    memcmp(output_ids, verify_ids,
                        (size_t)total * sizeof(uint16_t)) != 0) {
                    fclose(csv);
                    return false;
                }
            }

            const int generated = total - prompt_len;
            const GenerationStats stats = generation_stats(output_ids,
                prompt_len, generated);
            const int decoded_bytes = tokenizer_decode(tok, output_ids,
                total, decoded, (int)sizeof(decoded));
            if (decoded_bytes < 0) {
                fclose(csv);
                return false;
            }

            printf("    %-24s T=%.2f k=%d p=%.2f rep=%.2f | H=%.3f distinct=%.2f/%.2f/%.2f | %s\n",
                modes[m].name, modes[m].temperature, modes[m].top_k,
                modes[m].top_p, modes[m].repetition_penalty,
                stats.entropy_bits, stats.distinct[0], stats.distinct[1],
                stats.distinct[2], decoded);

            write_csv_string(csv, prompts[p]);
            fputc(',', csv);
            write_csv_string(csv, modes[m].name);
            fprintf(csv, ",%llu,%.4f,%d,%.4f,%.4f,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%d,",
                (unsigned long long)seed, modes[m].temperature, modes[m].top_k,
                modes[m].top_p, modes[m].repetition_penalty, generated,
                stats.unique_tokens, stats.entropy_bits, stats.distinct[0],
                stats.distinct[1], stats.distinct[2],
                stats.adjacent_repeat_rate, stats.max_repeated_ngram);
            write_csv_string(csv, decoded);
            fputc('\n', csv);
        }
    }

    if (fclose(csv) != 0) return false;
    printf("\n  generation diagnostics CSV: minima_generation_diagnostics.csv\n");
    printf("  deterministic greedy check: PASS\n");
    return true;
}

bool diagnostics_run(Model* model, const Tokenizer* tok,
    Arena* permanent_arena, Arena* temp_arena,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    int sequence_length) {
    if (!model || !tok || !permanent_arena || !temp_arena || !all_ids ||
        train_tokens <= 0 || validation_tokens <= 0 ||
        sequence_length <= 0 || sequence_length > model->max_seq)
        return false;

    uint64_t* train_counts = NULL;
    int* rank_by_id = NULL;
    int active_tokens = 0;
    if (!tokenizer_report(tok, permanent_arena, all_ids, train_tokens,
        validation_tokens, &train_counts, &rank_by_id, &active_tokens))
        return false;

    if (!frequency_loss_report(model, temp_arena, all_ids, train_tokens,
        validation_tokens, sequence_length, train_counts, rank_by_id,
        active_tokens))
        return false;

    KVCache cache;
    if (!kvcache_init(&cache, permanent_arena, model, model->max_seq))
        return false;
    if (!write_generation_report(model, tok, &cache))
        return false;

    printf("\nModel-quality diagnostics complete. No training state was changed.\n");
    return true;
}



#ifndef CONFIG_H
#define CONFIG_H

// Define constants and settings
#define HIDDEN_SIZE 768
#define QUERY_MAXLEN 32
#define DOC_MAXLEN 180
#define DIMENSION_SIZE 128
#define VOCAB_SIZE 30522
#define PAD_TOKEN_ID 0
#define CODEBOOK_COUNT 16 
#define CODES_COUNT 256
#define TORCH_DTYPE torch::kFloat32
#define SIMILARITY_METRIC "cosine"
#define CODEBOOK_DIM 8

// #define QUERY_FILE "../data/queries.dev.tsv"
// #define RESULTS_FILE "../data/retrieval-results-test.tsv"

// #define QUERY_FILE "../data/msmarco-test2019-queries-qrel.tsv"
// #define RESULTS_FILE "../data/retrieval-results.tsv"

#define QUERY_FILE "../data/queries.dev.tsv"
#define RESULTS_FILE "../data/index-sqhd-54-refcorrect-top1000.trec"
#define BASE_STORE_FILE "../data/output_result/result_"
#define OUTPUT_FILE "../output/results.trec"

#define IN_MEMORY_CODES false
#define STORE_SIZE 256


#define PRE_BATCH_SIZE 1
#define TOTAL_QUERIES 698000

// #define PRE_BATCH_SIZE 1
// #define TOTAL_QUERIES 1

#define USE_BECR false
#define WINDOW_SIZE 3
#define AVERAGE_SCORE false

#endif

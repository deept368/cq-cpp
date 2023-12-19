#ifndef CONFIG_H
#define CONFIG_H

// Define constants and settings

// Model-related configurations
#define HIDDEN_SIZE 768            // Size of the hidden layer in the neural network model
#define QUERY_MAXLEN 32            // Maximum length of a query input
#define DOC_MAXLEN 180             // Maximum length of a document input
#define DIMENSION_SIZE 128         // Size of the embedding dimension
#define VOCAB_SIZE 30522           // Vocabulary size
#define PAD_TOKEN_ID 0             // Token ID representing padding
#define TORCH_DTYPE torch::kFloat32 // Data type used by PyTorch (float32)
#define SIMILARITY_METRIC "cosine" // Similarity metric used for scoring (e.g., "cosine")

// Coding configurations
#define CODEBOOK_DIM 8             // Dimensionality of the codebook
#define CODEBOOK_COUNT 16          // Number of codebooks
#define CODES_COUNT 256            // Number of codes in each codebook

// File paths and data for input/output files
#define QUERY_FILE "../data/queries.dev.tsv"                        // Path to the query id -> query text pairs file
#define RESULTS_FILE "../data/index-sqhd-54-refcorrect-top1000.trec" // Path to the result file for initialize 1000 document fetching that will be used as input for our document reranking algorithm
#define OUTPUT_FILE "../output/results.trec"                         // Output file path for storing final results

// Codes store configuration
#define BASE_STORE_FILE "../data/output_result/result_"             // Base path for storing codes store file (code store split into multiple numbered files)
#define STORE_SIZE 256        // Number of files in code store
#define IN_MEMORY_CODES false // Flag indicating whether codes are stored in memory or loaded on-the-fly

// Batch and processing configurations
#define PRE_BATCH_SIZE 100 // Size of the batch processed in each iteration
#define TOTAL_QUERIES 6980 // Total number of queries to be processed

// BECR configurations
#define USE_BECR false      // Flag indicating whether to use BECR for code retrieval
#define WINDOW_SIZE 3       // Window size for BECR (applicable if USE_BECR is true)

#define AVERAGE_SCORE false  // Flag indicating whether to average scores for final ranking

#endif

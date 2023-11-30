#include "../bert/tokenizer.h"
#include "../config.h"
#include "../utils.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import_legacy.h>
#include "./nlohmann/json.hpp"

using json = nlohmann::json;

namespace lh{

    /**
     Does all the necessary preprocessing on input strings and computes the BERT embeddings of strings.  
    */
    class BecrCompute{

        public:
            explicit BecrCompute();
            ~BecrCompute();
            torch::Tensor compute(std::vector<std::string>* input_string);

        private:
            std::size_t query_maxlen;
            std::int64_t vocab_size_;
            std::int64_t dimension_size_;
            std::int64_t pad_token_id_;
            FullTokenizer* tokenizer_;
            torch::nn::EmbeddingImpl* non_contextual_embedding;
            torch::Tensor* unigram_emb;
            // json bigram_emb;
            unordered_map<string, unordered_map<string, vector<vector<double>>>> bigram_emb;

    };
}
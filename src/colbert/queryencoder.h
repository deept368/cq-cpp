#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "../utils.h"
#include "../config.h"
#include "../bert/bertcompute.h"
#include "../becr/becrcompute.h"


namespace lh{

    /**
     Encodes the query using the BERT model and returns embeddings for all the query input strings.
    */
    template<class T>
    class QueryEncoder{

        public:
            explicit QueryEncoder();
            ~QueryEncoder();
            torch::Tensor encode(std::vector<std::string>* input_strings);

        private:
            std::size_t query_maxlen;
            std::size_t hidden_size_;
            std::size_t dimension_size_;

            BertCompute<T>* bert_compute_;
            BecrCompute* becr_compute_;
            torch::nn::LinearImpl* linear_model_;
            
    };
}
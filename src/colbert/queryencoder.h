#include "../bert/bertcompute.h"



namespace lh{

    template<class T>
    class QueryEncoder{

        public:
            explicit QueryEncoder();
            ~QueryEncoder();
            torch::Tensor encode(std::vector<std::string> input_strings);

        private:
            std::size_t query_maxlen;
            std::size_t hidden_size_;
            std::size_t dimension_size_;

            BertCompute<T>* bert_compute_;
            
    };
}
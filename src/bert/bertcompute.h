#include "bert.h"
#include "tokenizer.h"
#include "model.pb.h"
#include "../config.h"
#include "../utils.h"


namespace lh{

    template<class T>
    class BertCompute{

        public:
            explicit BertCompute();
            ~BertCompute();
            std::vector<T> compute(std::vector<std::string> input_string, bool isQuery);

        private:
            std::size_t query_maxlen;
            std::size_t hidden_size_;
            
            Bert<T>* bert_;
            FullTokenizer* tokenizer_;

    };
}
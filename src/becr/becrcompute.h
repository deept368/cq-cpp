#include "bert.h"
#include "tokenizer.h"
#include "model.pb.h"
#include "../config.h"
#include "../utils.h"


namespace lh{

    /**
     Does all the necessary preprocessing on input strings and computes the BERT embeddings of strings.  
    */
    template<class T>
    class BecrCompute{

        public:
            explicit BecrCompute();
            ~BecrCompute();
            std::vector<T>* compute(std::vector<std::string>* input_string, bool isQuery);

        private:
            std::size_t query_maxlen;
            
            FullTokenizer* tokenizer_;

    };
}
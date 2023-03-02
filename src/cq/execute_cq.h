#include <torch/torch.h>
#include "../colbert/queryencoder.h"
#include "../colbert/score.h"
#include "decoder.h"
#include "../config.h"
#include "../utils.h"


namespace lh{
    /**
    Calls all the important components, from encoding the query 
    and getting approx document embeddings to computing the scores for all query-document pairs. 
    This store the scoring result in a map where queries correspond to their respective scoring tensors.
    */
    class ExecuteCQ{
        public:
            explicit ExecuteCQ();
            ~ExecuteCQ();
            void execute(vector<string> input_strings);
            
        private:
            Decoder* decoder_;
            QueryEncoder<float>* query_encoder_;
            Score* score_;
    };
}
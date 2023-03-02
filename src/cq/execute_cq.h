#include <torch/torch.h>
#include "../colbert/queryencoder.h"
#include "../colbert/score.h"
#include "decoder.h"
#include "../config.h"
#include "../utils.h"


namespace lh{

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
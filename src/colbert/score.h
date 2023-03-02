#include <torch/torch.h>
#include "../config.h"

namespace lh{
    /**
     Computes score of all documents for a given query and top K documents.
    */
    class Score{
        public:
            explicit Score();
            ~Score();
            torch::Tensor compute_scores(torch::Tensor Q, torch::Tensor D);
            
        private:
            std::string similarity_metric_;
           
    };
}
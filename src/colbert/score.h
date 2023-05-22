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
            std::vector<float> compute_scores(torch::Tensor Q, std::vector<torch::Tensor>& approx_tensors);
            // std::vector<float> compute_scores(std::vector<float>& Q, std::vector<torch::Tensor>& approx_tensors);
            
        private:
            std::string similarity_metric_;
           
    };
}
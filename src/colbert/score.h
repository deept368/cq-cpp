#include <torch/torch.h>
#include "../config.h"

namespace lh{

    class Score{
        public:
            explicit Score();
            ~Score();
            torch::Tensor compute_scores(torch::Tensor Q, torch::Tensor D);
            
        private:
            std::string similarity_metric_;
           
    };
}
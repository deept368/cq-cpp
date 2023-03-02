#include "score.h"
#include<iostream>

namespace lh{

    
    Score::Score(){
        similarity_metric_ = SIMILARITY_METRIC;
    }

  
    Score::~Score(){
        
    }    

    torch::Tensor Score::compute_scores(torch::Tensor Q, torch::Tensor D){
        if (similarity_metric_ == "cosine") {
            auto scores = std::get<0>(Q.matmul(D.permute({0, 2, 1})).max(2)).sum(1);   
            return scores;
        }

        assert(similarity_metric_ == "l2");
     
        auto Qs = torch::split(Q, 64);
        auto Ds = torch::split(D, 64);
        std::vector<torch::Tensor> scores;
        for (size_t i = 0; i < Qs.size(); i++) {
            auto q = Qs[i];
            auto d = Ds[i];
            auto score = std::get<0>((-1.0 * ((q.unsqueeze(2) - d.unsqueeze(1)).pow(2)).sum(-1)).max(-1)).sum(-1);
            scores.push_back(score);
        }

        torch::Tensor result = torch::cat(scores, 0);
        return result;
    }

}
   

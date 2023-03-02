#include "execute_cq.h"


#include<iostream>

using namespace std;

namespace lh{
    
    ExecuteCQ::ExecuteCQ(){
       decoder_ = new Decoder();
       query_encoder_ = new QueryEncoder<float>();
       score_ = new Score();
    }

  
    ExecuteCQ::~ExecuteCQ(){
       delete decoder_;
       delete query_encoder_;
       delete score_;
    }    

    void ExecuteCQ::execute(vector<string> input_strings){
        auto Q = query_encoder_->encode(input_strings);
      
        map<string, torch::Tensor> query_score_tensor_map;
        map<string, torch::Tensor> query_doc_emb_approx_map = decoder_->decode();
        for(auto& input: input_strings){
            auto D = query_doc_emb_approx_map[input];
            auto score = score_->compute_scores(Q, D); 
            cout<<score.sizes()<<endl;
            cout<<score[0]<<" "<<score[1]<<endl;
            query_score_tensor_map.insert(make_pair(input, score));
        }
    }
}
   

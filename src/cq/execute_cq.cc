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

    /**
    Ecodes a given list of input strings by encoding them into a query tensor using query encoder. Also, fetches
    Fetches the topK docuements and and their approx embeddings from the decode() method.
    Computing the scores between the query tensor and the document embeddings.
    The scores are then stored in a map where the key is the input string and the value
    is the corresponding score tensor.
    @param input_strings A list of input strings to be processed
    @return void
    */
    void ExecuteCQ::execute(vector<string> input_strings){
        //query input_strings are encoded
        auto Q = query_encoder_->encode(input_strings);
      
        map<string, torch::Tensor> query_score_tensor_map;

        //approx document embeddings are retrieved for topK documents for each query
        map<string, torch::Tensor> query_doc_emb_approx_map = decoder_->decode();
        //for each query, score is computed in a sequential manner
        for(auto& input: input_strings){
            auto D = query_doc_emb_approx_map[input];
            auto score = score_->compute_scores(Q, D); 
            cout<<score.sizes()<<endl;
            cout<<score[0]<<" "<<score[1]<<endl;
            query_score_tensor_map.insert(make_pair(input, score));
        }
    }
}
   

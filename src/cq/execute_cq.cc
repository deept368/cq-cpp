#include "execute_cq.h"

#include<chrono>
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

        #ifdef PRFILE_CQ
            auto begin = std::chrono::system_clock::now();
        #endif
        
        //query input_strings are encoded
        auto Q_all = query_encoder_->encode(input_strings);
        map<string, torch::Tensor> query_score_tensor_map;

        //approx document embeddings are retrieved for topK documents for each query
        map<string, torch::Tensor> query_doc_emb_approx_map = decoder_->decode();
        //for each query, score is computed in a sequential manner
        for(std::size_t idx=0; idx<input_strings.size(); idx++){
            auto D = query_doc_emb_approx_map[input_strings[idx]];
            auto score = score_->compute_scores(Q_all[idx].unsqueeze(0), D); 
            query_score_tensor_map.insert(make_pair(input_strings[idx], score));
        }
        cout<<query_score_tensor_map<<endl;

        #ifdef PRFILE_CQ
            auto end = std::chrono::system_clock::now();
            std::cout<<"total execution time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif
    }
}
   

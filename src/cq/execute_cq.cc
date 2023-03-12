#include "execute_cq.h"
#include "queryprocessor.h"

#include<chrono>
#include<iostream>

using namespace std;

namespace lh{
    
    ExecuteCQ::ExecuteCQ(){
       decoder_ = new Decoder();
       query_encoder_ = new QueryEncoder<float>();
       score_ = new Score();
       query_processor_ = new QueryProcessor();
    }

  
    ExecuteCQ::~ExecuteCQ(){
       delete decoder_;
       delete query_encoder_;
       delete score_;
       delete query_processor_;
    }    

    /**
    Encodes a list of input strings by encoding them into a query tensor using query encoder. Also, fetches
    Fetches the topK docuements and and their approx embeddings from the decode() method.
    Computing the scores between the query tensor and the document embeddings.
    The scores are then stored in a map where the key is the input string and the value
    is the corresponding score tensor.
    @param 
    @return void
    */
    void ExecuteCQ::execute(){

        #ifdef PRFILE_CQ
            auto begin = std::chrono::system_clock::now();
        #endif
        
        std::vector<std::string> input_strings;

        //approx document embeddings are retrieved for topK documents for each query
        map<std::size_t, torch::Tensor> query_doc_emb_approx_map = decoder_->decode();

        for (const auto& query_doc_emb_pair : query_doc_emb_approx_map) {
            std::string input_string = query_processor_->getQuery(query_doc_emb_pair.first);
            input_strings.push_back(input_string);
        }


        //query input_strings are encoded
        auto Q_all = query_encoder_->encode(input_strings);
        map<string, torch::Tensor> query_score_tensor_map;

       
        //for each query, score is computed in a sequential manner
        for(std::size_t idx=0; idx<input_strings.size(); idx++){
            std::size_t query_id = query_processor_->getQueryId(input_strings[idx]);
            auto D = query_doc_emb_approx_map[query_id];
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
   

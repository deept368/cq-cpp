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

    bool compare_pairs(const pair<string, float>& p1, const pair<string, float>& p2) {
        return p1.second > p2.second;
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

        //open trec file
        std::ofstream trec_file("../output/results.trec");

        std::vector<std::string> input_strings;

        //approx document embeddings are retrieved for topK documents for each query
        map<int, map<std::string, torch::Tensor>> query_doc_emb_approx_map = decoder_->decode();

        for (const auto& query_doc_emb_pair : query_doc_emb_approx_map) {
            std::string input_string = query_processor_->getQuery(query_doc_emb_pair.first);
            input_strings.push_back(input_string);
        }


        //query input_strings are encoded
        auto Q_all = query_encoder_->encode(input_strings);
       
        std::size_t idx = 0;
        //for each query, score is computed in a sequential manner
        for (const auto& query_doc_emb_pair : query_doc_emb_approx_map) {
            int query_id = query_doc_emb_pair.first;

            std::vector<torch::Tensor> approx_tensors;
            for (auto& doc_emb_pairs : query_doc_emb_pair.second){
                 approx_tensors.push_back(doc_emb_pairs.second);
            }

            auto doc_emb_approx = torch::cat(approx_tensors, 0);
            auto D = torch::nn::functional::normalize(doc_emb_approx,
                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(2)); 

            auto score = score_->compute_scores(Q_all[idx].unsqueeze(0), D); 
           
            std::size_t doc_idx = 0;
            map<std::string, float> doc_id_score_map;

            for (auto& doc_emb_pairs : query_doc_emb_pair.second){
                std::string doc_id = doc_emb_pairs.first;
                doc_id_score_map.insert(make_pair(doc_id, score[doc_idx].item<float>()));    
                doc_idx++;
            }
           

            vector<pair<std::string, float>> doc_id_score_vec(doc_id_score_map.begin(), doc_id_score_map.end());
            sort(doc_id_score_vec.begin(), doc_id_score_vec.end(), compare_pairs);
            std::size_t rank = 1;
            for (auto& doc_id_score_pair : doc_id_score_vec) {
                std::string doc_id = doc_id_score_pair.first;
                auto score = doc_id_score_pair.second;
                const std::string formatted_line = format_trec_line(query_id, doc_id, rank, score, "cq_rerank");
                cout<<formatted_line<<endl;
                rank++;
            }
            idx++;
        }

        #ifdef PRFILE_CQ
            auto end = std::chrono::system_clock::now();
            std::cout<<"total execution time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif
    }
}
   

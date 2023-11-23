#include "execute_cq.h"


#include<chrono>
#include<iostream>

using namespace std;

namespace lh{
    
    ExecuteCQ::ExecuteCQ(){
       decoder_ = new Decoder();
       query_encoder_ = new QueryEncoder<float>();
       score_ = new Score();
       query_mapping_ = new QueryMapping();
       query_processor_ = new QueryProcessor();
    }

  
    ExecuteCQ::~ExecuteCQ(){
       delete decoder_;
       delete query_encoder_;
       delete score_;
       delete query_mapping_;
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


        int offset = 0;

        vector<int> fetch_times, decoding_times, encoding_times, scoring_times;
        while(offset <= TOTAL_QUERIES / PRE_BATCH_SIZE){

             #ifdef PRFILE_CQ
                auto begin_fetch = std::chrono::system_clock::now();
            #endif

            unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* fetched_codes = query_processor_->getCodes(offset);
            
             #ifdef PRFILE_CQ
                auto end_fetch = std::chrono::system_clock::now();
                fetch_times.push_back((std::chrono::duration_cast<std::chrono::microseconds>(end_fetch-begin_fetch).count())/1000);
                std::cout<<"total fetch time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_fetch-begin_fetch).count())/1000 << std::endl;
            #endif

            #ifdef PRFILE_CQ
                auto begin_decoding = std::chrono::system_clock::now();
            #endif

            //approx document embeddings are retrieved for topK documents for each query
            map<int, map<std::string, torch::Tensor>*>* query_doc_emb_approx_map = decoder_->decode(fetched_codes);

            #ifdef PRFILE_CQ
                auto end_decoding = std::chrono::system_clock::now();
                decoding_times.push_back((std::chrono::duration_cast<std::chrono::microseconds>(end_decoding-begin_decoding).count())/1000);
                std::cout<<"total decoding time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_decoding-begin_decoding).count())/1000 << std::endl;
            #endif

            std::vector<std::string>* input_strings = new std::vector<std::string>();
            
            for (const auto& query_doc_codes_pair : *fetched_codes) {
                std::string input_string = query_mapping_->getQuery(query_doc_codes_pair.first);
                input_strings->push_back(input_string);
            }

            //query input_strings are encoded

              #ifdef PRFILE_CQ
                auto begin_encoding = std::chrono::system_clock::now();
              #endif
                   
                auto Q_all = query_encoder_->encode(input_strings);
                    
              #ifdef PRFILE_CQ
                auto end_encoding = std::chrono::system_clock::now();
                encoding_times.push_back((std::chrono::duration_cast<std::chrono::microseconds>(end_encoding-begin_encoding).count())/1000);
                std::cout<<"total query encoding time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_encoding-begin_encoding).count())/1000 << std::endl;
              #endif

            cout<<"encoded"<<endl;

            std::size_t idx = 0;

            #ifdef PRFILE_CQ
                auto begin_scoring = std::chrono::system_clock::now();
            #endif

            //for each query, score is computed in a sequential manner
            for (const auto& query_doc_emb_pair : *query_doc_emb_approx_map) {
                int query_id = query_doc_emb_pair.first;
              
                std::vector<torch::Tensor>* approx_tensors = new std::vector<torch::Tensor>();
                for (auto& doc_emb_pairs : *(query_doc_emb_pair.second)){
                    approx_tensors->push_back(doc_emb_pairs.second);
                }

                auto doc_emb_approx = torch::cat(*approx_tensors, 0);
                auto D = torch::nn::functional::normalize(doc_emb_approx,
                                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(2)); 

                
                auto score = score_->compute_scores(Q_all[idx].unsqueeze(0), D); 

                std::size_t doc_idx = 0;
                map<std::string, float>* doc_id_score_map = new map<std::string, float>();

                for (auto& doc_emb_pairs : *(query_doc_emb_pair.second)){
                    std::string doc_id = doc_emb_pairs.first;
                    doc_id_score_map->insert(make_pair(doc_id, score[doc_idx].item<float>()));    
                    doc_idx++;
                }


                std::vector<std::pair<std::string, float>>* doc_id_score_vec = new std::vector<std::pair<std::string, float>>(doc_id_score_map->begin(), doc_id_score_map->end());
                std::sort(doc_id_score_vec->begin(), doc_id_score_vec->end(), compare_pairs);
                std::size_t rank = 1;
                for (auto& doc_id_score_pair : *doc_id_score_vec) {
                    std::string doc_id = doc_id_score_pair.first;
                    auto score = doc_id_score_pair.second;
                    const std::string formatted_line = format_trec_line(query_id, doc_id, rank, score, "cq_rerank");
                    trec_file << formatted_line;
                    rank++;
                }
                delete doc_id_score_vec;
                delete doc_id_score_map;
                delete approx_tensors;
                idx++;
            }

            #ifdef PRFILE_CQ
                auto end_scoring = std::chrono::system_clock::now();
                scoring_times.push_back((std::chrono::duration_cast<std::chrono::microseconds>(end_scoring-begin_scoring).count())/1000);
                std::cout<<"total scoring time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_scoring-begin_scoring).count())/1000 << std::endl;
            #endif

            delete input_strings;

            query_doc_emb_approx_map->clear();
            delete query_doc_emb_approx_map;

            for (auto& kv1 : *fetched_codes)
            {
                for (auto& kv2 : *kv1.second) 
                {
                    if (! IN_MEMORY_CODES)
                    {
                        for (auto* ptr : *kv2.second) 
                            delete ptr;
                    }
                    cout << "Free level 2" << endl;
                    delete kv2.second;
                }
                    
                cout << "Free level 1" << endl;
                delete kv1.second;
            }
            cout << "Free level 0" << endl;
            delete fetched_codes;
            
            offset += PRE_BATCH_SIZE;

            #ifdef PRFILE_CQ
                cout << "Average times" << endl;
                cout << "Fetch time: " << accumulate(fetch_times.begin(), fetch_times.end(), 0) / fetch_times.size() << endl;
                cout << "Decoding time: " << accumulate(decoding_times.begin(), decoding_times.end(), 0) / decoding_times.size() << endl;
                cout << "Encoding time: " << accumulate(encoding_times.begin(), encoding_times.end(), 0) / encoding_times.size() << endl;
                cout << "Scoring time: " << accumulate(scoring_times.begin(), scoring_times.end(), 0) / scoring_times.size() << endl;
            #endif
        }

        #ifdef PRFILE_CQ
                auto end = std::chrono::system_clock::now();
                std::cout<<"total execution time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif

    }
}
   

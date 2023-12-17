#include "execute_cq.h"


#include<chrono>
#include<iostream>

using namespace std;

namespace lh{

    void printConvertedCodes(const std::unordered_map<int, std::unordered_map<std::string, std::vector<std::vector<int>*>*>*>& converted_codes) {
    for (const auto& outer_pair : converted_codes) {
        int outer_key = outer_pair.first;
        const auto& inner_map = outer_pair.second;

        for (const auto& inner_pair : *inner_map) {
            const std::string& inner_key = inner_pair.first;
            const auto& inner_vector = *inner_pair.second;

            std::cout << "Outer Key: " << outer_key << ", Inner Key: " << inner_key << ", Values: ";

            for (const auto& inner_vector_ptr : inner_vector) {
                const auto& converted_pair = *inner_vector_ptr;

                std::cout << "[ ";
                for (const auto& value : converted_pair) {
                    std::cout << value << " ";
                }
                std::cout << "] ";
            }

            std::cout << std::endl;
        }
    }
}
    
    ExecuteCQ::ExecuteCQ(){
    //    decoder_ = new Decoder();
    //    query_encoder_ = new QueryEncoder<float>();
    //    score_ = new Score();
    //    query_mapping_ = new QueryMapping();
       query_processor_ = new QueryProcessor();
    }

  
    ExecuteCQ::~ExecuteCQ(){
    //    delete decoder_;
    //    delete query_encoder_;
    //    delete score_;
    //    delete query_mapping_;
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
        std::ofstream trec_file(OUTPUT_FILE);

        
        auto original_scores = query_processor_->getOriginalScores();

        int offset = 0;

        vector<int> fetch_times, decoding_times, encoding_times, scoring_times;
        while(offset <= TOTAL_QUERIES){

            cout << "Offset is: " << offset << endl;

             #ifdef PRFILE_CQ
                auto begin_fetch = std::chrono::system_clock::now();
            #endif

            // cout << "Reach 1\n";
            unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>*>* fetched_codes = query_processor_->getCodes(offset);
            
            #ifdef PRFILE_CQ
                auto end_fetch = std::chrono::system_clock::now();
                fetch_times.push_back((std::chrono::duration_cast<std::chrono::microseconds>(end_fetch-begin_fetch).count())/1000);
                std::cout<<"total fetch time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_fetch-begin_fetch).count())/1000 << std::endl;
            #endif

            #ifdef PRFILE_CQ
                auto begin_decoding = std::chrono::system_clock::now();
            #endif

            // for (const auto& entry : *query_doc_emb_approx_map)
            // {
            //     int key = entry.first;
            //     auto& inner_map = entry.second;

            //     for (const auto& inner_entry : inner_map) {
            //         const std::string& inner_key = inner_entry.first;
            //         const auto& data_vector = inner_entry.second;

            //         // Convert data_vector to torch::Tensor manually
            //         // Example: torch::Tensor tensor_data = convertToTensor(data_vector);
            //     }
            // }




            //approx document embeddings are retrieved for topK documents for each query
            unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* converted_codes = new std::unordered_map<int, std::unordered_map<std::string, std::vector<std::vector<int>*>*>*>();;
            for (auto& query_id_map: *fetched_codes)
            {
                converted_codes->insert(make_pair(query_id_map.first, new unordered_map<string, vector<vector<int>*>*>));
                for (auto& doc_id_map: *(query_id_map.second))
                {
                    vector<pair<uint16_t, vector<uint8_t>*>>* data = doc_id_map.second;
                    vector<vector<int>*>* curr_data = new vector<vector<int>*>();
                    for (auto& codes: *data)
                    {
                        vector<int>* curr_vec = new vector<int>();
                        curr_vec->push_back(static_cast<int>(codes.first));
                        // cout << codes.first << " ";
                        for (auto& code: *(codes.second))
                        {
                            curr_vec->push_back(static_cast<int>(code));
                            // cout << static_cast<int>(code) << " ";
                        }
                        // cout << endl;
                        curr_data->push_back(curr_vec);
                    }

                    
                    (*converted_codes)[query_id_map.first]->insert(make_pair(doc_id_map.first,curr_data));
                    // cout << "level 1\n";
                }
            // cout << "level 2\n";
            }


// // Converted_codes structure
// std::unordered_map<int, std::unordered_map<std::string, std::vector<std::vector<int>*>*>*>* converted_codes;

// // Iterate through fetched_codes and convert to converted_codes
// for (auto outer_pair : *fetched_codes) {
//     int outer_key = outer_pair.first;
//     auto* inner_map = outer_pair.second;

//     for (auto inner_pair : *inner_map) {
//         string& inner_key = inner_pair.first;
//         auto* inner_vector = inner_pair.second;

//         // Create a new vector for converted_codes
//         vector<vector<int>*>* converted_vector = new vector<vector<int>*>();

//         // Iterate through the inner_vector and convert
//         for (const auto& pair : inner_vector) {
//             uint16_t first_value = pair.first;
//             const auto& second_vector = *pair.second;

//             // Create a new vector for the converted pair
//             std::vector<int>* converted_pair = new std::vector<int>();

//             // Add the first value of the pair
//             converted_pair->push_back(static_cast<int>(first_value));

//             // Add all other values in the second of the pair
//             for (const auto& value : second_vector) {
//                 converted_pair->push_back(static_cast<int>(value));
//             }

//             // Add the converted pair to the converted_vector
//             converted_vector->push_back(converted_pair);
//         }

//         // Add the converted_vector to converted_codes
//         (*(*converted_codes)[outer_key])[inner_key] = converted_vector;
//     }
// }

            map<int, map<std::string, torch::Tensor>*>* query_doc_emb_approx_map = decoder_->decode(converted_codes);
            cout << "Decoded\n";

// Deallocate memory for converted_codes
for (auto& outer_pair : *converted_codes) {
    for (auto& inner_pair : *outer_pair.second) {
        for (auto& vector_ptr : *inner_pair.second) {
            delete vector_ptr; // Deallocate each inner vector
        }
        delete inner_pair.second; // Deallocate the vector of inner vectors
    }
    delete outer_pair.second; // Deallocate the map of inner maps
}
delete converted_codes; // Deallocate the outer map


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
                    
                    if (AVERAGE_SCORE)
                        score = (score + original_scores[query_id][doc_id]) / 2;
                    
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
                if (! IN_MEMORY_CODES)
                {
                    for (auto& kv2 : *kv1.second) 
                    {
                            for (auto ptr : *kv2.second) 
                                delete ptr.second;
                        delete kv2.second;
                    }
                }  
                delete kv1.second;
            }
            delete fetched_codes;
            
            offset += PRE_BATCH_SIZE;

            // #ifdef PRFILE_CQ
            //     cout << "Average times" << endl;
            //     cout << "Fetch time: " << accumulate(fetch_times.begin(), fetch_times.end(), 0) / fetch_times.size() << endl;
            //     cout << "Decoding time: " << accumulate(decoding_times.begin(), decoding_times.end(), 0) / decoding_times.size() << endl;
            //     cout << "Encoding time: " << accumulate(encoding_times.begin(), encoding_times.end(), 0) / encoding_times.size() << endl;
            //     cout << "Scoring time: " << accumulate(scoring_times.begin(), scoring_times.end(), 0) / scoring_times.size() << endl;
            // #endif
        }

        #ifdef PRFILE_CQ
                auto end = std::chrono::system_clock::now();
                std::cout<<"total execution time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif

    }
}
   

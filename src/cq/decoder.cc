#include "decoder.h"
#include <torch/script.h>
#include<iostream>
#include<unordered_map>
#include<cmath>

using namespace std;

namespace lh{
    
    Decoder::Decoder(){
        vocab_size_ = VOCAB_SIZE;
        dimension_size_ = DIMENSION_SIZE;
        pad_token_id_ = PAD_TOKEN_ID; 
        M_ = CODEBOOK_COUNT;
        K_ = CODES_COUNT;
        codebook_dim_ = CODEBOOK_DIM;
        doc_maxlen_ = DOC_MAXLEN;


         //static embedding is fetched for all the vocabulary from a .pt file into a tensor [30522(vocab_size) * 128(embedding_dim)]
        
        torch::Tensor* static_embeddings = new torch::Tensor();
        torch::load(*static_embeddings, "../model/non_contextual_embeddings.pt");
        std::cout << "decoder.cc::static_embeddings tensor size: " << (*static_embeddings).sizes() << std::endl;

        // convert from torch to vector<vector<float>>
        // std::vector<std::vector<float>>* static_embeddings_vec = new ((*static_embeddings).size(0), std::vector<float>((*static_embeddings).size(1)));
        static_embeddings_vec = new std::vector<std::vector<float>>((*static_embeddings).size(0), std::vector<float>((*static_embeddings).size(1)));
        for (int i = 0; i < (*static_embeddings).size(0); ++i) {
            for (int j = 0; j < (*static_embeddings).size(1); ++j) {
                (*static_embeddings_vec)[i][j] = (*static_embeddings).index({i, j}).item<float>();
            }
        }

        // // print to check values
        // // Printing values in static_embeddings
        // std::cout << "Values in static_embeddings:" << std::endl;
        // for (int i = 0; i < (*static_embeddings).size(0); ++i) {
        //     std::cout << (*static_embeddings).index({i, 0}).item<float>() << " " << (*static_embeddings).index({i, 1}).item<float>();
        //     std::cout << std::endl;
        //     if (i>10) break;
        // }

        // // // Printing values in static_embeddings_vec
        // std::cout << "Values in static_embeddings_vec:" << std::endl;
        // int i = 0;
        // for (const auto& row : static_embeddings_vec) {
        //     std::cout << row[0] << " " << row[1];
        //     std::cout << std::endl;
        //     if (i>10) break;
        //     i++;
        // }

         //torch::nn::EmbeddingImpl(PyTorch C++) model is initialised and static_embeddings tensor is loaded as pretrained weight. 
        auto embedding_options = torch::nn::EmbeddingOptions(vocab_size_, dimension_size_).padding_idx(pad_token_id_)._weight(*static_embeddings);
        non_contextual_embedding = new torch::nn::EmbeddingImpl(embedding_options);
        // std::cout << "decoder.cc::non_contextual_embedding tensor size: " << (*non_contextual_embedding).sizes() << std::endl;

        //codebook is loaded. size: [M * codebook_dim_(8) * K]
        codebook = new torch::Tensor();
        torch::load(*codebook, "../model/codebook.pt");
        std::cout << "decoder.cc::codebook tensor size: " << (*codebook).sizes() << std::endl;

        // convert from torch to vector<vector<float>>
        codebook_vec = new std::vector<std::vector<std::vector<float>>>((*codebook).size(0), std::vector<std::vector<float>>(((*codebook).size(1)), std::vector<float>(((*codebook).size(2)))));
        for (int i = 0; i < (*codebook).size(0); ++i) {
            for (int j = 0; j < (*codebook).size(1); ++j) {
                for (int k = 0; k < (*codebook).size(2); ++k) {
                    (*codebook_vec)[i][j][k] = (*codebook).index({i, j, k}).item<float>();
                }
            }
        }

        //composition layers weights are loaded as a 2-d Tensor of [dim_size(128) * (2*dim_size)]
        torch::Tensor* composition_weights = new torch::Tensor();
        torch::load(*composition_weights, "../model/composition_c_e_linear_weights.pt");
        std::cout << "decoder.cc::composition_weights tensor size: " << (*composition_weights).sizes() << std::endl;

        // convert from torch to vector<vector<float>>
        composition_weights_vec = new std::vector<std::vector<float>>((*composition_weights).size(0), std::vector<float>((*composition_weights).size(1)));
        for (int i = 0; i < (*composition_weights).size(0); ++i) {
            for (int j = 0; j < (*composition_weights).size(1); ++j) {
                (*composition_weights_vec)[i][j] = (*composition_weights).index({i, j}).item<float>();
            }
        }

        //composition layer bias are loaded as a 1-d Tensor of [dim_size(128)]
        torch::Tensor* composition_bias = new torch::Tensor();
        torch::load(*composition_bias, "../model/composition_c_e_linear_bias.pt");
        std::cout << "decoder.cc::composition_bias tensor size: " << (*composition_bias).sizes() << std::endl;

        // convert from torch to vector<vector<float>>
        composition_bias_vec = new std::vector<float>((*composition_bias).size(0));
        for (int i = 0; i < (*composition_bias).size(0); ++i) {
            (*composition_bias_vec)[i] = (*composition_bias).index({i}).item<float>();
        }

        //torch::nn::LinearImpl(PyTorch C++) is used to initialise a linear compositon layers and weights and bias are set
        composition_layer = new torch::nn::LinearImpl(torch::nn::LinearOptions(2*dimension_size_, dimension_size_).bias(true));
        composition_layer->weight = *composition_weights;
        composition_layer->bias = *composition_bias;
    }

  
    Decoder::~Decoder(){
        delete non_contextual_embedding;
        delete codebook;
    }    
    
    /**
    Fetches codes for topK documents for all query input strings using the CodeFetcher class. Processes them and generate 
    approx embeddings for all the documents using static embeddings and embeddings decoded from codebook.
    @param void 
    @return A map of query input strings and the approx embedding tensor of their correpsonding topK documents.
    */

     map<int, map<std::string,std::vector<std::vector<std::vector<float>>>*>*>* Decoder::decode(unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* fetched_codes){
   
        /* 
        code_fetcher object is used to fetch a map of queries and their corresponding top K documents and (codes and tokens) of these topK documents.
        codes and tokens of each document are fetched as vec<vec<int>> (dimensions: num of tokens * (K+1)) where 1 in (K+1) is used for static embedding token id. 
        */

        map<int, map<std::string,torch::Tensor>*>* query_doc_approx_emb_map = new map<int, map<std::string,torch::Tensor>*>();
        map<int, map<std::string,std::vector<std::vector<std::vector<float>>>*>*>* query_doc_approx_emb_map_vec = new map<int, map<std::string,std::vector<std::vector<std::vector<float>>>*>*>();

        //we loop over each query string
        int query_counter = 0;
        for (auto&  query_doc_pairs : *fetched_codes) {
            int query_id = query_doc_pairs.first;
            cout << std::endl << std::endl;
            cout << "Processing for query: " << query_id << " " << query_counter++ << endl;
            unordered_map<string, vector<vector<int>*>*>* document_to_codes_map = query_doc_pairs.second;
            map<std::string, torch::Tensor>* docId_emb_map = new map<std::string, torch::Tensor>();
            map<std::string, std::vector<std::vector<std::vector<float>>>*>* docId_emb_map_vec = new std::map<std::string, std::vector<std::vector<std::vector<float>>>*>;
            //we loop over a single document for all the topK documents for one query string
            for (auto& doc_codes_pairs : *document_to_codes_map) {
                string doc_id = doc_codes_pairs.first;
                cout << "Processing for Document: " << doc_id << " "  << std::endl;
                vector<vector<int>*>* codes_vec = doc_codes_pairs.second;
                vector<int>* tokens = new vector<int>();
                //we fetch the first code that is the token id for static embedding and save the first codes to form a token vector containing static embedding token ids 
                for(auto& token_vec: *codes_vec){
                    tokens->push_back(token_vec->front());
                    token_vec->erase(token_vec->begin());   
                }

                //convert static embedding token ids vector to a tensor and compute the static embedding for the document
                auto token_tensor = torch::from_blob(tokens->data(), {(std::int64_t)tokens->size()}, torch::kInt);
                torch::Tensor static_embs = non_contextual_embedding->forward(token_tensor);
                std::cout << "decoder.cc::static_embs tensor size: " << static_embs.sizes() << std::endl;

                std::vector<std::vector<float>> static_embs_vec(tokens->size(), std::vector<float>((*static_embeddings_vec)[0].size()));
                for (int i = 0; i < tokens->size(); ++i) {
                    int token_index = (*tokens)[i];
                    static_embs_vec[i] = (*static_embeddings_vec)[token_index];
                }

                // // print to check values
                // // Printing values in static_embeddings
                // std::cout << "Values in static_embs:" << std::endl;
                // for (int i = 0; i < static_embs.size(0); ++i) {
                //     std::cout << static_embs.index({i, 0}).item<float>() << " " << static_embs.index({i, 1}).item<float>();
                //     std::cout << std::endl;
                //     if (i>10) break;
                // }

                // // // Printing values in static_embeddings_vec
                // std::cout << "Values in static_embs_vec:" << std::endl;
                // int i = 0;
                // for (const auto& row : static_embs_vec) {
                //     std::cout << row[0] << " " << row[1];
                //     std::cout << std::endl;
                //     if (i>10) break;
                //     i++;
                // }
                std::cout << "decoder.cc::codebook_vec size: " << (*codebook_vec).size() << " " << ((*codebook_vec)[0].size()) << " " << (*codebook_vec)[0][0].size() << " " << std::endl;

                std::cout << "decoder.cc::codes_vec size: " << codes_vec->size() << " " << ((*codes_vec)[0]->size()) << std::endl;
                // std::cout << "First codes_vec:" << std::endl;
                // for (int i=0; i < (*codes_vec)[0]->size(); i++)
                //     std::cout << (*(*codes_vec)[0])[i] << " ";
                // std::cout << std::endl;
                //compute the approx embeddings for the document using the codes and codebook
                auto* linear_codes_vec = linearize_vector_of_vectors(codes_vec);
                // std::cout << "decoder.cc::linear_codes_vec tensor size: " << linear_codes_vec->size() << std::endl;
                auto options_int = torch::TensorOptions().dtype(torch::kInt);
                auto codes = torch::from_blob(linear_codes_vec->data(),
                                  {1, int(linear_codes_vec->size())}, options_int).view({(std::int64_t)codes_vec->size(), (std::int64_t)(*codes_vec)[0]->size()});
                // std::cout << "decoder.cc::codes tensor size: " << codes.sizes() << std::endl;
                
                auto code_sparse = torch::zeros({codes.size(0), (std::int64_t)M_, (std::int64_t)K_}, torch::kFloat);
                auto indices = codes.unsqueeze(2).to(torch::kLong);
                code_sparse.scatter_(-1, indices, 1.0);
                // std::cout << "decoder.cc::code_sparse tensor size: " << code_sparse.sizes() << std::endl;
                
                auto decoded = torch::matmul(*codebook, code_sparse.unsqueeze(-1)).squeeze(-1);
                // std::cout << "decoder.cc::decoded tensor size: " << decoded.sizes() << std::endl;
                auto codeapprox = decoded.reshape({decoded.size(0), (std::int64_t)M_*(std::int64_t)codebook_dim_});
                // std::cout << "decoder.cc::codeapprox tensor size: " << codeapprox.sizes() << std::endl;

                //static_embs and approx_codes are concatenated
                auto cat_res = torch::cat({codeapprox, static_embs}, 1);
                std::cout << "decoder.cc::cat_res tensor size: " << cat_res.sizes() << std::endl;

                std::vector<std::vector<float>> cat_res_vec(codes_vec->size(), std::vector<float>(((std::int64_t)M_*(std::int64_t)codebook_dim_)* 2));
                for (int token_num=0; token_num < codes_vec->size(); token_num++) {
                    int cat_res_vec_index = 0;
                    for (int i=0; i<M_; i++) {
                        for (int j=0; j<codebook_dim_; j++) {
                            cat_res_vec[token_num][cat_res_vec_index] = (*codebook_vec)[i][j][(*(*codes_vec)[token_num])[i]];
                            cat_res_vec_index++;
                        }
                    }
                    for (int i=0; i < static_embs_vec[0].size(); i++){
                        cat_res_vec[token_num][M_*codebook_dim_ + i] = static_embs_vec[token_num][i];
                    }
                }

                // // print to check values
                // // Printing values in cat_res
                // std::cout << "Values in cat_res tensor:" << std::endl;
                // for (int i = 0; i < cat_res.size(0); ++i) {
                //     std::cout << cat_res.index({i, 0}).item<float>() << " " << cat_res.index({i, cat_res.size(1) - 1}).item<float>();
                //     std::cout << std::endl;
                //     if (i>10) break;
                // }
                // std::cout << cat_res.index({cat_res.size(0) - 1, 0}).item<float>() << " " << cat_res.index({cat_res.size(0) - 1, cat_res.size(1) - 1}).item<float>();

                // // // Printing values in cat_res_vec
                // std::cout << "Values in cat_res_vec:" << std::endl;
                // int i = 0;
                // for (const auto& row : cat_res_vec) {
                //     std::cout << row[0] << " " << row[row.size() - 1];
                //     std::cout << std::endl;
                //     if (i>10) break;
                //     i++;
                // }
                // std::cout << cat_res_vec[cat_res_vec.size()-1][0] << " " << cat_res_vec[cat_res_vec.size()-1][cat_res_vec[0].size() - 1];
                
                //composition layer is applied to get one final approx decoded embeddings for document
                auto composition_result = composition_layer->forward(cat_res);
                std::cout << "decoder.cc::composition_result tensor size: " << composition_result.sizes() << std::endl;
                composition_result = composition_result.unsqueeze(0);
                std::cout << "decoder.cc::composition_result tensor size: " << composition_result.sizes() << std::endl;

                //all tensors are made of same shape [1 * doc_maxlen * dim_size]
                torch::Tensor full_tensor = torch::zeros({1, doc_maxlen_, dimension_size_});
                std::cout << "decoder.cc::full_tensor tensor size: " << full_tensor.sizes() << std::endl;
                full_tensor.slice(1, 0, tokens->size()) = composition_result;
                std::cout << "decoder.cc::full_tensor tensor size after slice: " << full_tensor.sizes() << std::endl;

                std::vector<std::vector<std::vector<float>>>* full_tensor_vec;
                full_tensor_vec = new std::vector<std::vector<std::vector<float>>>(full_tensor.size(0), std::vector<std::vector<float>>((full_tensor.size(1)), std::vector<float>((full_tensor.size(2)))));
                for (int i = 0; i < full_tensor.size(0); ++i) {
                    for (int j = 0; j < full_tensor.size(1); ++j) {
                        for (int k = 0; k < full_tensor.size(2); ++k) {
                            (*full_tensor_vec)[i][j][k] = full_tensor.index({i, j, k}).item<float>();
                        }
                    }
                }

                // // print to check values
                // // Printing values in cat_res
                // std::cout << "Values in cat_res tensor:" << std::endl;
                // for (int i = 0; i < cat_res.size(0); ++i) {
                //     std::cout << cat_res.index({i, 0}).item<float>() << " " << cat_res.index({i, cat_res.size(1) - 1}).item<float>();
                //     std::cout << std::endl;
                //     if (i>10) break;
                // }
                // std::cout << cat_res.index({cat_res.size(0) - 1, 0}).item<float>() << " " << cat_res.index({cat_res.size(0) - 1, cat_res.size(1) - 1}).item<float>();

                // // // Printing values in cat_res_vec
                // std::cout << "Values in cat_res_vec:" << std::endl;
                // int i = 0;
                // for (const auto& row : cat_res_vec) {
                //     std::cout << row[0] << " " << row[row.size() - 1];
                //     std::cout << std::endl;
                //     if (i>10) break;
                //     i++;
                // }
                // std::cout << cat_res_vec[cat_res_vec.size()-1][0] << " " << cat_res_vec[cat_res_vec.size()-1][cat_res_vec[0].size() - 1];


                docId_emb_map->insert(make_pair(doc_id, full_tensor));
                docId_emb_map_vec->insert(make_pair(doc_id, full_tensor_vec));


                // // // Calculate composition_result (vector)
                // std::vector<float> composition_result_vec(dimension_size_, 0.0);
                // for (int i = 0; i < cat_res_vec.size(); i++) {
                //     for (int j = 0; j < dimension_size_; j++) {
                //         composition_result_vec[j] += cat_res_vec[i][j] * (*composition_weights_vec)[i][j];
                //     }
                // }
                // for (int i = 0; i < dimension_size_; i++) {
                //     composition_result_vec[i] = std::tanh(composition_result_vec[i] + composition_bias_vec[i]);
                // }

                // // Create full_tensor
                // std::vector<std::vector<float>> full_tensor(doc_maxlen_, std::vector<float>(dimension_size_, 0.0));
                // for (int i = 0; i < tokens->size(); i++) {
                //     for (int j = 0; j < dimension_size_; j++) {
                //         full_tensor[i][j] = composition_result_vec[j];
                //     }
                // }

                // // Convert full_tensor to a flat vector
                // std::vector<float> final_result;
                // for (const auto& row : full_tensor) {
                //     final_result.insert(final_result.end(), row.begin(), row.end());
                // }

                delete tokens;
            }           
            query_doc_approx_emb_map->insert(make_pair(query_id, docId_emb_map)); 
            query_doc_approx_emb_map_vec->insert(make_pair(query_id, docId_emb_map_vec));     
        }
        return query_doc_approx_emb_map_vec;
    }
}
   

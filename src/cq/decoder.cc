#include "decoder.h"

#include<iostream>

using namespace std;

namespace lh{
    
    Decoder::Decoder(){
        code_fetcher_ = new CodeFetcher();
        vocab_size_ = VOCAB_SIZE;
        dimension_size_ = DIMENSION_SIZE;
        pad_token_id_ = PAD_TOKEN_ID; 
        M_ = CODEBOOK_COUNT;
        K_ = CODES_COUNT;
        codebook_dim_ = CODEBOOK_DIM;
        doc_maxlen_ = DOC_MAXLEN;
    }

  
    Decoder::~Decoder(){
        delete code_fetcher_;
    }    

     map<string, torch::Tensor> Decoder::decode(){
        auto static_embs_vec = get_vec_from_file("../model/non_contextual_embeddings.txt");
        auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
        auto static_embeddings = torch::from_blob(static_embs_vec.data(),
                            {1, int(static_embs_vec.size())}, options).view({(std::int64_t)vocab_size_, (std::int64_t)dimension_size_});
        auto embedding_options = torch::nn::EmbeddingOptions(vocab_size_, dimension_size_).padding_idx(pad_token_id_)._weight(static_embeddings);
        auto non_contextual_embedding = new torch::nn::EmbeddingImpl(embedding_options);
       
        auto codebook_vec = get_vec_from_file("../model/codebook.txt");

        auto codebook = torch::from_blob(codebook_vec.data(),
                            {1, int(codebook_vec.size())}, options).view({M_, codebook_dim_, K_});
             
        auto composition_weights_vec = get_vec_from_file("../model/composition_c_e_linear_weights.txt");
        auto composition_weights = torch::from_blob(composition_weights_vec.data(),
                            {1, int(composition_weights_vec.size())}, options).view({dimension_size_, 2*dimension_size_});
        
        auto composition_bias_vec = get_vec_from_file("../model/composition_c_e_linear_bias.txt");
        auto composition_bias = torch::from_blob(composition_bias_vec.data(),
                            {1, int(composition_bias_vec.size())}, options).view({dimension_size_});
        
        auto composition_layer = new torch::nn::LinearImpl(torch::nn::LinearOptions(2*dimension_size_, dimension_size_).bias(true));
        composition_layer->weight = composition_weights;
        composition_layer->bias = composition_bias;

        map<string, map<string, vector<vector<int>>>> fetched_codes = code_fetcher_->fetch_codes();
        map<string, torch::Tensor> query_doc_approx_emb_map;

        for (auto&  query_doc_pairs : fetched_codes) {
            map<string, vector<vector<int>>>& document_to_codes_map = query_doc_pairs.second;
            std::vector<torch::Tensor> approx_tensors;
            for (auto& doc_codes_pairs : document_to_codes_map) {
                vector<vector<int>>& codes_vec = doc_codes_pairs.second;
                vector<int> tokens;
                for(auto& token_vec: codes_vec){
                    tokens.push_back(token_vec.front());
                    token_vec.erase(token_vec.begin());   
                }
                
                auto token_tensor = torch::from_blob(tokens.data(), {(std::int64_t)tokens.size()}, torch::kInt);
                auto static_embs = non_contextual_embedding->forward(token_tensor);
                
                auto linear_codes_vec = linearize_vector_of_vectors(codes_vec);
                auto options_three = torch::TensorOptions().dtype(torch::kInt);
                auto codes = torch::from_blob(linear_codes_vec.data(),
                                  {1, int(linear_codes_vec.size())}, options_three).view({(std::int64_t)codes_vec.size(), (std::int64_t)codes_vec[0].size()});
                auto code_sparse = torch::zeros({codes.size(0), (std::int64_t)M_, (std::int64_t)K_}, torch::kFloat);
                auto indices = codes.unsqueeze(2).to(torch::kLong);
                code_sparse.scatter_(-1, indices, 1.0);
                
                auto decoded = torch::matmul(codebook, code_sparse.unsqueeze(-1)).squeeze(-1);
                auto codeapprox = decoded.reshape({decoded.size(0), (std::int64_t)M_*(std::int64_t)codebook_dim_});
                
                auto cat_res = torch::cat({codeapprox, static_embs}, 1);
 
                auto composition_result = composition_layer->forward(cat_res);
                composition_result = composition_result.unsqueeze(0);
                
               
                torch::Tensor full_tensor = torch::zeros({1, doc_maxlen_, dimension_size_});
                full_tensor.slice(1, 0, tokens.size()) = composition_result;
                approx_tensors.push_back(full_tensor);
            }
            auto doc_emb_approx = torch::cat(approx_tensors, 0);
            doc_emb_approx = torch::nn::functional::normalize(doc_emb_approx,
                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(2)); 
            query_doc_approx_emb_map.insert(make_pair(query_doc_pairs.first, doc_emb_approx));      
        }
        return query_doc_approx_emb_map;
    }
}
   

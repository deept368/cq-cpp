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
    
    /**
    Fetches codes for topK documents for all query input strings using the CodeFetcher class. Processes them and generate 
    approx embeddings for all the documents using static embeddings and embeddings decoded from codebook.
    @param void 
    @return A map of query input strings and the approx embedding tensor of their correpsonding topK documents.
    */

     map<string, torch::Tensor> Decoder::decode(){

        //static embedding is fetched for all the vocabulary into to linear vec. size (30522(vocab_size) * 128(embedding_dim))
        auto static_embs_vec = get_vec_from_file("../model/non_contextual_embeddings.txt");
        
        //linear vec of static embeddings is converted to a 2-d tensor of size [vocab_size_(30522) * dimesnion_size_(128)] 
        auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
        auto static_embeddings = torch::from_blob(static_embs_vec.data(),
                            {1, int(static_embs_vec.size())}, options).view({(std::int64_t)vocab_size_, (std::int64_t)dimension_size_});
        
        //torch::nn::EmbeddingImpl(PyTorch C++) model is initialised and static_embeddings tensor is loaded as pretrained weight. 
        auto embedding_options = torch::nn::EmbeddingOptions(vocab_size_, dimension_size_).padding_idx(pad_token_id_)._weight(static_embeddings);
        auto non_contextual_embedding = new torch::nn::EmbeddingImpl(embedding_options);
       
       //codebook is loaded as a linear vec and converted to 3-d tensor of size M * codebook_dim_(8) * K
        auto codebook_vec = get_vec_from_file("../model/codebook.txt");
        auto codebook = torch::from_blob(codebook_vec.data(),
                            {1, int(codebook_vec.size())}, options).view({M_, codebook_dim_, K_});
             
        //composition layers weights are loaded as a linear vec and converted to a 2-d Tensor of [dim_size(128) * (2*dim_size)]
        auto composition_weights_vec = get_vec_from_file("../model/composition_c_e_linear_weights.txt");
        auto composition_weights = torch::from_blob(composition_weights_vec.data(),
                            {1, int(composition_weights_vec.size())}, options).view({dimension_size_, 2*dimension_size_});
        
        //composition layers bias are loaded as a linear vec and converted to a 1-d Tensor of [dim_size(128)]
        auto composition_bias_vec = get_vec_from_file("../model/composition_c_e_linear_bias.txt");
        auto composition_bias = torch::from_blob(composition_bias_vec.data(),
                            {1, int(composition_bias_vec.size())}, options).view({dimension_size_});
        
        //torch::nn::LinearImpl(PyTorch C++) is used to initialise a linear compositon layers and weights and bias are set
        auto composition_layer = new torch::nn::LinearImpl(torch::nn::LinearOptions(2*dimension_size_, dimension_size_).bias(true));
        composition_layer->weight = composition_weights;
        composition_layer->bias = composition_bias;

        /* 
        code_fetcher object is used to fetch a map of queries and their corresponding top K documents and (codes and tokens) of these topK documents.
        codes and tokens of each document are fetched as vec<vec<int>> (dimensions: num of tokens * (K+1)) where 1 in (K+1) is used for static embedding token id. 
        */
        map<string, map<string, vector<vector<int>>>> fetched_codes = code_fetcher_->fetch_codes();
        map<string, torch::Tensor> query_doc_approx_emb_map;

        //we loop over each query string
        for (auto&  query_doc_pairs : fetched_codes) {
            map<string, vector<vector<int>>>& document_to_codes_map = query_doc_pairs.second;
            std::vector<torch::Tensor> approx_tensors;
            //we loop over a single document for all the topK documents for one query string
            for (auto& doc_codes_pairs : document_to_codes_map) {
                vector<vector<int>>& codes_vec = doc_codes_pairs.second;
                vector<int> tokens;
                //we fetch the first code that is the token id for static embedding and save the first codes to form a token vector containing static embedding token ids 
                for(auto& token_vec: codes_vec){
                    tokens.push_back(token_vec.front());
                    token_vec.erase(token_vec.begin());   
                }
                
                //convert static embedding token ids vector to a tensor and compute the static embedding for the document
                auto token_tensor = torch::from_blob(tokens.data(), {(std::int64_t)tokens.size()}, torch::kInt);
                auto static_embs = non_contextual_embedding->forward(token_tensor);
                
                //compute the approx embeddings for the document using the codes and codebook
                auto linear_codes_vec = linearize_vector_of_vectors(codes_vec);
                auto options_int = torch::TensorOptions().dtype(torch::kInt);
                auto codes = torch::from_blob(linear_codes_vec.data(),
                                  {1, int(linear_codes_vec.size())}, options_int).view({(std::int64_t)codes_vec.size(), (std::int64_t)codes_vec[0].size()});
                auto code_sparse = torch::zeros({codes.size(0), (std::int64_t)M_, (std::int64_t)K_}, torch::kFloat);
                auto indices = codes.unsqueeze(2).to(torch::kLong);
                code_sparse.scatter_(-1, indices, 1.0);
                
                auto decoded = torch::matmul(codebook, code_sparse.unsqueeze(-1)).squeeze(-1);
                auto codeapprox = decoded.reshape({decoded.size(0), (std::int64_t)M_*(std::int64_t)codebook_dim_});
                

                //static_embs and approx_codes are concatenated
                auto cat_res = torch::cat({codeapprox, static_embs}, 1);
                
                //composition layer is applied to get one final approx decoded embeddings for document
                auto composition_result = composition_layer->forward(cat_res);
                composition_result = composition_result.unsqueeze(0);
                
                //all tensors are made of same shape [1 * doc_maxlen * dim_size]
                torch::Tensor full_tensor = torch::zeros({1, doc_maxlen_, dimension_size_});
                full_tensor.slice(1, 0, tokens.size()) = composition_result;
                approx_tensors.push_back(full_tensor);
            }
            //all approx embedding tensors for topk documents for a query are concatenated and normailised and put in a map correponding to the query string
            auto doc_emb_approx = torch::cat(approx_tensors, 0);
            doc_emb_approx = torch::nn::functional::normalize(doc_emb_approx,
                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(2)); 
            query_doc_approx_emb_map.insert(make_pair(query_doc_pairs.first, doc_emb_approx));      
        }
        return query_doc_approx_emb_map;
    }
}
   

#include "decoder.h"
#include "queryprocessor.h"

#include <torch/script.h>
#include<iostream>
#include<unordered_map>

using namespace std;

namespace lh{
    
    Decoder::Decoder(){
        query_processor_ = new QueryProcessor();
        vocab_size_ = VOCAB_SIZE;
        dimension_size_ = DIMENSION_SIZE;
        pad_token_id_ = PAD_TOKEN_ID; 
        M_ = CODEBOOK_COUNT;
        K_ = CODES_COUNT;
        codebook_dim_ = CODEBOOK_DIM;
        doc_maxlen_ = DOC_MAXLEN;
    }

  
    Decoder::~Decoder(){
        delete query_processor_;
    }    
    
    /**
    Fetches codes for topK documents for all query input strings using the CodeFetcher class. Processes them and generate 
    approx embeddings for all the documents using static embeddings and embeddings decoded from codebook.
    @param void 
    @return A map of query input strings and the approx embedding tensor of their correpsonding topK documents.
    */

     map<std::size_t, map<std::string,torch::Tensor>> Decoder::decode(){

        #ifdef PRFILE_CQ
            auto begin = std::chrono::system_clock::now();
        #endif
        
        #ifdef PRFILE_CQ
            auto begin_1 = std::chrono::system_clock::now();
        #endif

        //static embedding is fetched for all the vocabulary from a .pt file into a tensor [30522(vocab_size) * 128(embedding_dim)]
        torch::Tensor static_embeddings;
        torch::load(static_embeddings, "../model/non_contextual_embeddings.pt");

         #ifdef PRFILE_CQ
            auto end_1 = std::chrono::system_clock::now();
            std::cout<<"static_embs loading time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_1-begin_1).count())/1000 << std::endl;
        #endif

        //torch::nn::EmbeddingImpl(PyTorch C++) model is initialised and static_embeddings tensor is loaded as pretrained weight. 
        auto embedding_options = torch::nn::EmbeddingOptions(vocab_size_, dimension_size_).padding_idx(pad_token_id_)._weight(static_embeddings);
        auto non_contextual_embedding = new torch::nn::EmbeddingImpl(embedding_options);
       
        #ifdef PRFILE_CQ
            auto begin_2 = std::chrono::system_clock::now();
        #endif

       //codebook is loaded. size: [M * codebook_dim_(8) * K]
        torch::Tensor codebook;
        torch::load(codebook, "../model/codebook.pt");

        #ifdef PRFILE_CQ
            auto end_2 = std::chrono::system_clock::now();
            std::cout<<"codebok loading time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_2-begin_2).count())/1000 << std::endl;
        #endif

        //composition layers weights are loaded as a 2-d Tensor of [dim_size(128) * (2*dim_size)]
        torch::Tensor composition_weights;
        torch::load(composition_weights, "../model/composition_c_e_linear_weights.pt");

        //composition layer bias are loaded as a 1-d Tensor of [dim_size(128)]
        torch::Tensor composition_bias;
        torch::load(composition_bias, "../model/composition_c_e_linear_bias.pt");

        //torch::nn::LinearImpl(PyTorch C++) is used to initialise a linear compositon layers and weights and bias are set
        auto composition_layer = new torch::nn::LinearImpl(torch::nn::LinearOptions(2*dimension_size_, dimension_size_).bias(true));
        composition_layer->weight = composition_weights;
        composition_layer->bias = composition_bias;

        /* 
        code_fetcher object is used to fetch a map of queries and their corresponding top K documents and (codes and tokens) of these topK documents.
        codes and tokens of each document are fetched as vec<vec<int>> (dimensions: num of tokens * (K+1)) where 1 in (K+1) is used for static embedding token id. 
        */
        unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> fetched_codes = query_processor_->getCodes();
       // cout<<"map "<<fetched_codes[12345]["doc0"]<<endl;
        map<std::size_t, map<std::string,torch::Tensor>> query_doc_approx_emb_map;

         #ifdef PRFILE_CQ
            auto begin_3 = std::chrono::system_clock::now();
        #endif

        //we loop over each query string
        for (auto&  query_doc_pairs : fetched_codes) {
            unordered_map<string, vector<vector<std::size_t>>>& document_to_codes_map = query_doc_pairs.second;
            map<std::string, torch::Tensor> docId_emb_map;
            //we loop over a single document for all the topK documents for one query string
            for (auto& doc_codes_pairs : document_to_codes_map) {
                vector<vector<std::size_t>>& codes_vec = doc_codes_pairs.second;
                vector<std::size_t> tokens;
                //we fetch the first code that is the token id for static embedding and save the first codes to form a token vector containing static embedding token ids 
                for(auto& token_vec: codes_vec){
                    tokens.push_back(token_vec.front());
                    token_vec.erase(token_vec.begin());   
                }

                // if(doc_codes_pairs.first == "doc0"){
                //     cout<<tokens<<endl;
                // }

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
                
                //  if(doc_codes_pairs.first == "doc0"){
                //     cout<<"static "<<codeapprox[0][1]<<endl;
                //      cout<<"static "<<codeapprox[0][2]<<endl;
                //       cout<<"static "<<codeapprox[1][1]<<endl;
                //        cout<<"static "<<codeapprox[55][0]<<endl;
                //        cout<<"static "<<codeapprox[55][127]<<endl;
                // }


                //static_embs and approx_codes are concatenated
                auto cat_res = torch::cat({codeapprox, static_embs}, 1);
                
                //composition layer is applied to get one final approx decoded embeddings for document
                auto composition_result = composition_layer->forward(cat_res);
                composition_result = composition_result.unsqueeze(0);
                
                //all tensors are made of same shape [1 * doc_maxlen * dim_size]
                torch::Tensor full_tensor = torch::zeros({1, doc_maxlen_, dimension_size_});
                full_tensor.slice(1, 0, tokens.size()) = composition_result;
                // cout<<doc_codes_pairs.first<<" "<<full_tensor[0][0][0]<<endl;
                // cout<<doc_codes_pairs.first<<" "<<full_tensor[0][0][1]<<endl;
                // cout<<doc_codes_pairs.first<<" "<<full_tensor[0][36][127]<<endl;
                // cout<<doc_codes_pairs.first<<" "<<full_tensor[0][55][127]<<endl;
                docId_emb_map.insert(make_pair(doc_codes_pairs.first, full_tensor));
            }           
            query_doc_approx_emb_map.insert(make_pair(query_doc_pairs.first, docId_emb_map));      
        }

        #ifdef PRFILE_CQ
            auto end_3 = std::chrono::system_clock::now();
            std::cout<<"just decoding computation time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end_3-begin_3).count())/1000 << std::endl;
        #endif

        #ifdef PRFILE_CQ
            auto end = std::chrono::system_clock::now();
            std::cout<<"total document decoding time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif

        return query_doc_approx_emb_map;
    }
}
   

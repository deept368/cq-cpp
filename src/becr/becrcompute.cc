// std::vector<std::vector<std::string>>* input_tokens = new std::vector<std::vector<std::string>>(curr_batch_size);
//         for (std::size_t i = 0; i < curr_batch_size; i++){
//             tokenizer_->tokenize((*input_string)[i].c_str(), &(*input_tokens)[i], query_maxlen);
//             if(isQuery){
//                 (*input_tokens)[i].insert((*input_tokens)[i].begin(), "[unused0]");
//             }
//             (*input_tokens)[i].insert((*input_tokens)[i].begin(), "[CLS]");    
//             (*input_tokens)[i].push_back("[SEP]");
//         }


#include "becrcompute.h"
#include <fstream>
#include <iostream>
#include <chrono>

namespace lh{
   
    template<class T>
    BecrCompute<T>::BecrCompute(){
        // Model model;
        // Graph<float> graph;
        // std::fstream input("../model/model.proto", std::ios::in | std::ios::binary);
        // if (!model.ParseFromIstream(&input)) {
        //     throw std::invalid_argument("can not read protofile");
        // }
        // for (std::size_t i = 0; i < model.param_size(); i++){
        //     Model_Paramter paramter = model.param(i);
        //     int size = 1;
        //     std::vector<std::size_t> dims(paramter.n_dim());
        //     for (int j = 0; j < paramter.n_dim(); j++)
        //     {
        //         int dim = paramter.dim(j);
        //         size *= dim;
        //         dims[j] = dim;
        //     }
        //     float *data = new float[size];
        //     for (int i = 0; i < size; i++) {
        //         data[i] = paramter.data(i);
        //     }
        //     graph[paramter.name()] = make_pair(dims, data);
        // }
        // google::protobuf::ShutdownProtobufLibrary();
        // std::cout << "Successfully loaded parameters from protobuf!" << std::endl;

        // size_t pre_batch_size = PRE_BATCH_SIZE;
        // size_t pre_seq_len = 512;
        // size_t num_heads = 12;
        // size_t head_hidden_size = 64;
        // size_t intermediate_ratio = 4;
        // size_t num_layers = 12;

        // hidden_size_ = HIDDEN_SIZE;
        query_maxlen = QUERY_MAXLEN;

        // std::vector<std::string> names;
        // names.push_back("bert.embeddings.word_embeddings.weight");
        // names.push_back("bert.embeddings.position_embeddings.weight");
        // names.push_back("bert.embeddings.token_type_embeddings.weight");
        // names.push_back("bert.embeddings.LayerNorm.gamma");
        // names.push_back("bert.embeddings.LayerNorm.beta");
        // for (std::size_t idx; idx < num_layers; idx++){
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.query.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.query.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.key.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.key.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.value.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.self.value.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.output.dense.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.output.dense.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.output.LayerNorm.gamma");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".attention.output.LayerNorm.beta");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".intermediate.dense.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".intermediate.dense.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".output.dense.weight");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".output.dense.bias");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".output.LayerNorm.gamma");
        //     names.push_back("bert.encoder.layer." + std::to_string(idx) + ".output.LayerNorm.beta");
        // }
        // names.push_back("bert.pooler.dense.weight");
        // names.push_back("bert.pooler.dense.bias");

        // bert_ = new Bert<T>(names, graph, pre_batch_size, pre_seq_len, hidden_size_, num_heads, head_hidden_size, intermediate_ratio, num_layers);
        tokenizer_ = new FullTokenizer("../model/vocab.txt");

        std::cout << "Init model from protobuf file and tokenizer successfully!" << std::endl;
    }

    template<class T>
    BecrCompute<T>::~BecrCompute(){
        
        delete tokenizer_;
        
    }    

    /**
     Computes the BERT embedding of given input strings. Treats query and document differently. Currently support is ony added 
     for queries as this is only required for CQ.
     @param input_string the input strings to encode
     @param isQuery whether the input strings are queries or documents
     @return linear a vector containing the BERT embeddings of the input strings of size (BATCH_SIZE * QUERY_MAXLEN * HIDDEN_DIM_SIZE(768)) 
    */
    template<class T>
    std::vector<T>* BecrCompute<T>::compute(std::vector<std::string>* input_string, bool isQuery){
        
        //computing the batch size
        int curr_batch_size = input_string->size();

        //necessary tokens are added and the input strings are converted to tokens
        std::vector<std::vector<std::string>>* input_tokens = new std::vector<std::vector<std::string>>(curr_batch_size);
        for (std::size_t i = 0; i < curr_batch_size; i++){
            tokenizer_->tokenize((*input_string)[i].c_str(), &(*input_tokens)[i], query_maxlen);
            if(isQuery){
                (*input_tokens)[i].insert((*input_tokens)[i].begin(), "[unused0]");
            }
            (*input_tokens)[i].insert((*input_tokens)[i].begin(), "[CLS]");    
            (*input_tokens)[i].push_back("[SEP]");
        }

        //mask is computed and padding is applied to input strings
        uint64_t* mask = new uint64_t[curr_batch_size];
        for (std::size_t i = 0; i < curr_batch_size; i++){
            mask[i] = (*input_tokens)[i].size();
            for (int j = (*input_tokens)[i].size(); j < query_maxlen; j++){
                (*input_tokens)[i].push_back("[MASK]");
            }
        }

        //token ids are computed using tokens. vocab.txt is used for the same.
        uint64_t* input_ids = new uint64_t[curr_batch_size * query_maxlen];
        uint64_t* position_ids = new uint64_t[curr_batch_size * query_maxlen];
        uint64_t* type_ids = new uint64_t[curr_batch_size * query_maxlen];
        for (std::size_t i = 0; i < curr_batch_size; i++){
            tokenizer_->convert_tokens_to_ids((*input_tokens)[i], input_ids + i * query_maxlen);
            for (int j = 0; j < query_maxlen; j++){
                position_ids[i * query_maxlen + j] = j;
                type_ids[i * query_maxlen + j] = 0;
            }
        }
        
        //bert compute is called to generate the embeddings. output is stored in 1-d array seq_output, size: batch_size*query_maxlen*768(hidden_dim)
        std::size_t size = curr_batch_size * query_maxlen * hidden_size_;
        T* pool_output_ = new T[curr_batch_size * hidden_size_];  
        T* seq_output_= new T[size];    

        bert_->compute(curr_batch_size, query_maxlen, input_ids, position_ids, type_ids, mask, seq_output_, pool_output_);
        //array output is converted to vector before returning 
        vector<T>* result = convert_to_vector(seq_output_, size);

        delete[] mask;
        delete[] input_ids;
        delete[] position_ids;
        delete[] type_ids;
        delete[] seq_output_;
        delete[] pool_output_;
        delete input_tokens;

        return result;
        
    }
    template class BertCompute<float>;
}

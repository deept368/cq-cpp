#include "queryencoder.h"
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "../utils.h"

using namespace std;

namespace lh{

    template<class T>
    QueryEncoder<T>::QueryEncoder(){
        query_maxlen = QUERY_MAXLEN;
        hidden_size_ = HIDDEN_SIZE;
        dimension_size_ = DIMENSION_SIZE;
        bert_compute_ = new BertCompute<T>();
    }

    template<class T>
    QueryEncoder<T>::~QueryEncoder(){
        delete bert_compute_;  
        
    }    

    /**

    Encodes the input strings using the specified BERT model and linear layer weights, and returns a tensor containing the encoded query vectors.
    The method first computes the BERT embeddings for the input strings using the specified BERT model, and then applies a linear layer
    ith the specified weights to the BERT embeddings to obtain the encoded query vectors. The resulting vectors are then normalized using L2 normalization.
    @param input_strings A vector of input strings to encode
    @return A tensor containing the encoded query vectors of size BATCH_SIZE * query_maxlen(32) * EMBEDDING_DIM(128)
    */

    template<class T>
    torch::Tensor QueryEncoder<T>::encode(std::vector<std::string> input_strings){
        
         #ifdef PRFILE_CQ
            auto begin = std::chrono::system_clock::now();
        #endif

        //bert embeddings are computed for all the query strings and converted to tensor
        std::size_t batch_size = input_strings.size();
        std::vector<T> vec_bert_output= bert_compute_->compute(input_strings, true);
        
        auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
        auto bert_output_tensor = torch::from_blob(vec_bert_output.data(),
                                  {1, int(vec_bert_output.size())}, options).view({(std::int64_t)batch_size, (std::int64_t)query_maxlen, (std::int64_t)hidden_size_});

        // cout<<bert_output_tensor[0][0][0]<<endl;
        // cout<<bert_output_tensor[0][0][1]<<endl;
        // cout<<bert_output_tensor[0][1][2]<<endl;
        // cout<<bert_output_tensor[0][31][0]<<endl;
        // cout<<bert_output_tensor[0][31][1]<<endl;
        // cout<<bert_output_tensor[0][31][767]<<endl;

        //linear model is loaded and bert_output is passed through the linear layer to reduce dim size from 768 to 128
        torch::Tensor linear_layer_weight_tensor;
        torch::load(linear_layer_weight_tensor, "../model/colbert_linear_layer_weights.pt");
        auto linear_model_ = new torch::nn::LinearImpl(torch::nn::LinearOptions(hidden_size_, dimension_size_).bias(false));
        linear_model_->weight = linear_layer_weight_tensor;
        auto linear_output = linear_model_->forward(bert_output_tensor);

        //finally, linear_ouptut is normalised and returned
        auto normalised_output = torch::nn::functional::normalize(linear_output,
                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));  

        // cout<<normalised_output[0][0][0]<<endl;
        // cout<<normalised_output[0][0][1]<<endl;
        // cout<<normalised_output[0][1][2]<<endl;
        // cout<<normalised_output[0][12][0]<<endl;
        // cout<<normalised_output[0][12][1]<<endl;
        // cout<<normalised_output[0][12][127]<<endl;
         #ifdef PRFILE_CQ
            auto end = std::chrono::system_clock::now();
            std::cout<<"query encoding time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif
        
      

        return normalised_output;
    }
    
    template class QueryEncoder<float>;
}
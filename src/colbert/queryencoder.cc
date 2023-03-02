#include "queryencoder.h"
#include <torch/torch.h>
#include <iostream>
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


    template<class T>
    torch::Tensor QueryEncoder<T>::encode(std::vector<std::string> input_strings){
        
        std::size_t batch_size = input_strings.size();
        std::vector<T> vec_bert_output= bert_compute_->compute(input_strings, true);
        
        auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
        auto bert_output_tensor = torch::from_blob(vec_bert_output.data(),
                                  {1, int(vec_bert_output.size())}, options).view({(std::int64_t)batch_size, (std::int64_t)query_maxlen, (std::int64_t)hidden_size_});
 
        std::vector<T> weight_vec = get_vec_from_file("../model/colbert_linear_layer_weights.txt");
        auto linear_weight_tensor = torch::from_blob(weight_vec.data(),
                                  {1, int(weight_vec.size())}, options).view({(std::int64_t)dimension_size_, (std::int64_t)hidden_size_});
        auto linear_model_ = new torch::nn::LinearImpl(torch::nn::LinearOptions(hidden_size_, dimension_size_).bias(false));
        linear_model_->weight = linear_weight_tensor;
        auto linear_output = linear_model_->forward(bert_output_tensor);

        auto normalised_output = torch::nn::functional::normalize(linear_output,
                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));  
        return normalised_output;
    }
    
    template class QueryEncoder<float>;
}
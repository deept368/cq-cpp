#include "queryencoder.h"


using namespace std;

namespace lh{

    template<class T>
    QueryEncoder<T>::QueryEncoder(){
        query_maxlen = QUERY_MAXLEN;
        hidden_size_ = HIDDEN_SIZE;
        dimension_size_ = DIMENSION_SIZE;

        if (USE_BECR)
        {
            becr_compute_ = new BecrCompute();
        }
        else
        {
            bert_compute_ = new BertCompute<T>();
        }
        

        torch::Tensor* linear_layer_weight_tensor = new torch::Tensor();
        torch::load(*linear_layer_weight_tensor, "../model/colbert_linear_layer_weights.pt");
        linear_model_ = new torch::nn::LinearImpl(torch::nn::LinearOptions(hidden_size_, dimension_size_).bias(false));
        linear_model_->weight = *linear_layer_weight_tensor;
    }

    template<class T>
    QueryEncoder<T>::~QueryEncoder(){
        if (USE_BECR)
        {
            delete becr_compute_;
        }
        else
        {
            delete bert_compute_; 
            delete linear_model_; 
        }
        
    }    

    /**

    Encodes the input strings using the specified BERT model and linear layer weights, and returns a tensor containing the encoded query vectors.
    The method first computes the BERT embeddings for the input strings using the specified BERT model, and then applies a linear layer
    ith the specified weights to the BERT embeddings to obtain the encoded query vectors. The resulting vectors are then normalized using L2 normalization.
    @param input_strings A vector of input strings to encode
    @return A tensor containing the encoded query vectors of size BATCH_SIZE * query_maxlen(32) * EMBEDDING_DIM(128)
    */

    template<class T>
    torch::Tensor QueryEncoder<T>::encode(std::vector<std::string>* input_strings){
        
        //bert embeddings are computed for all the query strings and converted to tensor
        std::size_t batch_size = input_strings->size();
        if (USE_BECR)
        {
            auto output= becr_compute_->compute(input_strings);
            auto normalised_output = torch::nn::functional::normalize(output,
                                torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
            
            return normalised_output;
        }
        
        std::vector<T>* vec_output= bert_compute_->compute(input_strings, true);
        auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
        auto bert_output_tensor = torch::from_blob(vec_output->data(),
                                {1, int(vec_output->size())}, options).view({(std::int64_t)batch_size, (std::int64_t)query_maxlen, (std::int64_t)hidden_size_});

        //linear model is loaded and bert_output is passed through the linear layer to reduce dim size from 768 to 128
        auto output = linear_model_->forward(bert_output_tensor);
        delete vec_output;

        //finally, linear_ouptut is normalised and returned
        auto normalised_output = torch::nn::functional::normalize(output,
                                torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));  
        
        return normalised_output;
    }
    
    template class QueryEncoder<float>;
}
#include "score.h"
#include<iostream>
#include<chrono>
#include <vector>
#include <cmath>

namespace lh{

    
    Score::Score(){
        similarity_metric_ = SIMILARITY_METRIC;
    }

  
    Score::~Score(){
        
    }    

    
    /**
    Computes the scores between the query tensor Q and the document embeddings tensor D, using the specified similarity metric.
    If the similarity metric is "cosine", the method computes the cosine similarity between Q and each document embedding in D,
    and returns the sum of the maximum values in each row of the resulting matrix.
    If the similarity metric is "l2", the method splits Q and D into batches of size 64, computes the L2 distance between each query and document pair
    in each batch, and returns the sum of the maximum values in each row of the resulting matrix.
    @param Q The query tensor of size QUERY_BATCH_SIZE(supports only 1) * query_maxlen(32) * embdding_dim(128) 
    @param D The document embeddings tensor of size DOCUMENT_BATCH_SIZE * doc_maxlen(180) * embdding_dim(128) 
    @return A linear tensor containing the scores between the query tensor Q and the document embeddings tensor D of size DOCUMENT_BATCH_SIZE.
    @throws An assertion error if the similarity metric is not "cosine" or "l2"
    */

    std::vector<float> Score::compute_scores(torch::Tensor Q, std::vector<torch::Tensor>& approx_tensors) {
    auto doc_emb_approx = torch::cat(approx_tensors, 0);
    auto D = torch::nn::functional::normalize(
        doc_emb_approx,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(2)
    );

    auto scores = std::get<0>(Q.matmul(D.permute({0, 2, 1})).max(2)).sum(1);

    std::cout << "score.cc::scores tensor size: " << scores.sizes() << std::endl;
    std::vector<float> scores_vec(scores.data_ptr<float>(), scores.data_ptr<float>() + scores.numel());

    return scores_vec;
}

    // std::vector<float> Score::compute_scores(std::vector<float>& Q, std::vector<std::vector<float>>& approx_tensors) {
    //     // Concatenate the approx_tensors into a single vector
    //     std::vector<float> doc_emb_approx;
    //     for (const auto& approx_tensor : approx_tensors) {
    //         doc_emb_approx.insert(doc_emb_approx.end(), approx_tensor.begin(), approx_tensor.end());
    //     }

    //     // Normalize the document embeddings
    //     std::vector<float> D;
    //     for (size_t i = 0; i < doc_emb_approx.size(); i += embdding_dim) {
    //         float norm = 0.0f;
    //         for (size_t j = 0; j < embdding_dim; j++) {
    //             float value = doc_emb_approx[i + j];
    //             norm += value * value;
    //         }
    //         norm = std::sqrt(norm);
    //         for (size_t j = 0; j < embdding_dim; j++) {
    //             D.push_back(doc_emb_approx[i + j] / norm);
    //         }
    //     }

    //     // Compute the scores
    //     std::vector<float> scores(DOCUMENT_BATCH_SIZE, 0.0f);
    //     for (size_t i = 0; i < Q.size(); i += query_maxlen * embdding_dim) {
    //         for (size_t j = 0; j < DOCUMENT_BATCH_SIZE; j++) {
    //             float max_score = 0.0f;
    //             for (size_t k = 0; k < doc_maxlen; k++) {
    //                 float score = 0.0f;
    //                 for (size_t l = 0; l < embdding_dim; l++) {
    //                     float q_value = Q[i + k * embdding_dim + l];
    //                     float d_value = D[j * doc_maxlen * embdding_dim + k * embdding_dim + l];
    //                     score += q_value * d_value;
    //                 }
    //                 if (score > max_score) {
    //                     max_score = score;
    //                 }
    //             }
    //             scores[j] += max_score;
    //         }
    //     }

    //     std::cout << "score.cc::scores vector size: " << scores.size() << std::endl;
    //     return scores;
    // }

}
   

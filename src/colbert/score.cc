#include "score.h"
#include<iostream>
#include<chrono>

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

    torch::Tensor Score::compute_scores(torch::Tensor Q, torch::Tensor D){
        
        #ifdef PRFILE_CQ
            auto begin = std::chrono::system_clock::now();
        #endif
        
        auto scores = std::get<0>(Q.matmul(D.permute({0, 2, 1})).max(2)).sum(1);   
        
        #ifdef PRFILE_CQ
        auto end = std::chrono::system_clock::now();
        std::cout<<"scoring time in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
        #endif

        return scores;
    }

}
   


#include "src/cq/execute_cq.h"
#include "src/utils.h"
#include "src/config.h"

using namespace std;
using namespace lh;

#include <torch/torch.h>
#include <iostream>

int main()
{   
    string input_string_1 = {"acura integra timing belt replacement cost"};
   
    std::vector<std::string> is;
    is.push_back(input_string_1);
   
    ExecuteCQ cq;
    cq.execute(is);



    
    // Decoder decoder;
    // decoder.decode();

   
    // string input_string = {"acura integra timing belt replacement cost"};
    // std::vector<std::string> is;
    // is.push_back(input_string);
    // QueryEncoder<float> qe;
    // torch::Tensor Q = qe.encode(is);


    // auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
    // std::vector<float> query = get_vec_from_file("/home/deept/cq-cpp/test/query_normalize_out.txt");
    // std::vector<float> doc0_vec = get_vec_from_file("/home/deept/cq-cpp/test/doc0_D_final.txt");
    // std::vector<float> doc1_vec = get_vec_from_file("/home/deept/cq-cpp/test/doc1_D_final.txt");

    // //  auto Q = torch::from_blob(query.data(),
    // //                                {1, int(query.size())}, options).view({1, 32, 128});
    
    //  auto D0 = torch::from_blob(doc0_vec.data(),
    //                                {1, int(doc0_vec.size())}, options).view({1, 56, 128});

    //  auto D1 = torch::from_blob(doc1_vec.data(),
    //                               {1, int(doc1_vec.size())}, options).view({1, 37, 128});

    // torch::Tensor new_tensor0 = torch::zeros({1, 512, 128});

    // new_tensor0.slice(1, 0, 56) = D0;

   

    // torch::Tensor new_tensor1 = torch::zeros({1, 512, 128});

    // new_tensor1.slice(1, 0, 37) = D1;

    // torch::Tensor D = torch::cat({new_tensor0, new_tensor1}, 0);

    // // cout<<D.sizes()<<endl;

    // Score score;

    // torch::Tensor ans = score.compute_scores(Q, D);
    // std::cout<<ans.sizes()<<std::endl;
    // std::cout<<ans[0]<<" "<<ans[1]<<std::endl;
    

}
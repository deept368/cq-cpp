// #include "src/colbert/queryencoder.h"
// #include "src/config.h"
// #include "src/colbert/codefetcher.h"
#include "src/cq/queryprocessor.h"
// using namespace std;
using namespace lh;

// #include <torch/torch.h>
#include <iostream>
#include<unordered_map>

// int main()
// {   
//     string input_string_1 = {"acura integra timing belt replacement cost"};
   
//     std::vector<std::string> is;
//     is.push_back(input_string_1);
   
//     ExecuteCQ cq;
//     cq.execute(is);



    
//     // Decoder decoder;
//     // decoder.decode();

   
//     // string input_string = {"acura integra timing belt replacement cost"};
//     // std::vector<std::string> is;
//     // is.push_back(input_string);
//     // QueryEncoder<float> qe;
//     // torch::Tensor Q = qe.encode(is);


//     // auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
//     // std::vector<float> query = get_vec_from_file("/home/deept/cq-cpp/test/query_normalize_out.txt");
//     // std::vector<float> doc0_vec = get_vec_from_file("/home/deept/cq-cpp/test/doc0_D_final.txt");
//     // std::vector<float> doc1_vec = get_vec_from_file("/home/deept/cq-cpp/test/doc1_D_final.txt");

//     // //  auto Q = torch::from_blob(query.data(),
//     // //                                {1, int(query.size())}, options).view({1, 32, 128});
    
//     //  auto D0 = torch::from_blob(doc0_vec.data(),
//     //                                {1, int(doc0_vec.size())}, options).view({1, 56, 128});

//     //  auto D1 = torch::from_blob(doc1_vec.data(),
//     //                               {1, int(doc1_vec.size())}, options).view({1, 37, 128});
using namespace std;

// int main()
// {


   
//     string input_string = {"acura integra timing belt replacement cost"};
//     QueryEncoder<float> query;
//     query.encode(input_string);


// }

// int main()
// {
//     Fetcher fetcher("../Store/output_result/result_", 256);
//     std::vector<std::string> doc_ids = {"0", "1"};
//     std::unordered_map<std::string, std::vector<std::vector<int>>> data;

//     data = fetcher.get_codes(doc_ids);
//     fetcher.print_doc_data(data);

//     return 0;
// }

int main()
{
    QueryProcessor qp("../data/msmarco-test2019-queries-qrel.tsv", "../data/retrieval-results.tsv");
    unordered_map<int, unordered_map<string, vector<vector<int>>>> code_map = qp.getCodes();

    // print data
    for (auto p : code_map)
    {
        cout << "Query id: " << p.first << ", Query: " << qp.getQuery(p.first) << "No. of documents: " << p.second.size() << endl;
        (qp.get_code_fetcher())->print_doc_data(p.second);
        cout << endl;
    }

}
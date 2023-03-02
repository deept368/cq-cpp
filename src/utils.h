#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "config.h"

using namespace std;

#ifndef UTILS_H
#define UTILS_H

namespace lh {
    
    template<class T>
    inline std::vector<T> convert_to_vector(T* input, std::size_t size){
        std::vector<T> ans;
        for(std::size_t idx=0; idx<size; idx++){
            ans.push_back((T)input[idx]);
        }
        return ans;
    }

    template<class T>
    inline std::vector<T> linearize_vector_of_vectors(std::vector<std::vector<T>> input) {
        std::vector<T> ans;
        for (const auto& v : input) {
            for (auto d : v) {
                ans.push_back(d);
            }
        }
        return ans;
    }
    
    // template<class T>
    // auto convert_linear_vec_to_2d_tensor(std::vector<T> input, std::int64_t rows, std::int64_t cols){
    //     auto options = torch::TensorOptions().dtype(TORCH_DTYPE);
    //     auto data = torch::from_blob(input.data(), {1, int(input.size())}, options);
    //     auto ans = data.view({rows, cols});
    //    // checked and this works
    //     for(int i=0; i<5; i++){
    //         std::cout<<"index : "<<i<<std::endl;
    //         std::cout<<input[0*768+i]<<" ";
    //         std::cout<<ans[0][i]<<std::endl;
    //     }
    //     return ans;
    // }

    inline std::vector<float> get_vec_from_file(std::string file_path){
        std::vector<float> ans;
        std::ifstream in;
        in.open(file_path);
        float element;
        if (in.is_open()) {
            while (in >> element) {
                ans.push_back(element);
            }
        }
        in.close();
        return ans;
    }

    inline std::vector<int> get_int_vec_from_file(std::string file_path){
        std::vector<int> ans;
        std::ifstream in;
        in.open(file_path);
        float element;
        if (in.is_open()) {
            while (in >> element) {
                ans.push_back(element);
            }
        }
        in.close();
        return ans;
    }

    inline std::vector<std::vector<int>> get_vec_of_vecs_from_file(std::string file_path){
        ifstream file(file_path);

        vector<vector<int>> vec;

        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            int num;

            vector<int> innerVec;
            while (ss >> num) {
                innerVec.push_back(num);
            }

            vec.push_back(innerVec);
        }
        return vec;
    }
   


}

#endif //UTILS_H



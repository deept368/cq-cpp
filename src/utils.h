#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "config.h"
#include <cstdint>
#include <arpa/inet.h>
#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>
#include <bitset>
#include <sstream>
#include <boost/filesystem.hpp>

using namespace std;

#ifndef UTILS_H
#define UTILS_H

namespace lh {
    
    template<class T>
    inline std::vector<T>* convert_to_vector(T* input, std::size_t size){
        std::vector<T>* ans = new std::vector<T>();
        for(std::size_t idx=0; idx<size; idx++){
            ans->push_back((T)input[idx]);
        }
        return ans;
    }

    template<class T>
    inline std::vector<T>* linearize_vector_of_vectors(std::vector<std::vector<T>*>* input) {
       std::vector<T>* ans = new std::vector<T>();
        for (const auto& v : *input) {
            for (auto& d : *v) {
                ans->push_back(d);
            }
        }
        return ans; 
    }


    inline std::vector<float>* get_vec_from_file(std::string file_path){
        std::vector<float>* ans;
        std::ifstream in;
        in.open(file_path);
        float element;
        if (in.is_open()) {
            while (in >> element) {
                ans->push_back(element);
            }
        }
        in.close();
        return ans;
    }

    inline std::vector<int>* get_int_vec_from_file(std::string file_path){
        std::vector<int>* ans;
        std::ifstream in;
        in.open(file_path);
        float element;
        if (in.is_open()) {
            while (in >> element) {
                ans->push_back(element);
            }
        }
        in.close();
        return ans;
    }

    inline std::vector<std::vector<int>>* get_vec_of_vecs_from_file(std::string file_path){
        ifstream file(file_path);

        vector<vector<int>>* vec;

        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            int num;

            vector<int>* innerVec;
            while (ss >> num) {
                innerVec->push_back(num);
            }
            vec->push_back(*innerVec);
        }
        return vec;
    }

    inline string get_hex(uint32_t num){
        bitset<32> bits(num); // convert to binary string of length 32
        stringstream ss;
        ss << setfill('0') << setw(8) << hex << uppercase << bits.to_ulong(); // convert binary string to hex string
        string hex_string = ss.str();
        return hex_string;
    }

    inline string compute_hash(string doc_id){
        boost::uuids::detail::md5 hash;
        boost::uuids::detail::md5::digest_type digest;
        hash.process_bytes(doc_id.data(), doc_id.size());
        hash.get_digest(digest);

        // Convert the first 4 bytes to a string in hexadecimal format
        string hex_hash;
        boost::algorithm::hex(digest, digest + 1, back_inserter(hex_hash));

        return hex_hash;
    }

    inline std::string format_trec_line(const int query_id, const std::string& doc_id, int rank, float score, const std::string& run_id) {
        std::stringstream ss;
        ss << query_id << " Q0 " << doc_id << " " << rank << " " << score << " " << run_id << std::endl;
        return ss.str();
    }



}

#endif //UTILS_H



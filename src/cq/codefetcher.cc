#include "codefetcher.h"
#include "../utils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "codefetcher.h"
#include <arpa/inet.h>
#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>
#include <bitset>
#include <sstream>
#include <boost/filesystem.hpp>

using namespace std;

namespace lh{

    CodeFetcher::CodeFetcher(string filename, uint32_t number_of_files) : base_filename(filename), number_of_files(number_of_files){
        load_metadata();
        initialize_file_ptrs();
        initialize_key_offset_store();

        cout << "Total number of documents: " << key_offset_store.size() << endl;
    }

    CodeFetcher::~CodeFetcher(){
        // Release any resources
        // clean up file pointers
        for (auto ptr : file_ptrs) {
            (*ptr).close();
            delete ptr;
        }
    }

    void CodeFetcher::load_metadata(){
        string filename = base_filename + "0";
        ifstream infile(filename, ios::binary);

        if (!infile)
        {
            cerr << "Failed to open file " << filename << endl;
        }

        // Read metadata
        char metadata[128];

        infile.read(metadata, 128);

        M = ntohl(*reinterpret_cast<uint32_t *>(&metadata[0]));
        K = ntohl(*reinterpret_cast<uint32_t *>(&metadata[4]));
        key_bytes = ntohl(*reinterpret_cast<uint32_t *>(&metadata[12]));
        offset_bytes = ntohl(*reinterpret_cast<uint32_t *>(&metadata[16]));
        token_bytes = ntohl(*reinterpret_cast<uint32_t *>(&metadata[20]));

        infile.close();
    }

    void CodeFetcher::initialize_file_ptrs(){
        for (int i = 0; i < number_of_files; i++) {
            file_ptrs.push_back(new ifstream(base_filename + to_string(i), ios::binary));
        }
    }

    bool CodeFetcher::read_file(int file_num){
        int file_idx = file_num % 256;
        ifstream& infile = *file_ptrs[file_idx];

        if (!infile)
        {
            cerr << "Failed to open file " << base_filename + to_string(file_num) << endl;
            return false;
        }

        // Read metadata
        uint32_t M, K, num_docs, key_bytes, offset_bytes, token_bytes, remaining;
        char metadata[128]; // initialize to null bytes

        infile.read(metadata, 128);

        num_docs = ntohl(*reinterpret_cast<uint32_t *>(&metadata[8]));

        // Read document key and offset information
        for (int i = 0; i < num_docs; i++){
            uint32_t doc_key, doc_offset;
            char data[8]; // initialize to null bytes

            infile.read(data, 8);
            doc_key = ntohl(*reinterpret_cast<uint32_t *>(&data[0]));
            doc_offset = ntohl(*reinterpret_cast<uint32_t *>(&data[4]));

            key_offset_store[get_hex(doc_key)] = doc_offset;
        }
        // cout << key_offset_store.size() << endl;
        num_docs += num_docs;
        return true;
    }

    void CodeFetcher::initialize_key_offset_store(){
        // Read from multiple files
        for (int i = 0; i < number_of_files; i++){
            bool success = read_file(i);
            if (!success){
                cout << "Failed to read data from file " << base_filename + to_string(i) << endl;
            }
        }
    }

    unordered_map<string, vector<vector<std::size_t>>> CodeFetcher::get_codes(vector<string> document_ids){
        // read document data
        unordered_map<string, vector<vector<std::size_t>>> doc_data_map;
        for (auto doc_id : document_ids){
            std::size_t i = 0;
            string md5_hex = compute_hash(doc_id);
            std::size_t file_idx = stoi(doc_id) % 256;
            std::size_t bit_offset = key_offset_store[md5_hex];

            ifstream &file = *file_ptrs[file_idx];
            file.seekg(bit_offset / 8, ios::beg);
            std::size_t token_id = -1;
            vector<vector<std::size_t>> doc_data;

            while (token_id != 102){
                char buffer[18];
                file.read(buffer, 18);

                token_id = ntohs(*(reinterpret_cast<uint16_t *>(buffer)));
                vector<std::size_t> token_data;
                token_data.push_back(token_id);
                for (int i = 2; i < 18; i++)
                {
                    token_data.push_back((unsigned char)buffer[i]);
                }

                doc_data.push_back(token_data);
                i++;
            }
            doc_data_map[doc_id] = doc_data;
        }

        return doc_data_map;
    }
}

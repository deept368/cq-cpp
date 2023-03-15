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
#include "../config.h"

using namespace std;

namespace lh{

    CodeFetcher::CodeFetcher(){
        total_docs = 0;
        base_filename = BASE_STORE_FILE;
        number_of_files = STORE_SIZE;
        
        for (int i = 0; i < number_of_files; i++) {
            file_ptrs.push_back(new ifstream(base_filename + to_string(i), ios::binary));
        }

        // Read from multiple files
        for (int i = 0; i < number_of_files; i++){
        
            ifstream& infile = *file_ptrs[i];

            if (!infile){
                cerr << "Failed to open file " << base_filename + to_string(i) << endl;
            } else {
                int num_docs;
                char metadata[128];
                infile.read(metadata, 128);

                num_docs = 34538;
                if (i <= 94){
                    num_docs++;
                }
                // Read document key and offset information
                for (int j = 0; j < num_docs; j++){
                    int doc_key, doc_offset;
                    char data[8]; // initialize to null bytes

                    infile.read(data, 8);
                    string doc_id = to_string(256*(j) + i);
                    doc_offset = ntohl(*reinterpret_cast<uint32_t *>(&data[4]));
                    key_offset_store.insert(make_pair(doc_id, doc_offset));
                
                }
                total_docs += num_docs;
            }
        }
        cout << "Total documents: " << total_docs << endl;
    }

    CodeFetcher::~CodeFetcher(){
        // Release any resources
        // clean up file pointers
        for (auto ptr : file_ptrs) {
            (*ptr).close();
            delete ptr;
        }
    }


    // unordered_map<string, vector<vector<int>>> CodeFetcher::get_codes(vector<string> document_ids){
    //     // read document data
    //     unordered_map<string, vector<vector<int>>> doc_data_map;

    //     for (auto doc_id : document_ids){
    //         string md5_hex = compute_hash(doc_id);
    //         int file_idx = stoi(doc_id) % 256;
    //         int bit_offset = key_offset_store.at(md5_hex);
    //         ifstream &file = *file_ptrs[file_idx];
    //         file.seekg(bit_offset / 8, ios::beg);
    //         int token_id = -1;
    //         vector<vector<int>> doc_data;

    //         while (token_id != 102){
    //             char buffer[18];
    //             file.read(buffer, 18);

    //             token_id = ntohs(*(reinterpret_cast<int* >(buffer)));
    //             vector<int> token_data;
    //             token_data.push_back(token_id);
    //             for (int i = 2; i < 18; i++){
    //                 token_data.push_back((unsigned char)buffer[i]);
    //             }
    //             doc_data.push_back(token_data);
    //         }
    //         doc_data_map.insert(make_pair(doc_id, doc_data));
    //     }
    //     return doc_data_map;
    // }

    unordered_map<string, vector<vector<int>>> CodeFetcher::get_codes(vector<string> document_ids){
        // read document data
        unordered_map<string, vector<vector<int>>> doc_data_map;

        for (auto doc_id : document_ids){
            int file_idx = stoi(doc_id) % 256;
            int bit_offset = key_offset_store.at(doc_id);
            ifstream &file = *file_ptrs[file_idx];
            file.seekg(bit_offset / 8, ios::beg);
            int token_id = -1;
            vector<vector<int>> doc_data;

            while (token_id != 102){
                char buffer[18];
                file.read(buffer, 18);

                token_id = ntohs(*(reinterpret_cast<int* >(buffer)));
                vector<int> token_data;
                token_data.push_back(token_id);
                for (int i = 2; i < 18; i++){
                    token_data.push_back((unsigned char)buffer[i]);
                }
                doc_data.push_back(token_data);
            }
            doc_data_map.insert(make_pair(doc_id, doc_data));
        }
        return doc_data_map;
    }
}

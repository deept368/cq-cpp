#ifndef CODEFETCHER_H
#define CODEFETCHER_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

namespace lh{

    class CodeFetcher
    {
        private:
            string base_filename;
            uint32_t M, K, num_docs = 0, key_bytes, offset_bytes, token_bytes;
            uint32_t number_of_files;
            vector<ifstream *> file_ptrs;
            unordered_map<string, int> key_offset_store;

        public:
            // Constructor
            CodeFetcher(string filename, uint32_t number_of_files);
            ~CodeFetcher();
            void load_metadata();
            void initialize_file_ptrs();
            bool read_file(int file_num);
            void initialize_key_offset_store();
            string get_hex(uint32_t num);
            string compute_hash(string doc_id);
            unordered_map<string, vector<vector<int>>> get_codes(vector<string> document_ids);
            void print_doc_data(unordered_map<string, vector<vector<int>>> doc_data_map);
    };
}

#endif // CODEFETCHER_H
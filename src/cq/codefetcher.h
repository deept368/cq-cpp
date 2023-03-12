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
            size_t M, K, num_docs = 0, key_bytes, offset_bytes, token_bytes;
            size_t number_of_files;
            vector<ifstream *> file_ptrs;
            unordered_map<string, size_t> key_offset_store;

            void load_metadata();
            void initialize_file_ptrs();
            bool read_file(int file_num);
            void initialize_key_offset_store();

            

        public:
            // Constructor
            CodeFetcher(string filename, size_t number_of_files);
            ~CodeFetcher();
            unordered_map<string, vector<vector<std::size_t>>> get_codes(vector<string> document_ids);
    };
}

#endif // CODEFETCHER_H
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
            int number_of_files;
            vector<ifstream *>* file_ptrs;
            unordered_map<string, int>* key_offset_store;
            void get_codes_partial(vector<string>* document_ids, int start_idx, int step_size, unordered_map<string, vector<vector<int>*>*>* doc_data_partial);
            int total_docs;
    
        public:
            // Constructor
            CodeFetcher();
            ~CodeFetcher();
            unordered_map<string, vector<vector<int>*>*>* get_codes(vector<string>* document_ids);
    };
}

#endif // CODEFETCHER_H
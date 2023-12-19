#ifndef CODEFETCHER_H
#define CODEFETCHER_H


#include "../utils.h"
#include <string>
#include "codefetcher.h"
#include <arpa/inet.h>
#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>
#include <bitset>
#include <sstream>
#include <boost/filesystem.hpp>
#include "../config.h"
#include<future>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include "../config.h"

using namespace std;

namespace lh{

    class CodeFetcher
    {
        private:
            string base_filename;
            int number_of_files;
            vector<ifstream *>* file_ptrs;
            unordered_map<string, int>* key_offset_store;
            unordered_map<string, vector< pair<uint16_t, vector<uint8_t>*> >*>* codes_store;
            int total_docs;
    
        public:
            // Constructor
            CodeFetcher();
            ~CodeFetcher();
            unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>* get_codes(vector<string>* document_ids);
    };
}

#endif // CODEFETCHER_H
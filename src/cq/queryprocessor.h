#ifndef QUERYPROCESSOR_H
#define QUERYPROCESSOR_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "codefetcher.h"

using namespace std;

namespace lh{
    
    class QueryProcessor
    {
    private:
        unordered_map<std::size_t, string> queryMapping;
        unordered_map<std::size_t, vector<string>> queryResults;
        static CodeFetcher* code_fetcher;

        void readQueryMapping(string queryFile);
        void readQueryResults(string resultFile);
        vector<string> getQueryResults(std::size_t queryId);
        static CodeFetcher* get_code_fetcher();
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> getCodes();
        string getQuery(std::size_t queryId);
        void print_doc_data(unordered_map<string, vector<vector<std::size_t>>> doc_data_map);
       
    };
}

#endif // QUERYPROCESSOR_H
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
        unordered_map<int, string> queryMapping;
        unordered_map<int, vector<string>> queryResults;
        CodeFetcher* code_fetcher;

        void readQueryMapping(string queryFile);
        void readQueryResults(string resultFile);
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<int, unordered_map<string, vector<vector<int>>>> getCodes();
        string getQuery(int queryId);
        // void print_doc_data(unordered_map<string, vector<vector<int>>> doc_data_map);
       
    };
}

#endif // QUERYPROCESSOR_H
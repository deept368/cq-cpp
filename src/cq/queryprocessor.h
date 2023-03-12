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
        unordered_map<string, int> queryIdMapping;
        unordered_map<int, vector<string>> queryResults;
        static CodeFetcher* code_fetcher;

        void readQueryMapping(string queryFile);
        void readQueryResults(string resultFile);
        vector<string> getQueryResults(int queryId);
        static CodeFetcher* get_code_fetcher();
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<int, unordered_map<string, vector<vector<int>>>> getCodes();
        string getQuery(int queryId);
        int getQueryId(string queryId);
        void print_doc_data(unordered_map<string, vector<vector<int>>> doc_data_map);
       
    };
}

#endif // QUERYPROCESSOR_H
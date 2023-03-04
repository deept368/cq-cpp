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
        static CodeFetcher* code_fetcher;
    public:
        QueryProcessor(string queryFile, string resultFile);
        ~QueryProcessor();
        void readQueryMapping(string queryFile);
        void readQueryResults(string resultFile);
        string getQuery(int queryId);
        vector<string> getQueryResults(int queryId);
        unordered_map<int, unordered_map<string, vector<vector<int>>>> getCodes();
        static CodeFetcher* get_code_fetcher();
    };
}

#endif // QUERYPROCESSOR_H
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
    
    class QueryProcessor{
    private:
        unordered_map<int, vector<string>*>* queryResults;
        CodeFetcher* code_fetcher;
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* getCodes();
    };
}

#endif // QUERYPROCESSOR_H
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
        unordered_map<int, vector<string>*>* queryResults;
        unordered_map<int, unordered_map<string, float>> originalScores;
        CodeFetcher* code_fetcher;
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>*>* getCodes(int offset);
        const unordered_map<int, unordered_map<string, float>>& getOriginalScores();
       
    };
}

#endif // QUERYPROCESSOR_H
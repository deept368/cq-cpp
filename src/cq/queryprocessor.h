#ifndef QUERYPROCESSOR_H
#define QUERYPROCESSOR_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>


using namespace std;

namespace lh{
    
    class QueryProcessor
    {
    private:
       
    
    public:
        QueryProcessor();
        ~QueryProcessor();
        unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* getCodes();
       
    };
}

#endif // QUERYPROCESSOR_H
#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"
#include "../utils.h"

using namespace std;

namespace lh
{

    QueryProcessor::QueryProcessor(){
    }

    QueryProcessor::~QueryProcessor(){
       
    }   
    
   
    string QueryProcessor::getQuery(int queryId){
       return "acura integra timing belt replacement cost";
    }

    

    unordered_map<int, unordered_map<string, vector<vector<int>>>> QueryProcessor::getCodes(){
        unordered_map<int, unordered_map<string, vector<vector<int>>>> result;
         unordered_map<string, vector<vector<int>>> internal_map;

         vector<vector<int>> doc0_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc0_values.txt");
         vector<vector<int>> doc1_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc1_values.txt");

         internal_map.insert(make_pair("doc0", doc0_vec));
         internal_map.insert(make_pair("doc1", doc1_vec));

         result.insert(make_pair(12345, internal_map));

         return result;
    }

    
    
};

#include "codefetcher.h"
#include "../utils.h"
#include<iostream>

using namespace std;

namespace lh{

    
    CodeFetcher::CodeFetcher(){  
    }

  
    CodeFetcher::~CodeFetcher(){    
    }    

    map<string, map<string, vector<vector<int>>>> CodeFetcher::fetch_codes(){
        map<string, map<string, vector<vector<int>>>> result;
        map<string, vector<vector<int>>> internal_map;
       
        vector<vector<int>> doc0_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc0_values.txt");
        vector<vector<int>> doc1_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc1_values.txt");
        
        internal_map.insert(make_pair("doc0", doc0_vec));
        internal_map.insert(make_pair("doc1", doc1_vec));

        result.insert(make_pair("acura integra timing belt replacement cost", internal_map));

        return result;
    }

}
   

#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"

using namespace std;

namespace lh
{

    QueryProcessor::QueryProcessor(){
        code_fetcher = new CodeFetcher();
        queryResults = new unordered_map<int, vector<string>*>();
        ifstream file(RESULTS_FILE);
        if (!file) {
            cerr << "Error opening result file: " << RESULTS_FILE << endl;
            return;
        }

        string line;
        while (getline(file, line)){
            istringstream ss(line);
            string queryId, temp1, docId, temp2, temp3, temp4;
            ss >> queryId >> temp1 >> docId >> temp2 >> temp3 >> temp4;
            
            if (queryResults->find(stoi(queryId)) == queryResults->end()){
                queryResults->insert({stoi(queryId), new std::vector<std::string>()});
            } 
            (*queryResults)[stoi(queryId)]->push_back(docId);            
        }

        file.close();
        cout << "Loading query results completed." << endl;
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
        for (auto& kv : *queryResults) {
            delete kv.second;
        }
        delete queryResults;
    }   
    

    unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* QueryProcessor::getCodes(int offset){
        unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* code_map = new unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>();
       

        int batch_size = BATCH_SIZE;

        auto it = queryResults->begin();
        std::advance(it, offset);
        cout<<"offset is "<<offset<<endl;

       
        for (int i = 0; i < batch_size && it != queryResults->end(); ++i, ++it) {
            int query_id = it->first;

            cout << "Now processing for query: " << query_id << " " << i << endl;

            unordered_map<string, vector<vector<int>*>*>* codes = code_fetcher->get_codes(it->second);
            code_map->insert(make_pair(query_id, codes));

            cout << "CodeMap size is " << code_map->size() << endl;
        }
        cout<<"returning codes"<<endl;
        return code_map;
    }

};

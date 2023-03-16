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
        unordered_map<int, vector<string>*>* queryResults =  new unordered_map<int, vector<string>*>();
        ifstream file(RESULTS_FILE);
        if (!file) {
            cerr << "Error opening result file: " << RESULTS_FILE << endl;
            return;
        }

        cout<<"here"<<endl;
        string line;
        while (getline(file, line)){
            istringstream ss(line);
            string queryId, temp1, docId, temp2, temp3, temp4;
            ss >> queryId >> temp1 >> docId >> temp2 >> temp3 >> temp4;
            
            if (queryResults->find(stoi(queryId)) != queryResults->end()){
                ((*queryResults)[stoi(queryId)])->push_back(docId);
            } else{
                vector<string>* docId_vec = new vector<string>();
                docId_vec->push_back(docId);
                queryResults->insert(make_pair(stoi(queryId), docId_vec));
            }           
        }

        cout << *(*queryResults)[527433] <<endl;

        file.close();
        cout << "Loading query results completed." << endl;
        cout << (*queryResults)[527433] <<endl;
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
        
        for (auto& kv : *queryResults) {
            delete kv.second;
        }
        delete queryResults;
    }   
    

    unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* QueryProcessor::getCodes(){
        unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>* code_map = new unordered_map<int, unordered_map<string, vector<vector<int>*>*>*>();
        cout<<"reached start"<<endl;
        cout << (*queryResults)[527433] <<endl;
        cout<<"reached here"<<endl;
        int c=1;
        for (auto &queryDocMap : *queryResults) {
            
            int query_id = queryDocMap.first;
            cout<<"qid "<<query_id<<endl;
            cout << "Now processing for query: " << query_id << " " << c++ << endl;

            unordered_map<string, vector<vector<int>*>*>* codes = code_fetcher->get_codes(queryDocMap.second);
            code_map->insert(make_pair(query_id, codes));

            cout << "CodeMap size is " << code_map->size() << endl;
        }

        cout<<"returning codes"<<endl;
        return code_map;
    }

};

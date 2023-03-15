#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"

using namespace std;

namespace lh
{

    QueryProcessor::QueryProcessor(){
        readQueryResults(RESULTS_FILE);
        code_fetcher = new CodeFetcher();
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
    }   
    

    void QueryProcessor::readQueryResults(string resultFile){
        ifstream file(resultFile);
        if (!file) {
            cerr << "Error opening result file: " << resultFile << endl;
            return;
        }

        string line;
        while (getline(file, line)){
            istringstream ss(line);
            string queryId, temp1, docId, temp2, temp3, temp4;
            ss >> queryId >> temp1 >> docId >> temp2 >> temp3 >> temp4;
            
            if (queryResults.find(stoi(queryId)) != queryResults.end()){
                queryResults[stoi(queryId)].push_back(docId);
            }
            else{
                vector<string> docId_vec;
                docId_vec.push_back(docId);
                queryResults.insert(make_pair(stoi(queryId), docId_vec));
            }
            
        }
        file.close();
        cout << "Loading query results completed." << endl;
    }

    unordered_map<int, unordered_map<string, vector<vector<int>>>> QueryProcessor::getCodes(){
        unordered_map<int, unordered_map<string, vector<vector<int>>>> code_map;

        int count = 0;
        int z =0;
        for (const auto &queryDocMap : queryResults) {
            count++;
            // if (queryDocMap.first == 1099803) {
            //     continue;
            // }
            z++;
            cout<<"query id "<<queryDocMap.first<<endl;

            if (z < 615){
                continue;
            }
            unordered_map<string, vector<vector<int>>> codes =code_fetcher->get_codes(queryDocMap.second);
            cout << "Z is " << z <<endl;
            
            code_map.insert(make_pair(queryDocMap.first, codes));
        }

        cout<<"returning codes"<<endl;
        return code_map;
    }

};

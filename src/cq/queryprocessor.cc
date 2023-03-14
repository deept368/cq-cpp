#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"

using namespace std;

namespace lh
{

    QueryProcessor::QueryProcessor(){
        readQueryMapping(QUERY_FILE);
        readQueryResults(RESULTS_FILE);
        code_fetcher = new CodeFetcher();
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
    }   
    
   

    void QueryProcessor::readQueryMapping(string queryFile){
        ifstream file(queryFile);
        if (!file) {
            cerr << "Error opening query file: " << queryFile << endl;
            return;
        }

        string line;
        while (getline(file, line)) {
            istringstream ss(line);
            string queryId, query;
            ss >> queryId;
            
            // Extract substring starting from first non-tab character
            int start = line.find_first_not_of('\t', queryId.length());
            query = line.substr(start);
            
           
            queryMapping.insert(make_pair(stoi(queryId), query));
        }
        file.close();
        cout << "Loading query mapping completed." << endl;
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

    string QueryProcessor::getQuery(int queryId){
        if (queryMapping.find(queryId) == queryMapping.end()) {
            cerr << "Query ID " << queryId << " not found." << endl;
            return "";
        }
        return queryMapping[queryId];
    }

    unordered_map<int, unordered_map<string, vector<vector<int>>>> QueryProcessor::getCodes(){
        unordered_map<int, unordered_map<string, vector<vector<int>>>> code_map;
        for (const auto &queryDocMap : queryResults) {
            unordered_map<string, vector<vector<int>>> codes =code_fetcher->get_codes(queryDocMap.second);
            code_map.insert(make_pair(queryDocMap.first, codes));
        }

        return code_map;
    }

    // void QueryProcessor::print_doc_data(unordered_map<string, vector<vector<int>>> doc_data_map){
    //     // print document data
    //     for (auto p : doc_data_map){
    //             cout << "Document " << p.first << ": (" << p.second.size() << " tokens)" << endl;
    //         for (auto d : p.second){
    //             for (int i = 0; i < d.size(); i++)
    //                 cout << d.at(i) << ",";
    //             cout << endl;
    //         }
    //         cout << endl;
    //     }
    // }
};

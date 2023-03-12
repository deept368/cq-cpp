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
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
    }   
    
    CodeFetcher* QueryProcessor::code_fetcher = new CodeFetcher(BASE_STORE_FILE, STORE_SIZE);

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
            size_t start = line.find_first_not_of('\t', queryId.length());
            query = line.substr(start);
            
            queryMapping[stoi(queryId)] = query;
            queryIdMapping[query] = stoi(queryId);
        }
        file.close();
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
            queryResults[stoi(queryId)].push_back(docId);
        }
        file.close();
    }

    string QueryProcessor::getQuery(std::size_t queryId){
        if (queryMapping.find(queryId) == queryMapping.end()) {
            cerr << "Query ID " << queryId << " not found." << endl;
            return "";
        }
        return queryMapping[queryId];
    }

    size_t QueryProcessor::getQueryId(string query){
        if (queryIdMapping.find(query) == queryIdMapping.end()) {
            cerr << "Query ID " << query << " not found." << endl;
            return -1;
        }
        return queryIdMapping[query];
    }

    vector<string> QueryProcessor::getQueryResults(std::size_t queryId){
        if (queryResults.find(queryId) == queryResults.end()){
            cerr << "Query ID " << queryId << " not found." << endl;
            return vector<string>();
        }
        return queryResults[queryId];
    }

    unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> QueryProcessor::getCodes(){
        unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> code_map;
        for (const auto &queryDocMap : queryResults) {
            code_map[queryDocMap.first] = code_fetcher->get_codes(queryDocMap.second);
        }

        return code_map;
    }

    CodeFetcher* QueryProcessor::get_code_fetcher(){
        return code_fetcher;
    }

    void QueryProcessor::print_doc_data(unordered_map<string, vector<vector<std::size_t>>> doc_data_map){
        // print document data
        for (auto p : doc_data_map){
                cout << "Document " << p.first << ": (" << p.second.size() << " tokens)" << endl;
            for (auto d : p.second){
                for (int i = 0; i < d.size(); i++)
                    cout << d.at(i) << ",";
                cout << endl;
            }
            cout << endl;
        }
    }
};

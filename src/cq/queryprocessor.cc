#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

namespace lh
{

    QueryProcessor::QueryProcessor(string queryFile, string resultFile)
    {
        readQueryMapping(queryFile);
        readQueryResults(resultFile);
    }

    QueryProcessor::~QueryProcessor(){
        delete code_fetcher;
    }   
    
    CodeFetcher* QueryProcessor::code_fetcher = new CodeFetcher("../data/output_result/result_", 256);

    void QueryProcessor::readQueryMapping(string queryFile)
    {
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
        }
        file.close();
    }

    void QueryProcessor::readQueryResults(string resultFile)
    {
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

    string QueryProcessor::getQuery(int queryId)
    {
        if (queryMapping.find(queryId) == queryMapping.end()) {
            cerr << "Query ID " << queryId << " not found." << endl;
            return "";
        }
        return queryMapping[queryId];
    }

    vector<string> QueryProcessor::getQueryResults(int queryId)
    {
        if (queryResults.find(queryId) == queryResults.end()) {
            cerr << "Query ID " << queryId << " not found." << endl;
            return vector<string>();
        }
        return queryResults[queryId];
    }

    unordered_map<int, unordered_map<string, vector<vector<int>>>> QueryProcessor::getCodes()
    {
        unordered_map<int, unordered_map<string, vector<vector<int>>>> code_map;
        for (const auto &queryDocMap : queryResults) {
            code_map[queryDocMap.first] = code_fetcher->get_codes(queryDocMap.second);
        }

        return code_map;
    }

    CodeFetcher* QueryProcessor::get_code_fetcher()
    {
        return code_fetcher;
    }
};

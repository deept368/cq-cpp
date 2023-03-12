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
      
        return "acura integra timing belt replacement cost";
    }

    size_t QueryProcessor::getQueryId(string query){
       
        return 12345;
    }

    vector<string> QueryProcessor::getQueryResults(std::size_t queryId){
        if (queryResults.find(queryId) == queryResults.end()){
            cerr << "Query ID " << queryId << " not found." << endl;
            return vector<string>();
        }
        return queryResults[queryId];
    }

    unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> QueryProcessor::getCodes(){
        unordered_map<std::size_t, unordered_map<string, vector<vector<std::size_t>>>> a;
        unordered_map<string, vector<vector<std::size_t>>> b;

         vector<vector<size_t>> doc0_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc0_values.txt");
         vector<vector<size_t>> doc1_vec = get_vec_of_vecs_from_file("/home/deept/cq-cpp/test/doc1_values.txt");

         b.insert(make_pair("doc0", doc0_vec));
         b.insert(make_pair("doc1", doc1_vec));
        a.insert(make_pair(12345, b));
        return a;
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

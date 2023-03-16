#include "querymapping.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"

using namespace std;

namespace lh
{

    QueryMapping::QueryMapping(){
        queryMapping = new unordered_map<int, string>;
        ifstream file(QUERY_FILE);
        if (!file) {
            cerr << "Error opening query file: " << QUERY_FILE << endl;
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
            
           
            queryMapping->insert(make_pair(stoi(queryId), query));
        }
        file.close();
        cout << "Loading query mapping completed." << endl;
    }

    QueryMapping::~QueryMapping(){
        delete queryMapping;
    }   
    
    string QueryMapping::getQuery(int queryId){
        if (queryMapping->find(queryId) == queryMapping->end()) {
            cerr << "Query ID " << queryId << " not found." << endl;
            return "";
        }
        return (*queryMapping)[queryId];
    }
};

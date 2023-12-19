#include "queryprocessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../config.h"

using namespace std;

namespace lh
{

    // Constructor
    QueryProcessor::QueryProcessor()
    {
        // Initialize CodeFetcher to retrieve document codes
        code_fetcher = new CodeFetcher();

        // Initialize data structures to store query results and original scores
        queryResults = new unordered_map<int, vector<string> *>();

        // Read query results from a file
        ifstream file(RESULTS_FILE);
        if (!file)
        {
            cerr << "Error opening result file: " << RESULTS_FILE << endl;
            return;
        }

        string line;
        // Parse each line of the result file
        while (getline(file, line))
        {
            istringstream ss(line);
            string queryId, temp1, docId, temp2, original_score, temp4;
            ss >> queryId >> temp1 >> docId >> temp2 >> original_score >> temp4;

            // Store document IDs for each query
            if (queryResults->find(stoi(queryId)) == queryResults->end())
            {
                queryResults->insert({stoi(queryId), new std::vector<std::string>()});
            }
            (*queryResults)[stoi(queryId)]->push_back(docId);

            if (AVERAGE_SCORE)
            {
                // Convert score from string representation of int (float with precision 4) to a float
                originalScores[stoi(queryId)][docId] = stof(original_score.substr(0, original_score.length() - 4) + '.' + original_score.substr(original_score.length() - 4));
            }
        }

        file.close();
        cout << "Loading query results completed." << endl;
    }

    // Destructor
    QueryProcessor::~QueryProcessor()
    {
        // Release resources: clean up CodeFetcher and query result data structures
        delete code_fetcher;
        for (auto &kv : *queryResults)
        {
            delete kv.second;
        }
        delete queryResults;
    }

    // Get document codes for a batch of queries starting from the given offset
    unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t> *>> *> *> *QueryProcessor::getCodes(int offset)
    {
        unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t> *>> *> *> *code_map = new unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t> *>> *> *>();
        int batch_size = PRE_BATCH_SIZE;

        // Iterate over the query results starting from the given offset
        auto it = queryResults->begin();
        std::advance(it, offset);
        for (int i = 0; i < batch_size && it != queryResults->end(); ++i, ++it)
        {
            int query_id = it->first;
            // Retrieve document codes for each query
            unordered_map<string, vector<pair<uint16_t, vector<uint8_t> *>> *> *codes = code_fetcher->get_codes(it->second);
            code_map->insert(make_pair(query_id, codes));
        }
        return code_map;
    }

    // Get the original scores for the queries
    const unordered_map<int, unordered_map<string, float>> &QueryProcessor::getOriginalScores()
    {
        return originalScores;
    }

} // namespace lh

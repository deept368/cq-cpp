#include "codefetcher.h"

using namespace std;

namespace lh{

    CodeFetcher::CodeFetcher(){
        total_docs = 0;
        base_filename = BASE_STORE_FILE;
        number_of_files = STORE_SIZE;
        key_offset_store = new unordered_map<string, int>();
        file_ptrs = new vector<ifstream *>();
        codes_store = new unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>();
        
        for (int i = 0; i < number_of_files; i++) {
            (*file_ptrs).push_back(new ifstream(base_filename + to_string(i), ios::binary));
        }

        // Read from multiple files
        for (int i = 0; i < number_of_files; i++){
        
            ifstream& infile = *(*file_ptrs)[i];

            if (!infile){
                cerr << "Failed to open file " << base_filename + to_string(i) << endl;
            } else {
                int num_docs;
                char metadata[128];
                infile.read(metadata, 128);

                num_docs = 34538;
                if (i <= 94){
                    num_docs++;
                }
                // Read document key and offset information
                for (int j = 0; j < num_docs; j++){
                    int doc_key, doc_offset;
                    char data[8]; // initialize to null bytes

                    infile.read(data, 8);
                    string doc_id = to_string(256*(j) + i);
                    doc_offset = ntohl(*reinterpret_cast<uint32_t *>(&data[4]));
                    key_offset_store->insert(make_pair(doc_id, doc_offset));
                
                }
                total_docs += num_docs;
                
                if (IN_MEMORY_CODES) {
                    infile.seekg((*key_offset_store)[to_string(i)] / 8, ios::beg);

                    for (int j = 0; j < num_docs; j++) {
                        string doc_id = to_string(256*(j) + i);
                        int token_id = -1;
                        vector<pair<uint16_t, vector<uint8_t>*>>* doc_data = new vector<pair<uint16_t, vector<uint8_t>*>>();
                        
                        while (token_id != 102){
                            char buffer[18];
                            infile.read(buffer, 18);

                            token_id = ntohs(*(reinterpret_cast<uint16_t*>(buffer)));
                            pair<uint16_t, vector<uint8_t>*> token_data = make_pair(token_id, new vector<uint8_t>());
                            for (int i = 2; i < 18; i++){
                                token_data.second->push_back((unsigned char)buffer[i]);
                            }
                            doc_data->push_back(token_data);
                        }
                        
                        
                        codes_store->insert(make_pair(doc_id, doc_data));
                    }

                    cout << "Processed file " << i << std::endl;
                }
            }
        }
        cout << "Total documents: " << total_docs << endl;
    }

    CodeFetcher::~CodeFetcher(){
        // Release any resources
        // clean up file pointers
        for (auto ptr : *file_ptrs) {
            (*ptr).close();
            delete ptr;
        }

        delete file_ptrs;
        delete key_offset_store;
    }

    unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>* CodeFetcher::get_codes(vector<string>* document_ids){
        // read document data
        // doc_data_map -> (doc id (as string), list of tokens in a document where for each token, we have [token id + 16 codes for each token])
        unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>* doc_data_map = new unordered_map<string, vector<pair<uint16_t, vector<uint8_t>*>>*>();

        for (auto& doc_id : *document_ids){
            vector<pair<uint16_t, vector<uint8_t>*>>* doc_data = new vector<pair<uint16_t, vector<uint8_t>*>>();
            if (IN_MEMORY_CODES) {
                doc_data = (*codes_store)[doc_id];
            }
            else {
                int file_idx = stoi(doc_id) % number_of_files;
                int bit_offset = key_offset_store->at(doc_id);
                ifstream &file = *(*file_ptrs)[file_idx];
                file.seekg(bit_offset / 8, ios::beg);
                uint16_t token_id = 0;
                
                while (token_id != 102){
                    char buffer[18];
                    file.read(buffer, 18);

                    token_id = ntohs(*(reinterpret_cast<int* >(buffer)));
                    pair<uint16_t, vector<uint8_t>*> token_data = make_pair(token_id, new vector<uint8_t>());
                    for (int i = 2; i < 18; i++){
                        token_data.second->push_back((unsigned char)buffer[i]);
                    }
                    doc_data->push_back(token_data);
                }
            }
            
            doc_data_map->insert(make_pair(doc_id, doc_data));
        }
        return doc_data_map;
    }
}

#include "src/cq/execute_cq.h"
using namespace std;
using namespace lh;

#include <torch/torch.h>
#include <iostream>
#include<unordered_map>

int main(){      

    #ifdef PRFILE_CQ
        auto begin = std::chrono::system_clock::now();
    #endif

    CodeFetcher cf;
    // vector<string> good_ids = {"0", "1", "256", "257"};
    // vector<string> bad_ids = {"7736561", "1106356", "4612451", "5808251", "6168742", "4128663", "3912065", "6048468", "3295101", "198398", "7918977", "4286169", "4039633", "286905", "8380109", "5651066", "7121405"};
    // cout << "Good ones:" << endl;
    // unordered_map<string, vector<vector<int>>> results_good = cf.get_codes(good_ids);
    // cout << endl << "Bad ones:" << endl;
    // unordered_map<string, vector<vector<int>>> results_bad = cf.get_codes(bad_ids);

    vector<string> doc_ids = {"2801", "55025", "7736561"};

    // for (int i=0;i < 34539 -241; i+=1) doc_ids.push_back(to_string(256*i + 241));
    unordered_map<string, vector<vector<int>>> results_good = cf.get_codes(doc_ids);

    // for (auto p : results){
    //     cout << "Document id: " << p.first << " , number of tokens: " << p.second.size() << endl;
    //     // cout << "[";
    //     // for(auto& row : p.second){
    //     //     cout << "[";
    //     //     for(auto& col : row){
    //     //         cout << col << ",";
    //     //     }
    //     //     cout << "],";
    //     // }
    //     // cout << "]" << endl << endl;
    // }

    // QueryProcessor qp;
    // unordered_map<int, unordered_map<string, vector<vector<int>>>> code_map = qp.getCodes();

    // for (auto doc_data_map : code_map){
    //     cout << "Query Id: " << doc_data_map.first << ",";;
    //     for (auto p : doc_data_map.second){
    //         cout << "Document " << p.first << ": (" << p.second.size() << " tokens)" << endl << endl;
    //         // cout << "[";
    //         // for (auto d : p.second){
    //         //     cout << "[";
    //         //     for (int i = 0; i < d.size(); i++)
    //         //         cout << d.at(i) << ",";
    //         //     cout << "],";
    //         // }
    //         // cout << "]" << endl;
    //     }
    // }

    #ifdef PRFILE_CQ
        auto end = std::chrono::system_clock::now();
        std::cout<<"total execution time including loading in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
    #endif
}

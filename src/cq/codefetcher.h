#include <torch/torch.h>
#include "../config.h"
#include <string>
#include <map>
#include <vector>

using namespace std;

namespace lh{

    class CodeFetcher{
        public:
            explicit CodeFetcher();
            ~CodeFetcher();
            map<string, map<string, vector<vector<int>>>> fetch_codes();
            
        private:
           
           
    };
}
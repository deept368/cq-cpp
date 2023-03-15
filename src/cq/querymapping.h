#include<unordered_map>
#include "../config.h"
#include "../utils.h"


namespace lh{
    class QueryMapping{
        public:
            explicit QueryMapping();
            ~QueryMapping();
            string getQuery(int queryId);
            
        private:
            unordered_map<int, string>* queryMapping;
            void readQueryMapping(string queryFile);

    };
}
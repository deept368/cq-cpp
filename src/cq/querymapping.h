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
        

    };
}
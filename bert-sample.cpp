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

    ExecuteCQ cq;
    cq.execute();
    

    #ifdef PRFILE_CQ
        auto end = std::chrono::system_clock::now();
        std::cout<<"total execution time including loading in milli-seconds "<< (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000 << std::endl;
    #endif
}

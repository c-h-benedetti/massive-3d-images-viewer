#include "compare.hpp"
#include <fstream>
#include <string.h>
#include <iostream>

uint64_t compare_files(const char* f1, const char* f2, uint64_t max, bool& same){
    std::ifstream stream1, stream2;
    stream1.open(f1, std::ios::binary);
    stream2.open(f2, std::ios::binary);

    if((stream1.is_open()) and (stream2.is_open())){
        bool continuer = true;
        uint64_t count = 0;

        while(continuer){
            u_char c1 = stream1.get();
            u_char c2 = stream2.get();

            if(c1 == c2){
                if(c1 == EOF){
                    same = true;
                    continuer = false;
                }
                if(count == max-1){
                    same = true;
                    continuer = false;
                }
                count++;
            }

            else{
                same = false;
                continuer = false;
            }

            if(count % 1000 == 0){
                std::cerr << (int)(((float)count/(float)6756793344)*100) << "%  " << "(" << (int)c1 << ")" << "                \r";
            }

        }


        stream1.close();
        stream2.close();

        return count;

    }

    else{
        same = false;
        uint64_t ret = 0;
        memset(&ret, 255, 8);
        return ret;
    }

    uint64_t ret = 0;
    memset(&ret, 255, 8);
    return ret;
}

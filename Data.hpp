#ifndef DATA_BLOCK_HPP_INCLUDED
#define DATA_BLOCK_HPP_INCLUDED

#include <glm/glm.hpp>
#include "Box.hpp"
#include <vector>


class Data{
    public:
        uint8_t* data;
        Box b;
        glm::vec3 center; // Center according "real world" coordinates
        uint64_t position;
        bool loaded = false;
        uint8_t padding;

        void destroy();
        void assign_data(uint8_t* d);

        uint64_t size() const{return this->b.size();}

        Data();
        Data(uint8_t* d, Box box);
        Data(Box box, glm::vec3 c, uint64_t p, uint8_t pad);
        Data(Box box, glm::vec3 c, uint64_t p, uint8_t pad, uint8_t* d);

};

void destroy_all(std::vector<Data*>& d);

#endif //DATA_BLOCK_HPP_INCLUDED

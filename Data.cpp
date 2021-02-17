#include "Data.hpp"

void Data::destroy(){
    delete[] this->data;
    this->loaded = false;
    this->data = NULL;
}

Data::Data(){
    this->loaded = false;
    this->data = NULL;
}

Data::Data(uint8_t* d, Box box){
    this->data = d;
    this->b = box;
    this->loaded = true;
}

void Data::assign_data(uint8_t* d){
    this->data = d;
    this->loaded = true;
}

Data::Data(Box box, glm::vec3 c, uint64_t p, uint8_t pad){
    this->b = box;
    this->center = c;
    this->position = p;
    this->loaded = false;
    this->data = NULL;
    this->padding = pad;
}

Data::Data(Box box, glm::vec3 c, uint64_t p, uint8_t pad, uint8_t* d){
    this->b = box;
    this->center = c;
    this->position = p;
    this->loaded = true;
    this->data = d;
    this->padding = pad;
}

void destroy_all(std::vector<Data*>& d){
    for(u_int i = 0 ; i < d.size() ; i++){
        delete d[i];
    }
}

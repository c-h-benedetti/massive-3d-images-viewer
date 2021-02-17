#ifndef MASSIVE_IMAGES_READER_HPP_INCLUDED
#define MASSIVE_IMAGES_READER_HPP_INCLUDED

#include <string>
#include <fstream>
#include <vector>
#include "memory.h"
#include "Data.hpp"


class ImageReader{
    public:

        ImageReader();
        ImageReader(std::string name, bool aligned=false);
        ImageReader(std::string name, Box b, bool aligned=false);

        ~ImageReader();

        bool load_dims();
        bool load_data();
        uint64_t next();
        uint64_t next_aligned();
        bool extract_dims(std::ifstream& s);
        bool extract_data(std::ifstream& s);
        void load_blocks();

        void set_size(Box b, bool correct = true);
        bool is_valid() const;
        bool is_valid(uint64_t l, uint64_t c, uint64_t d) const;
        void histogram_data(std::string file) const;
        float volume(bool verbose = false) const;
        void set_overlap(uint64_t o);

        uint8_t at(uint64_t l, uint64_t c, uint64_t d, bool* b = NULL);
        std::string tell_state() const;
        std::string tell_padding() const;
        void check_padding();
        bool load_aligned(uint64_t idx, uint64_t size);

        uint8_t* get_data(){return this->data;}
        uint8_t get_padding() const{return this->padding;}

        uint64_t* get_shape() const;
        Box loaded_area() const;
        Box get_size() const{return this->shape;}

        void write_aligned_file(std::string s);
        std::vector<Data*> get_blocks() const;

        float* get_dims(float coef=1.0) const;

        bool autoclean = true;

    private:
        uint64_t    width;
        uint64_t    height;
        uint64_t    depth;
        float       dim_x;
        float       dim_y;
        float       dim_z;
        uint8_t*    data;
        Box         loaded;
        Box         save_box;
        Box         shape;
        bool        init = false;
        std::string path;
        uint64_t*   state = NULL;
        uint64_t*   limits = NULL;
        uint64_t    overlap = 0;
        uint8_t     padding = 0;
        std::ifstream stream;
        int sz_block = 1;
        std::vector<Data*> blocks;
        bool aligned_file=false;
        uint64_t idx_aligned = 0;
};

#endif //MASSIVE_IMAGES_READER_HPP_INCLUDED

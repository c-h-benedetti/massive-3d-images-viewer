#ifndef MASSIVE_IMAGES_WRITER_HPP_INCLUDED
#define MASSIVE_IMAGES_WRITER_HPP_INCLUDED

#include <string>
#include <fstream>
#include <future>
#include <vector>
#include "ImageReader.hpp"
#include "memory.h"
#include "Data.hpp"

class ImageWriter{
    public:

        bool      init();
        uint64_t  get_size() const;
        uint64_t* get_shape() const;
        bool      set_file(std::string p);
        void      set_location(Box b);
        void      add_data(Data* d);
        void      write_dims(float* dims_voxels);

        ImageWriter();
        ImageWriter(uint64_t* shape);
        ImageWriter(std::string path, uint64_t* shape, bool destroy=false, bool aligned=false);
        ImageWriter(std::string path, uint64_t* shape, float* dims, bool destroy=false, bool aligned=false);

        ~ImageWriter();

        static void to_file(std::ofstream* file, Data* data, uint64_t* shape, std::mutex* m);

    private:
        uint64_t*          shape_canvas;
        std::string        path;
        std::vector< std::future<void> > blocks;
        std::ofstream      output_ima;
        std::ofstream      output_dims;
        std::mutex         mutex_writer;
        bool               usable = false;
        bool               aligned_file = false;
};

#endif //MASSIVE_IMAGES_WRITER_HPP_INCLUDED

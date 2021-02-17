#ifndef GAUSSIAN_PYRAMID_HPP_INCLUDED
#define GAUSSIAN_PYRAMID_HPP_INCLUDED

#include "ImageReader.hpp"
#include "ImageWriter.hpp"
#include <string>

#define LowResGrid GaussianPyramid

struct Fragment;
typedef struct Fragment Fragment;

struct Fragment{
    float t;
    float alpha;

    u_char r;
    u_char g;
    u_char b;

    Fragment(float a, float b, u_char c, u_char d, u_char e);
    Fragment(float a, float b, u_char* c);
    Fragment();

    std::string to_string() const;
};

class GaussianPyramid{
    public:

        void build();
        Data* get_data();

        GaussianPyramid();
        GaussianPyramid(std::string path, int m);

        std::string last_floor;

    private:
        ImageReader* imIn;
        ImageWriter* imOut;
        int max_floor;
        int floor = 0;
        uint64_t max_memory = (1024*1024*1024);
        std::string path_from;
        Box standard_shape = Box(1024);

};

#endif //GAUSSIAN_PYRAMID_HPP_INCLUDED

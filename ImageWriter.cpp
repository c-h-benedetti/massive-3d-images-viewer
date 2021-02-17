#include "ImageWriter.hpp"
#include <string.h>
#include <iostream>


void ImageWriter::to_file(std::ofstream* file, Data* data, uint64_t* shape, std::mutex* m){
    uint64_t len_line = data->b.size_x();
    uint64_t d_max = data->b.top;
    uint64_t h_max = data->b.front;
    uint64_t shift = 0;

    // Locking mutex to book file descriptor and write in file
    std::lock_guard<std::mutex> lock(*m);

    for(uint64_t d = data->b.bottom ; d < d_max ; d++){
        for(uint64_t h = data->b.back ; h < h_max ; h++){
            uint64_t idx_start = (d * shape[0] * shape[1]) + (h * shape[0]) + data->b.left;
            file->seekp(idx_start);
            file->write((char*)(data->data + shift), len_line);
            shift += len_line;
        }
    }

    data->destroy();
    delete data;
}


bool ImageWriter::init(){
    if(this->usable){
        uint64_t sz_seg = this->shape_canvas[0];
        uint64_t nb_itrs = this->shape_canvas[1] * this->shape_canvas[2];
        char* empty = new char [sz_seg];
        memset(empty, 0, sz_seg);

        // Fills the temporary file at the correct size
        for(uint64_t i = 0 ; i < nb_itrs ; i++){
            this->output_ima.write(empty, sz_seg);
        }

        delete[] empty;
        return true;
    }
    else{
        std::cerr << "Impossible to init output, missing data" << std::endl;
        return false;
    }
}


bool ImageWriter::set_file(std::string p){
    this->path = p;
    std::string data = p + ".ima";
    std::string dims = p + ".dim";

    this->output_ima = std::ofstream(data.c_str(), std::ios::binary);
    this->output_dims = std::ofstream(dims.c_str());

    if(!this->output_ima.is_open() || !this->output_dims.is_open()){
        std::cerr << "[WRITER] (" << p << ") Failed to open files" << std::endl;
        return false;
    }
    else{
        return true;
    }
}


void ImageWriter::add_data(Data* d){
    //this->blocks.push_back(std::async(std::launch::async, to_file, &(this->output), d, this->shape_canvas, &(this->mutex_writer)));
    to_file(&(this->output_ima), d, this->shape_canvas, &(this->mutex_writer));
}


uint64_t ImageWriter::get_size() const{
    return this->shape_canvas[0] * this->shape_canvas[1] * this->shape_canvas[2];
}

uint64_t* ImageWriter::get_shape() const{
    return this->shape_canvas;
}


ImageWriter::ImageWriter(){
    this->shape_canvas = new uint64_t [3];
    memset(this->shape_canvas, 0, sizeof(uint64_t)*3);
}

ImageWriter::ImageWriter(uint64_t* shape){
    this->shape_canvas = new uint64_t [3];
    memcpy(this->shape_canvas, shape, sizeof(uint64_t)*3);
}

ImageWriter::ImageWriter(std::string path, uint64_t* shape, bool destroy, bool aligned){
    this->shape_canvas = new uint64_t [3];
    memcpy(this->shape_canvas, shape, sizeof(uint64_t)*3);
    if(destroy){
        delete[] shape;
    }
    this->set_file(path);
    this->usable = true;
    if(!aligned){
        this->init();
    }
    this->aligned_file = aligned;
}

ImageWriter::ImageWriter(std::string path, uint64_t* shape, float* dims, bool destroy, bool aligned){
    this->shape_canvas = new uint64_t [3];
    this->aligned_file = aligned;
    memcpy(this->shape_canvas, shape, sizeof(uint64_t)*3);

    this->set_file(path);
    this->write_dims(dims);
    this->usable = true;
    this->init();

    if(destroy){
        delete[] shape;
        delete[] dims;
    }
}

void ImageWriter::write_dims(float* dims_voxels){
    this->output_dims << this->shape_canvas[0] << " " << this->shape_canvas[1] << " " << this->shape_canvas[2] << std::endl;
    this->output_dims << "-type U8" << std::endl;
    this->output_dims << "-dx " << dims_voxels[0] << std::endl;
    this->output_dims << "-dy " << dims_voxels[1] << std::endl;
    this->output_dims << "-dz " << dims_voxels[2] << std::endl;
    this->output_dims.close();
}

ImageWriter::~ImageWriter(){
    this->output_ima.close();
    this->output_dims.close();
    delete[] this->shape_canvas;
}

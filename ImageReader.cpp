#include "ImageReader.hpp"
#include <fstream>
#include <iostream>
#include <string.h>
#include <algorithm>

ImageReader::ImageReader(){
    this->width = 0;
    this->height = 0;
    this->depth = 0;
    this->data = NULL;
    this->path = "";
    this->set_size(Box(0));
}

ImageReader::ImageReader(std::string name, bool aligned){
    this->path = name;
    this->data = NULL;
    this->load_dims();
    this->aligned_file = aligned;
    if(this->aligned_file){
        this->load_blocks();
    }
}

ImageReader::ImageReader(std::string name, Box b, bool aligned){
    this->path = name;
    this->data = NULL;
    this->load_dims();
    this->set_size(b);
    this->aligned_file = aligned;
}


ImageReader::~ImageReader(){
    if(this->autoclean){
        //delete[] this->data;
        destroy_all(this->blocks);
    }
    delete[] this->state;
    this->stream.close();
}

float* ImageReader::get_dims(float coef) const{
    float* voxels_size = new float [3];

    voxels_size[0] = coef * this->dim_x;
    voxels_size[1] = coef * this->dim_y;
    voxels_size[2] = coef * this->dim_z;

    return voxels_size;
}

void ImageReader::load_blocks(){
    std::string file_name = this->path + ".inf";
    std::ifstream file(file_name.c_str());

    u_int nb_iters = 0;
    file >> nb_iters;
    uint64_t left, right, back, front, bottom, top, pos;
    float x, y, z;
    uint8_t pdg;

    for(u_int i = 0 ; i < nb_iters ; i++){
        file >> left >> right >> back >> front >> bottom >> top;
        file >> x >> y >> z;
        file >> pos;
        file >> pdg;

        this->blocks.push_back(
            new Data(
                Box(left, right, back, front, bottom, top),
                glm::vec3(x, y, z),
                pos,
                pdg
            )
        );
    }

    file.close();
}

float ImageReader::volume(bool verbose) const{
    float v = 0.0;
    float v2 = 0.0;

    if(this->data){
        uint64_t size_total = this->width * this->height * this->depth;
        uint64_t size_fragment = this->loaded.size();

        v = sizeof(uint8_t) * size_fragment;
        v2 = sizeof(uint8_t) * size_total;
    }
    if(verbose){
        printf("--- %f MB (%f MB) ---\n", v/1000000, v2/1000000);
    }
    return v;
}

bool ImageReader::extract_dims(std::ifstream& s){
    // Extracting number of voxels
    s >> this->width >> this->height >> this->depth;
    std::cout << "W: " << this->width << "  |  H: " << this->height << "  |  D: " << this->depth << std::endl;

    // Extracting format of data
    std::string argument;
    std::string value_str;
    s >> argument >> value_str;
    if(value_str == "U8"){
        this->sz_block = 1;
    }
    if(value_str == "U16"){
        this->sz_block = 2;
    }

    if((value_str != "U8") && (value_str != "U16")){
        printf("Warning. Data depth not implemented\n");
    }

    // Extraction size of voxels
    s >> argument >> this->dim_x;
    s >> argument >> this->dim_y;
    s >> argument >> this->dim_z;
    printf("Vw: %f  |  Vh: %f  |  Vd: %f\n", this->dim_x, this->dim_y, this->dim_z);
    s.close();
    return true;
}

bool ImageReader::load_dims(){
    if(this->path.size() <= 0){
        puts("Error. Image name not provided");
        return false;
    }
    else{
        std::string name = this->path + ".dim";
        std::ifstream file(name.c_str());
        printf("File dims: %s\n", name.c_str());

        if(file.is_open()){
            this->extract_dims(file);
            return true;
        }
        else{
            puts("Error. Unable to open file");
            return false;
        }
    }
}


void ImageReader::histogram_data(std::string file) const{
    u_int histo[256];

    for(u_int i = 0 ; i < 256 ; i++){
        histo[i] = 0;
    }
    for(u_int i = 0 ; i < this->loaded.size() ; i++){
        histo[this->data[i]]++;
    }

    std::ofstream file_histo(file.c_str());

    if(file_histo.is_open()){
        std::cout << "Box: " << this->loaded.size() << std::endl;
        this->volume(true);
        for(u_int i = 0 ; i < 256 ; i++){
            file_histo << i << "  " << histo[i] << std::endl;
            std::cout << i << ":  " << histo[i] << std::endl;
        }
        file_histo.close();
    }

}

bool ImageReader::is_valid() const{
    bool corner1 = this->is_valid(
        this->loaded.back,
        this->loaded.left,
        this->loaded.bottom
    );

    bool corner2 = this->is_valid(
        this->loaded.front-1,
        this->loaded.right-1,
        this->loaded.top-1
    );

    return (corner1 && corner2);
}

bool ImageReader::is_valid(uint64_t l, uint64_t c, uint64_t d) const{
    if((l >= this->height) || (c >= this->width) || (d >= this->depth)){
        return false;
    }
    else{
        return true;
    }
}

bool ImageReader::extract_data(std::ifstream& s){
    uint64_t size = this->loaded.size_x(); // Size of a line (beam)
    uint64_t shift = 0; // Shift in data pointer (for filling it)

    if(this->autoclean && this->data){ // If the data pointer is already in use, flush it.
        delete[] this->data;
        this->data = NULL;
    }

    this->data = new uint8_t [this->loaded.size()];

    for(uint64_t d = this->loaded.bottom ; d < this->loaded.top ; d++){
        for(uint64_t h = this->loaded.back ; h < this->loaded.front ; h++){
            // First position of the fragment
            uint64_t index = (d * this->width * this->height) + (h * this->width) + this->loaded.left;
            s.seekg(index);
            s.read((char*)(this->data + shift), size);
            shift += size;
        }
    }

    return true;
}

bool ImageReader::load_data(){
    if(this->path.size() <= 0){
        puts("Error. Image name not provided");
        return false;
    }
    else{
        std::string name = this->path + ".ima";
        if(this->stream.is_open()){
            // Nothing to do
        }
        else{
            this->stream = std::ifstream(name.c_str(), std::ios::in | std::ios::binary);
        }

        if(this->stream.is_open()){
            this->extract_data(this->stream);
            return true;
        }
        else{
            puts("Error. Unable to open file");
            return false;
        }
    }
}

uint64_t* ImageReader::get_shape() const{
    uint64_t* s = new uint64_t [3];
    s[0] = this->width;
    s[1] = this->height;
    s[2] = this->depth;
    return s;
}

Box ImageReader::loaded_area() const{
    return this->loaded;
}

void ImageReader::check_padding(){
    this->padding = 0;

    /*if((this->state[0] == 0) && (this->limits[0] > 1)){this->padding |= Padding::Left;}
    if(this->state[0] == this->limits[0]-1){this->padding |= Padding::Right;}

    if((this->state[1] == 0) && (this->limits[1] > 1)){this->padding |= Padding::Back;}
    if(this->state[1] == this->limits[1]-1){this->padding |= Padding::Front;}

    if((this->state[2] == 0) && (this->limits[2] > 1)){this->padding |= Padding::Bottom;}
    if(this->state[2] == this->limits[2]-1){this->padding |= Padding::Top;}*/

    if(this->loaded.left == 0){this->padding |= Padding::Left;}
    if(this->loaded.right == this->width){this->padding |= Padding::Right;}

    if(this->loaded.back == 0){this->padding |= Padding::Back;}
    if(this->loaded.front == this->height){this->padding |= Padding::Front;}

    if(this->loaded.bottom == 0){this->padding |= Padding::Bottom;}
    if(this->loaded.top == this->depth){this->padding |= Padding::Top;}
}

bool ImageReader::load_aligned(uint64_t idx, uint64_t size){
    if(this->path.size() <= 0){
        puts("Error. Image name not provided");
        return false;
    }
    else{
        std::string name = this->path + ".ima";
        if(this->stream.is_open()){
            // Nothing to do
        }
        else{
            this->stream = std::ifstream(name.c_str(), std::ios::in | std::ios::binary);
        }

        if(this->stream.is_open()){
            this->data = new uint8_t [size];
            this->stream.seekg(idx);
            this->stream.read((char*)(this->data), size);
            return true;
        }
        else{
            puts("Error. Unable to read file");
            return false;
        }
    }
}

uint64_t ImageReader::next_aligned(){
    if(!this->aligned_file){
        std::cerr << "ERROR. Impossible to read that type of file, try next() instead" << std::endl;
        return 0;
    }
    else{
        if(this->idx_aligned >= this->blocks.size()){
            this->idx_aligned = 0;
            return 0;
        }
        else{
            this->load_aligned(
                this->blocks[this->idx_aligned]->position,
                this->blocks[this->idx_aligned]->size()
            );
            this->loaded = this->blocks[this->idx_aligned]->b;
            this->idx_aligned++;
            return 1;
        }
        return 0;
    }
}

// Returns the size of what's been read. (So 0 at the end of the file, or for an error)
// This function moves the reading box, and reshapes it if we are stuck to a border
// Then it flushes and refreshes the data segment from disk

uint64_t ImageReader::next(){
    if(this->aligned_file){
        std::cerr << "ERROR. Try to call next_aligned() for this type of file instead" << std::endl;
        return 0;
    }
    else{
        bool fin = false;

        if(this->init){
            // Processing new position for the box
            this->state[0]++; // Going forward through width
            if(this->state[0] >= this->limits[0]){
                this->state[0] = 0;

                this->state[1]++;
                if(this->state[1] >= this->limits[1]){
                    this->state[1] = 0;

                    this->state[2]++;
                    if(this->state[2] >= this->limits[2]){
                        this->state[2] = 0;
                        fin = true;
                    }
                }
            }

            uint64_t left   = this->state[0] * (this->shape.size_x() - this->overlap);
            uint64_t back   = this->state[1] * (this->shape.size_y() - this->overlap);
            uint64_t bottom = this->state[2] * (this->shape.size_z() - this->overlap);

            uint64_t right = ((left + this->shape.size_x()) <= this->width) ? (left + this->shape.size_x()) : (left + (this->width - left));
            uint64_t front = ((back + this->shape.size_y()) <= this->height) ? (back + this->shape.size_y()) : (back + (this->height - back));
            uint64_t top   = ((bottom + this->shape.size_z()) <= this->depth) ? (bottom + this->shape.size_z()) : (bottom + (this->depth - bottom));

            this->loaded = Box(left, right, back, front, bottom, top);
        }
        else{
            this->init = true;
        }

        this->load_data();
        uint64_t c = this->loaded.size();
        this->check_padding();

        if(fin){
            this->stream.close();
            return 0; // End of image reached
        }
        else{
            return c;
        }
    }
}

void ImageReader::set_overlap(uint64_t o){
    this->overlap = o;
}

std::string ImageReader::tell_padding() const{
    std::string retour = "Padding: ";
    if(this->padding & Padding::Left){retour += "Left  ";}
    if(this->padding & Padding::Right){retour += "Right  ";}

    if(this->padding & Padding::Back){retour += "Back  ";}
    if(this->padding & Padding::Front){retour += "Front  ";}

    if(this->padding & Padding::Bottom){retour += "Bottom  ";}
    if(this->padding & Padding::Top){retour += "Top  ";}

    return retour;
}

std::string ImageReader::tell_state() const{
    std::string s = "";
    s += ("[" + std::to_string(this->state[0]) + ", " + std::to_string(this->state[1]) + ", " + std::to_string(this->state[2]) + "]");
    s += "   ";
    s += ("[" + std::to_string(this->limits[0]) + ", " + std::to_string(this->limits[1]) + ", " + std::to_string(this->limits[2]) + "]");
    s += "\n";
    s += this->loaded.to_string();
    s += "\n";
    s += this->tell_padding();
    return s;
}


uint8_t ImageReader::at(uint64_t l, uint64_t c, uint64_t d, bool* b){
    if(this->loaded.inside(l, c, d)){
        l -= this->loaded.back;
        c -= this->loaded.left;
        d -= this->loaded.bottom;
        uint64_t index = (d * this->loaded.size_x() * this->loaded.size_y()) + (l * this->loaded.size_x()) + c;
        if(b){
            *b = true;
        }
        return this->data[index];
    }
    else{
        if(b){
            *b = false;
        }
        return 0;
    }
}

uint32_t nb_iterations(uint64_t big, uint64_t small){
    uint32_t base = big / small;
    if(big % small > 0){
        base++;
    }
    return base;
}

void ImageReader::set_size(Box b, bool correct){
    if(!this->state){
        this->state = new uint64_t [3];
    }
    if(!this->limits){
        this->limits = new uint64_t [3];
    }

    memset(this->state, 0, 3 * sizeof(uint64_t));
    memset(this->limits, 0, 3 * sizeof(uint64_t));

    if(correct){
        uint64_t width_fix  = (b.size_x() > this->width)  ? (this->width)  : (b.size_x());
        uint64_t height_fix = (b.size_y() > this->height) ? (this->height) : (b.size_y());
        uint64_t depth_fix  = (b.size_z() > this->depth)  ? (this->depth)  : (b.size_z());

        this->loaded = Box(width_fix, height_fix, depth_fix);
    }
    else{
        this->loaded = b;
    }

    this->shape = this->loaded;
    this->shape.reset();

    this->limits[0] = nb_iterations(this->width,  this->shape.size_x() - this->overlap);
    this->limits[1] = nb_iterations(this->height, this->shape.size_y() - this->overlap);
    this->limits[2] = nb_iterations(this->depth,  this->shape.size_z() - this->overlap);

    //this->padding = (Padding::Bottom | Padding::Left | Padding::Back); // Coin superieur gauche au fond
    this->check_padding();

}

std::vector<Data*> ImageReader::get_blocks() const{
    return this->blocks;
}

void ImageReader::write_aligned_file(std::string s){
    std::string ima = s + ".ima";
    std::string dim = s + ".dim";
    std::string inf = s + ".inf";

    std::ofstream aligned_dim(dim.c_str());
    aligned_dim << this->width << " " << this->height << " " << this->depth << std::endl;
    aligned_dim << "-type U8" << std::endl;
    aligned_dim << "-dx " << this->dim_x << std::endl;
    aligned_dim << "-dy " << this->dim_y << std::endl;
    aligned_dim << "-dz " << this->dim_z << std::endl;
    aligned_dim.close();


    std::ofstream aligned_file(ima.c_str(), std::ios::binary);
    uint64_t index_global = 0;

    while(this->next()){
        aligned_file.write((char*)this->get_data(), this->loaded.size());
        float x = (((float)this->loaded.left * this->dim_x) + ((float)this->loaded.right * this->dim_x)) / 2.0;
        float y = (((float)this->loaded.back * this->dim_y) + ((float)this->loaded.front * this->dim_y)) / 2.0;
        float z = (((float)this->loaded.bottom * this->dim_z) + ((float)this->loaded.top * this->dim_z)) / 2.0;
        this->blocks.push_back(
            new Data(
                this->loaded,
                glm::vec3(x, y, z),
                index_global,
                this->padding
            )
        );
        index_global += this->loaded.size();
        delete[] this->data;
    }

    aligned_file.close();

    std::ofstream info_file(inf.c_str());

    info_file << this->blocks.size() << std::endl;
    for(u_int i = 0 ; i < this->blocks.size() ; i++){
        info_file << this->blocks[i]->b.left << " " << this->blocks[i]->b.right << " " << this->blocks[i]->b.back << " " << this->blocks[i]->b.front << " " << this->blocks[i]->b.bottom << " " << this->blocks[i]->b.top << std::endl;
        info_file << this->blocks[i]->center.x << " " << this->blocks[i]->center.y << " " << this->blocks[i]->center.z << std::endl;
        info_file << this->blocks[i]->position << std::endl;
        info_file << this->blocks[i]->padding << std::endl;
    }
    info_file.close();
}

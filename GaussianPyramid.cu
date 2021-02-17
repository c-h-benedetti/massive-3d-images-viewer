#include <iostream>
#include "GaussianPyramid.hpp"

#define FILTER 3

__device__ void div_2_sup(uint64_t* a, uint64_t* b, uint64_t* c){
    (*a) += 1;
    (*b) += 1;
    (*c) += 1;

    (*a) /= 2;
    (*b) /= 2;
    (*c) /= 2;
}


__global__ void filter_and_subsample(uint8_t* data_in, uint8_t* data_out, uint8_t* padding_ptr, uint64_t* box_in, uint64_t* box_after){

    uint8_t padding = *padding_ptr;

    // Taille du bloc chargé en mémoire
    uint64_t width = gridDim.x;
    uint64_t height = gridDim.y;
    uint64_t depth = blockDim.x;


    //div_2_sup(&width_out, &height_out, &depth_out);

    uint64_t c = blockIdx.x;
    uint64_t l = blockIdx.y;
    uint64_t d = threadIdx.x;

    // Extremums of the for loop
    uint64_t min_c = c - 1;
    uint64_t min_l = l - 1;
    uint64_t min_d = d - 1;

    uint64_t max_c = c + 1;
    uint64_t max_l = l + 1;
    uint64_t max_d = d + 1;

    uint64_t c_out = c / 2;
    uint64_t l_out = l / 2;
    uint64_t d_out = d / 2;


    // Coo's globales
    uint64_t x = c + box_in[0];
    uint64_t y = l + box_in[2];
    uint64_t z = d + box_in[4];


    if((x % 2 == 0) && (y % 2 == 0) && (z % 2 == 0)){
        // Check of padding to adjust convolution. Determine which voxels will be their own center of convolution.
        uint64_t index_in  = (height * width * d) + (l * width) + c;

        uint64_t accumulateur = 0;
        uint64_t diviseur = 0;

        // = = = = = Checking padding = = = = = = =


        if(padding & Padding::Left){ // Le padding a gauche est absent
            if(c == 0){
                min_c = c; // On ramène la borne inférieure à c.
            }
        }
        else{ // Le padding a gauche est présent
            if(c == 0){ // Si on est dans la bande de padding
                return; // On abandonne simplement le calcul, il ne nous concerne pas.
            }
        }
        if(padding & Padding::Right){
            if(c == width-1){
                max_c = c;
            }
        }
        else{
            if(c == width-1){
                return;
            }
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(padding & Padding::Back){
            if(l == 0){
                min_l = l;
            }
        }
        else{
            if(l == 0){
                return;
            }
        }
        if(padding & Padding::Front){
            if(l == height-1){
                max_l = l;
            }
        }
        else{
            if(l == height-1){
                return;
            }
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(padding & Padding::Bottom){
            if(d == 0){
                min_d = d;
            }

        }
        else{
            if(d == 0){
                return;
            }
        }
        if(padding & Padding::Top){
            if(d == depth-1){
                max_d = d;
            }
        }
        else{
            if(d == depth-1){
                return;
            }
        }


        // = = = = = = = = = = = = = = = = = = = = = = =

        //div_2_sup(&width_out, &height_out, &depth_out);
        //uint64_t index_out = (height_out * width_out * d_out) + (width_out * l_out) + c_out;
        uint64_t index_out = (box_after[0] * box_after[1] * d_out) + (box_after[0] * l_out) + c_out;

        // - - - - - - - - - - - - - - - - - - - - - - - - - -

        for(uint64_t d_work = min_d ; d_work <= max_d ; d_work++){
            for(uint64_t l_work = min_l ; l_work <= max_l ; l_work++){
                for(uint64_t c_work = min_c ; c_work <= max_c ; c_work++){
                    uint64_t index = (height * width * d_work) + (l_work * width) + c_work;
                    accumulateur += data_in[index];
                    diviseur++;
                }
            }
        }

        accumulateur /= diviseur;
        data_out[index_out] = accumulateur;

    }

}

Fragment::Fragment(float a, float b, u_char c, u_char d, u_char e){
    this->t     = a;
    this->alpha = b;
    this->r     = c;
    this->g     = d;
    this->b     = e;
}

Fragment::Fragment(float a, float b, u_char* c){
    this->t     = a;
    this->alpha = b;
    this->r     = c[0];
    this->g     = c[1];
    this->b     = c[2];
}

Fragment::Fragment(){
    this->t     = 0.0;
    this->alpha = 0.0;
    this->r     = 0;
    this->g     = 0;
    this->b     = 0;
}

std::string Fragment::to_string() const{
    std::string s = "";
    s += ("(" + std::to_string(t) + ")  |  ");
    s += ("[" + std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + "; " + std::to_string(alpha) + "]");
    return s;
}

void div_2_shape(uint64_t* shape){
    shape[0]++;
    shape[1]++;
    shape[2]++;
    shape[0] /= 2;
    shape[1] /= 2;
    shape[2] /= 2;
}

void GaussianPyramid::build(){
    uint64_t size = 0;
    std::string name_from = this->path_from;
    std::string name_to = name_from + "_" + std::to_string(this->floor);
    this->last_floor = name_to;

    do{

        this->last_floor = name_to;

        this->imIn = new ImageReader(name_from);
        this->imIn->set_size(this->standard_shape);
        this->imIn->set_overlap(1); // Depends on the size of the filter. Set to 1 corresponds to hardcode the size of the filter at 3
        this->imIn->autoclean = false;

        std::cerr << "[GaussianPyramid] Generation of receptor file" << std::endl;
        uint64_t* shape_global = this->imIn->get_shape();
        float* dims_voxels = this->imIn->get_dims(2.0);

        div_2_shape(shape_global);

        this->imOut = new ImageWriter(name_to, shape_global, dims_voxels, true);
        std::cerr << "[GaussianPyramid] Starting iteration of the gaussian pyramid." << std::endl;
        size = 0; // Reset of size at each iteration


        while(this->imIn->next()){

            uint64_t sz_1d = this->imIn->loaded_area().size();
            std::cerr << this->imIn->tell_state() << std::endl;
            Box receptor = this->imIn->loaded_area().without_padding(this->imIn->get_padding());
            std::cerr << receptor.to_string() << std::endl;
            receptor.reduced_area();
            std::cerr << receptor.to_string() << std::endl;
            std::cerr << "= = = = = = = = =" << std::endl;
            size += receptor.size();
            uint64_t sz_1d_reduced = receptor.size();
            uint8_t *data_device, *receptor_device, *padding_device;
            uint64_t *box_data_device;
            uint64_t *size_after;
            uint64_t box_data[6] = {
                this->imIn->loaded_area().left,
                this->imIn->loaded_area().right,
                this->imIn->loaded_area().back,
                this->imIn->loaded_area().front,
                this->imIn->loaded_area().bottom,
                this->imIn->loaded_area().top
            };

            uint64_t box_after[3] = {
                receptor.size_x(),
                receptor.size_y(),
                receptor.size_z()
            };

            cudaMalloc((void**)&data_device, sz_1d);
            cudaMalloc((void**)&receptor_device, sz_1d_reduced);
            cudaMalloc((void**)&padding_device, 1);
            cudaMalloc((void**)&box_data_device, sizeof(uint64_t)*6);
            cudaMalloc((void**)&size_after, sizeof(uint64_t)*3);

            uint8_t padding_block = this->imIn->get_padding();
            cudaMemcpy(data_device, this->imIn->get_data(), sz_1d, cudaMemcpyHostToDevice);
            cudaMemcpy(padding_device, &padding_block, 1, cudaMemcpyHostToDevice);
            cudaMemcpy(box_data_device, box_data, sizeof(uint64_t)*6, cudaMemcpyHostToDevice);
            cudaMemcpy(size_after, box_after, sizeof(uint64_t)*3, cudaMemcpyHostToDevice);

            delete[] this->imIn->get_data();

            dim3 blocks(this->imIn->loaded_area().size_x(), this->imIn->loaded_area().size_y());
            filter_and_subsample<<<blocks, this->imIn->loaded_area().size_z()>>>(data_device, receptor_device, padding_device, box_data_device, size_after);
            cudaDeviceSynchronize();

            uint8_t* processed_data = new uint8_t [sz_1d_reduced];

            cudaMemcpy(processed_data, receptor_device, sz_1d_reduced, cudaMemcpyDeviceToHost);

            cudaFree(data_device);
            cudaFree(receptor_device);
            cudaFree(padding_device);
            cudaFree(box_data_device);
            cudaFree(size_after);

            this->imOut->add_data(
                new Data(
                    processed_data,
                    receptor
                )
            );

        }

        std::cerr << "[GaussianPyramid] Floor " << this->floor << " processed." << std::endl;
        std::cerr << "[GaussianPyramid] Current size: " << size << " (limit: " << this->max_memory << ")" << std::endl;
        this->floor++;
        name_from = name_to;
        name_to = this->path_from + "_" + std::to_string(this->floor);
        delete this->imIn;
        delete this->imOut;
    }while((this->floor < this->max_floor) && (size > this->max_memory));
}

GaussianPyramid::GaussianPyramid(std::string path, int m){
    this->imIn = nullptr;
    this->imOut = nullptr;
    this->max_floor = m;
    this->floor = 0;
    this->path_from = path;
}

GaussianPyramid::GaussianPyramid(){
    this->imIn = nullptr;
    this->imOut = nullptr;
    this->max_floor = 10;
    this->floor = 0;
}

#include <iostream>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string.h>
#include <string>
#include <SFML/Graphics.hpp>
#include <SFML/Window/Event.hpp>
#include "GaussianPyramid.hpp"
#include "memory.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/common.hpp>


#define LIMIT 1000
#define R 0
#define G 1
#define B 2
#define A 3
#define THRESHOLD 0.02f
#define RAY_MAX 1000.0f
// Nombre de niveaux de couleurs dans le rendu final:
#define NB_LEVELS 5
// Ecart entre 2 niveaux pour qu'ils soient considérés identiques
#define ECART 5

using namespace std;

#include "glm_overloads.hpp"

uint64_t frames = 0;
uint64_t NB_LOADED = 0;
glm::vec2 last_clic(0.0, 0.0);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Camera{
    public:
        u_int pixels_width;
        u_int pixels_height;
        u_int sz_bloc = 750;
        u_int C = 0;
        u_int L = 0;
        u_int iter_width;
        u_int iter_height;
        float focus;
        float ratio;
        float fov;
        float h;
        float w;
        glm::mat4 M;
        glm::mat4 base_M;
        glm::mat4 V;
        glm::mat4 reversed;

        glm::vec4 pos_cam;     // Position de la camera dans le repère actuel
        glm::vec3 pos_cam_world;
        glm::vec3 target;
        glm::vec3 forward;     // Dans quelle direction la camera regarde
        glm::vec3 up;          // Axe qui indique le haut de la camera (selon gravité)
        glm::vec3 right;       // Direction à droite de la camera
        glm::vec4 pos_light;   // Position de la lumière (en repère camera)
        glm::vec3 light_color = glm::vec3(1.0, 1.0, 1.0);
        float light_power = 1.0;

        float k_A = 0.04;
        float k_D = 0.9;
        float k_S = 0.7;
        float alpha_spec = 100.0;

        float speed_zoom = 0.01;
        float pitch = 0.0;
        float yaw = 0.0;
        float distance_camera = 1.0;
        int clipping = 0;

        uint8_t* frame_buffer_host   = nullptr;
        uint8_t* frame_buffer_device = nullptr;

        void reprocess_pos_cam(){
            this->pos_cam_world = glm::vec3(
                this->distance_camera * glm::cos(this->pitch) * glm::sin(this->yaw),
                this->distance_camera * glm::sin(this->pitch),
                this->distance_camera * glm::cos(this->pitch) * glm::cos(this->yaw)
            );
        }


        void orbital(float p, float y){
            // Values in parameter are deltas, not actual values
            this->pitch += p;
            this->yaw += y;
            this->reprocess_pos_cam();
            this->update_transformations(glm::lookAt(this->pos_cam_world, this->target, this->up));
        }

        void rotate_object(float p, float y){
            this->pitch += p;
            this->yaw += y;
            glm::mat4 mat1 = glm::rotate(y, glm::vec3(0.0, 1.0, 0.0));
            glm::mat4 mat2 = glm::rotate(p, glm::vec3(1.0, 0.0, 0.0));
            glm::mat4 m_rotate = mat1 * mat2;
            this->update_transformations(m_rotate * this->M, this->V);
        }

        void zoom_along(float direction){
            //glm::vec3 move = direction * this->speed_zoom * glm::length(this->target - this->pos_cam_world) * this->forward; // Plus on se rapproche du centre, plus on ralenti
            //this->pos_cam_world += move;
            this->distance_camera += (-1.0f * speed_zoom * direction * (this->distance_camera* 0.5f));
            this->reprocess_pos_cam();
            this->update_transformations(glm::lookAt(this->pos_cam_world, this->target, this->up));
        }



        bool next(u_int* device_angle = nullptr){
            bool fin = false;

            this->C++;
            if(this->C >= this->iter_width){
                this->C = 0;
                this->L++;
                if(this->L >= this->iter_height){
                    this->L = 0;
                    fin = true;
                }
            }

            if(device_angle){
                u_int angle[2] = {
                    this->C * this->sz_bloc,
                    this->L * this->sz_bloc
                };
                cudaMemcpy(device_angle, angle, 2 * sizeof(u_int), cudaMemcpyHostToDevice);
            }

            return fin;
        }


        void update_cam_device(Camera* cam_device){
            cudaMemcpy(cam_device, this, sizeof(Camera), cudaMemcpyHostToDevice);
        }

        Camera* to_device_cam(u_int** angle){
            Camera* device_cam;
            uint64_t nb_pixels = this->pixels_width * this->pixels_height * 4;

            cudaMalloc((void**)&frame_buffer_device, nb_pixels);
            cudaMalloc((void**)&device_cam, sizeof(Camera));
            cudaMemcpy(device_cam, this, sizeof(Camera), cudaMemcpyHostToDevice);

            NB_LOADED += sizeof(Camera);
            NB_LOADED += nb_pixels;

            cudaMalloc((void**)angle, sizeof(u_int)*2);
            cudaMemset(*angle, 0, sizeof(u_int)*2);

            return device_cam;
        }

        void calc_iters(){
            this->iter_width = this->pixels_width / this->sz_bloc;
            if(this->pixels_width % this->sz_bloc){
                this->iter_width++;
            }

            this->iter_height = this->pixels_height / this->sz_bloc;
            if(this->pixels_height % this->sz_bloc){
                this->iter_height++;
            }
        }

        Camera(){

        }

        Camera(float rt, float FOV, float focus_cam, u_int max_height){
            this->ratio         = rt;
            this->fov           = glm::radians(FOV);
            this->focus         = focus_cam;
            this->pixels_height = max_height;
            this->pixels_width  = this->ratio * max_height;
            this->calc_iters();
            this->h             = glm::tan(this->fov) * this->focus;
            this->w             = this->h * this->ratio;
            this->M             = glm::mat4(1.0f);
            this->base_M        = this->M;
            this->V             = glm::mat4(1.0f);
            this->reversed      = glm::mat4(1.0f);

            this->pos_cam       = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            this->forward       = glm::vec3(0.0f, 0.0f, -this->focus);
            this->up            = glm::vec3(0.0f, 1.0f, 0.0f);
            this->right         = glm::vec3(1.0f, 0.0f, 0.0f);
            this->pos_light     = glm::vec4(250, 250, 250, 1.0);
        }

        Camera(u_int W, u_int H, float FOV, float fcs){
            this->pixels_height = H;
            this->pixels_width  = W;
            this->calc_iters();
            this->ratio         = (float)W / (float)H;
            this->focus         = fcs;
            this->fov           = FOV;
            this->h             = glm::tan(this->fov) * this->focus;
            this->w             = this->h * this->ratio;
            this->M             = glm::mat4(1.0f);
            this->base_M        = this->M;
            this->V             = glm::mat4(1.0f);
            this->reversed      = glm::mat4(1.0f);

            this->pos_cam       = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            this->forward       = glm::vec3(0.0f, 0.0f, -this->focus);
            this->up            = glm::vec3(0.0f, 1.0f, 0.0f);
            this->right         = glm::vec3(1.0f, 0.0f, 0.0f);
            this->pos_light     = glm::vec4(250, 250, 250, 1.0);
        }


        void update_properties_view(glm::vec3 pos, glm::vec3 tgt, glm::vec3 up_v){
            this->pos_cam_world = pos;
            this->target = tgt;
            this->up = up_v;
            this->distance_camera = glm::length(pos);
            this->update_transformations(
                glm::lookAt(
                    pos,
                    tgt,
                    up_v
                )
            );
        }

        void update_properties_model(float scale_factor, glm::vec3 axes_translate){
            glm::mat4 m_t = glm::mat4(1.0);
            glm::mat4 m_s = glm::mat4(1.0);
            glm::mat4 m   = glm::mat4(1.0);
            m_t = glm::translate(m_t, glm::vec3(axes_translate.x, axes_translate.y, axes_translate.z));
            m_s = glm::scale(m_s, glm::vec3(scale_factor, scale_factor, scale_factor));
            m = m_s * m_t * m;
            this->update_transformations(m, this->V);
        }

        ~Camera(){
            delete[] frame_buffer_host;
            cudaFree(frame_buffer_device);
        }

        void update_transformations(glm::mat4 view_m){
            this->V = view_m;
            glm::mat4 V_1 = glm::inverse(view_m);
            glm::mat4 M_1 = glm::inverse(this->M);
            this->reversed = M_1 * V_1;
        }

        void update_transformations(glm::mat4 model_m, glm::mat4 view_m){
            this->M = model_m;
            this->V = view_m;
            glm::mat4 V_1 = glm::inverse(this->V);
            glm::mat4 M_1 = glm::inverse(this->M);
            this->reversed = M_1 * V_1;
        }

        void set_resolution(u_int W, u_int H){
            this->pixels_height = H;
            this->pixels_width  = W;
            this->calc_iters();
            this->ratio         = (float)W / (float)H;
            this->h             = glm::tan(this->fov) * this->focus;
            this->w             = this->h * this->ratio;
        }

        void set_fov(float FOV){
            this->fov          = FOV;
            this->h            = glm::tan(this->fov) * this->focus;
            this->w            = this->h * this->ratio;
            this->pixels_width = this->ratio * this->pixels_height;
        }

        void init_memory(){
            uint64_t size = this->pixels_width * this->pixels_height * 4;
            this->frame_buffer_host = new uint8_t [size];
            //cudaMalloc((void**)&(this->frame_buffer_device), size);
        }

        void swap_buffers(){
            uint64_t size = this->pixels_width * this->pixels_height * 4;
            cudaMemcpy(this->frame_buffer_host, this->frame_buffer_device, size, cudaMemcpyDeviceToHost);
        }

};



class Voxels{
    public:
        uint8_t* data_device = nullptr;
        float* dims_voxels = nullptr;
        uint64_t* shape = nullptr;

        uint8_t padding;

        __device__ uint64_t index_of(uint64_t c, uint64_t l, uint64_t d){
            return (this->shape[0] * this->shape[1] * d) + (this->shape[0] * l) + c;
        }

        __device__ uint8_t data_from(uint64_t c, uint64_t l, uint64_t d){
            uint64_t index = this->index_of(c, l, d);
            return this->data_device[index];
        }

        Voxels* transfer(){
            Voxels* device_data;
            cudaMalloc((void**)&device_data, sizeof(Voxels));

            uint8_t* temp_data;
            float* temp_dims;
            uint64_t* temp_shape;
            uint64_t size_data = shape[0] * shape[1] * shape[2];

            cudaMalloc((void**)&temp_data, size_data);
            cudaMalloc((void**)&temp_dims, 3 * sizeof(float));
            cudaMalloc((void**)&temp_shape, 3 * sizeof(uint64_t));


            cudaMemcpy(temp_data, data_device, size_data, cudaMemcpyHostToDevice);
            cudaMemcpy(temp_dims, dims_voxels, 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(temp_shape, shape, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);

            delete[] this->data_device;
            delete[] this->dims_voxels;
            delete[] this->shape;

            this->data_device = temp_data;
            this->dims_voxels = temp_dims;
            this->shape       = temp_shape;

            cudaMemcpy(device_data, this, sizeof(Voxels), cudaMemcpyHostToDevice);
            NB_LOADED += sizeof(Voxels);
            NB_LOADED += (3 * sizeof(float));
            NB_LOADED += (3 * sizeof(uint64_t));
            NB_LOADED += size_data;

            return device_data;
        }

        ~Voxels(){
            cudaFree(this->data_device);
            cudaFree(this->dims_voxels);
            cudaFree(this->shape);
        }
};



uint8_t mode = 0;     // 0 = partial resolution


sf::Texture canvas;
sf::Sprite  sprite;

// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// #   FONCTIONS DEVICE SIDE                                                   #
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

__device__ float hue2rgb(float p, float q, float t){

    if(t < 0.0f) t += 1.0f;
    if(t > 1.0f) t -= 1.0f;
    if(t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if(t < 1.0f/2.0f) return q;
    if(t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

__device__ void hsl_to_rgb(float h, float s, float l, glm::vec3* color_out){
    float c_R, c_G, c_B;

    if(s == 0){
        c_R = c_G = c_B = l; // achromatic
    }else{
        float q = (l < 0.5) ? (l * (1.0f + s)) : (l + s - l * s);
        float p = 2.0f * l - q;
        c_R = hue2rgb(p, q, h + 1.0f/3.0f);
        c_G = hue2rgb(p, q, h);
        c_B = hue2rgb(p, q, h - 1.0f/3.0f);
    }

    color_out->x = c_R;
    color_out->y = c_G;
    color_out->z = c_B;
}




// Function suitable only in cached mode
__device__ void in_box(uint64_t* shape, float* sz_voxels, glm::vec4* pos, bool* res, glm::ivec3* coords, float* addr_alpha, float* addr_beta, float* addr_gamma, glm::vec3* co_inter){
    float extremum_right = (float)shape[0] * sz_voxels[0]; // Largeur du bloc de voxels en unité de scène
    float extremum_front = (float)shape[1] * sz_voxels[1]; // Pronfondeur du bloc de voxels en unité de scène
    float extremum_top   = (float)shape[2] * sz_voxels[2]; // Hauteur du bloc de voxels en unité de scène

    // Le point "pos" est-il dans le bloc de voxels, sachant qu'on est en coordonnées locales ?
    if((pos->x > 0.0f) && (pos->y > 0.0f) && (pos->z > 0.0f) && (pos->x < extremum_right) && (pos->y < extremum_front) && (pos->z < extremum_top)){
        *res = true;

        // Le point est bien dedans, on reconstruit donc les coordonnées en terme de tableau
        coords->x = (u_int)(pos->x / sz_voxels[0]);
        coords->y = (u_int)(pos->y / sz_voxels[1]);
        coords->z = (u_int)(pos->z / sz_voxels[2]);
        //printf("pos: %d  %d  %d\n", coords->x, coords->y, coords->z);

        // On recherche ensuite les coordonnées "barycentriques"
        // Elles servent à interpoler la data avec les voxels voisins
        float center_x = ((float)coords->x * (float)sz_voxels[0]) + (0.5 * (float)sz_voxels[0]); //(left_v + right_v) / 2.0;
        float center_y = ((float)coords->y * (float)sz_voxels[1]) + (0.5 * (float)sz_voxels[1]); //(back_v + front_v) / 2.0;
        float center_z = ((float)coords->z * (float)sz_voxels[2]) + (0.5 * (float)sz_voxels[2]); //(bottom_v + top_v) / 2.0;

        glm::vec3 voxel_center(center_x, center_y, center_z);
        *co_inter = voxel_center;

        float bary_x = (pos->x - center_x) / (0.5f * sz_voxels[0]);
        float bary_y = (pos->y - center_y) / (0.5f * sz_voxels[1]);
        float bary_z = (pos->z - center_z) / (0.5f * sz_voxels[2]);


        if(bary_x < -1.0){bary_x = -1.0;}else if(bary_x > 1.0){bary_x = 1.0;}
        if(bary_y < -1.0){bary_y = -1.0;}else if(bary_y > 1.0){bary_y = 1.0;}
        if(bary_z < -1.0){bary_z = -1.0;}else if(bary_z > 1.0){bary_z = 1.0;}

        *addr_alpha = bary_x;
        *addr_beta  = bary_y;
        *addr_gamma = bary_z; //bary_z;
    }
    else{
        *res = false;
        *addr_alpha = 0.5;
        *addr_beta  = 0.5;
        *addr_gamma = 0.5;
    }

}

/*__device__ float max(float a, float b){
    if(a > b){return a;}
    else{return b;}
}*/

__device__ void process_normal(Voxels* voxels, glm::ivec3* cos_tableau, glm::vec3* N, float* status, int marge=1){
    float val_vox = (float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * (uint64_t)cos_tableau->y) + (uint64_t)cos_tableau->x]);

    float v4 = (cos_tableau->x - marge >= 0) ? ((float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * cos_tableau->y) + (cos_tableau->x - marge)])) : (0.0);
    float v2 = (cos_tableau->x + marge < voxels->shape[0]) ? ((float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * cos_tableau->y) + (cos_tableau->x + marge)])) : (0.0);

    float v1 = (cos_tableau->y - marge >= 0) ? ((float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * (cos_tableau->y - marge)) + (cos_tableau->x)])) : (0.0);
    float v3 = (cos_tableau->y + marge < voxels->shape[1]) ? ((float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * (cos_tableau->y + marge)) + (cos_tableau->x)])) : (0.0);

    float v5 = (cos_tableau->z - marge >= 0) ? ((float)(voxels->data_device[((cos_tableau->z - marge) * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * cos_tableau->y) + (cos_tableau->x)])) : (0.0);
    float v6 = (cos_tableau->z + marge < voxels->shape[2]) ? ((float)(voxels->data_device[((cos_tableau->z + marge) * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * cos_tableau->y) + (cos_tableau->x)])) : (0.0);

    glm::vec3 normale(0.0, 0.0, 0.0);

    normale.x = (val_vox - v2) + (val_vox - v4);
    normale.y = (v1 - val_vox) + (v3 - val_vox);
    normale.z = (val_vox - v5) + (val_vox - v6);

    *status = (glm::length(normale) > THRESHOLD) ? (1.0f) : (0.0f);

    *N = glm::normalize(normale);
}

__device__ void clamp(glm::vec4* v){
    if(v->x > 1.0f){v->x = 1.0f;}
    if(v->y > 1.0f){v->y = 1.0f;}
    if(v->z > 1.0f){v->z = 1.0f;}
}

__device__ float density_to_alpha(float density, float seuil_b=0.01, float seuil_h=0.99){
    if(density > seuil_h){
        return 0.0;
    }
    else if(density < seuil_b){
        return 0.0;
    }
    else{
        return 1.0;
    }
}

__device__ void base_color_phong(Camera* camera, Voxels* voxels, glm::ivec3* cos_tableau, uint16_t val_vox, glm::vec3* cos_intersection, glm::vec3 ray, glm::vec4* res){
    glm::vec3 base_color(0.0, 0.0, 0.0);          // Conteneur de la couleur associé à la densité
    /*if((val_vox > 25) || (val_vox < 20)){val_vox = 0;}
    else{
        val_vox = 125;
    }*/
    float color_vox = (float)val_vox / 255.0f;    // Valeur du voxel entre 0.0 et 1.0 (== opacité)
    hsl_to_rgb(color_vox, 0.8, 0.6, &base_color); // Couleur propre à la densité

    int nb_tries = 0;
    glm::vec3 N(0.0, 0.0, 0.0);
    float status_normal = 0.0;
    while((status_normal < 1.0) && (nb_tries < 6)){
        process_normal(voxels, cos_tableau, &N, &status_normal, nb_tries);
        nb_tries++;
    }

    N = glm::normalize(N);

    glm::vec3 L = glm::normalize(camera->pos_light - camera->pos_cam);

    float NdotL = glm::max(glm::dot(N, L), 0.0f);

    base_color *= NdotL;

    float opacity = NdotL * color_vox;

    *res = glm::vec4(base_color, opacity);

}


__device__ void interpolate(Voxels* voxels, glm::ivec3* cos_tableau, float alpha, float beta, float gamma, uint16_t* res){
    float val_vox = (float)(voxels->data_device[(cos_tableau->z * voxels->shape[0] * voxels->shape[1]) + (voxels->shape[0] * (uint64_t)cos_tableau->y) + ((uint64_t)cos_tableau->x)]);

    *res = val_vox;
}

__device__ bool different(float a, float b, float diff){
    float k = a - b;
    if(k < 0.0f){
        k *= -1.0f;
    }
    return k > diff;
}

__device__ bool in_iso_surface(int centre, int tolerance, uint16_t valeur){
    int val_vox = (int)valeur;
    val_vox -= centre;
    return glm::abs(val_vox) < tolerance;
}

__global__ void raycaster_cached(Camera* camera, Voxels* voxels, u_int* pos_tale){

    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
    // =   Relatif à la position d'output                                                                      =
    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

    u_int width  = camera->pixels_width;
    u_int height = camera->pixels_height;
    u_int c = blockIdx.x + pos_tale[0];
    u_int l = threadIdx.x + pos_tale[1];

    if(c >= width){return;}
    if(l >= height){return;}

    uint64_t idx_out = (4 * width * l) + (4 * c); // Dans tableau de pixels affiché

    float u = ((2.0 * (float)c) / (float)camera->pixels_width) - 1.0;   // Coefficient entre -1.0 et 1.0 selon la colonne où on se trouve
    float v = ((2.0 * (float)l) / (float)camera->pixels_height) - 1.0;  // Coefficient entre -1.0 et 1.0 selon la ligne où on se trouve

    glm::vec4 pos_canvas = camera->pos_cam + glm::vec4(camera->forward, 0) + (camera->h * v * glm::vec4(camera->up, 0)) + (camera->w * u * glm::vec4(camera->right, 0));  // Position du pixel sur le canvas (qui va servir à projeter un rayon)
    glm::vec4 ray = glm::normalize(glm::vec4(pos_canvas.x, pos_canvas.y, pos_canvas.z, 0.0f));  // Rayon normalizé de la camera vers le pixel

    float increment = 0.05f; // Distance parcourue entre 2 samples du ray-marching (le long du rayon)
    bool reached_box = false;  // Est-ce qu'on est rentré dans la bounding-box du modèle

    // Mode de projection :
    //     0: MIP
    //     1: minIP
    //     2: alpha-blending
    //     3: AIP
    //     4: Iso-surface

    u_int mode = 2;

    // Recepteur du ray-marching
    glm::vec4 buffer(0.0, 0.0, 0.0, 0.0);

    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
    // =   Data des modes de projection (algos de raymarching)                                                 =
    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

    // MIP
    uint16_t max = 0;

    // minIP
    uint16_t min = 255;

    // AIP
    double avg = 0.0;
    uint64_t counter = 1;

    // Alpha-blending
    float previous = 0;
    float transparency = 1.0f;

    // Iso-surface
    int centre = 40; // Centre de l'iso-surface
    int range = 6; // Tolerance a gauche et à droite

    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
    // =   Début du raymarching                                                                                =
    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

    for(float t_t = 0.0 ; t_t < RAY_MAX ; t_t += increment){
        // Calcul de la coordonnée à sample, en coordonnées Camera
        bool is_in = false;                                         // Le point echantilloné est-il dans le bloc de voxels
        glm::ivec3 cos_tableau(0, 0, 0);                            // Si le point est dans le bloc de voxel, à quel voxel est-ce qu'il correspond
        glm::vec4 sampling_point = pos_canvas + t_t * ray;          // Point d'échantillonage le long du rayon
        glm::vec4 point_local = camera->reversed * sampling_point;  // Passage en coordonnées du bloc de voxels (model)

        // Coordonnées "barycentrique" de l'endroit où se trouve le point d'échantillonage dans le voxel.
        // Pour l'interpoler avec ses voisins
        float b_alpha = 0;
        float b_beta = 0;
        float b_gamma = 0;
        glm::vec3 cos_intersection(0.0, 0.0, 0.0);

        // Sampling du volume
        in_box(voxels->shape, voxels->dims_voxels, &point_local, &is_in, &cos_tableau, &b_alpha, &b_beta, &b_gamma, &cos_intersection);

        // Si le point qu'on vient de sample est bien dans le volume
        if(is_in){
            if(!reached_box){
                increment = 0.03; // On affine l'increment quand on est dans le volume
            }
            else{
                increment *= 1.001;
            }
            reached_box = true;
            uint16_t recepteur = 0;
            interpolate(voxels, &cos_tableau, b_alpha, b_beta, b_gamma, &recepteur);

            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
            // =   MIP                                                                                                 =
            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

            if(mode == 0){
                if(recepteur > max){
                    max = recepteur;
                    buffer = glm::vec4((float)max/255.0f, (float)max/255.0f, (float)max/255.0f, 1.0f);
                }
            }

            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
            // =   minIP                                                                                               =
            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

            else if(mode == 1){
                if((recepteur < min) && (recepteur > 20)){
                    min = recepteur;
                    buffer = glm::vec4((float)min/255.0f, (float)min/255.0f, (float)min/255.0f, 1.0f);
                }
            }

            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
            // =   Alpha-blending -> Ombrage de Phong                                                                  =
            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

            else if(mode == 2){ // Alpha blendnig

                glm::vec3 color(0.0, 0.0, 0.0);
                float opacity = 0.0f;
                float density = (float)recepteur / 255.0f;
                density *= 6.0;

                if(different(density, previous, 0.013)){

                    glm::vec4 hit_color(0.0, 0.0, 0.0, 0.0);
                    base_color_phong(camera, voxels, &cos_tableau, recepteur, &cos_intersection, ray, &hit_color);
                    color = glm::vec3(hit_color.x, hit_color.y, hit_color.z);
                    opacity = hit_color.w;

                }

                color *= (opacity * transparency);
                transparency *= (1.0f - opacity);
                buffer += glm::vec4(color, 0.0f);

                previous = density;

                if(transparency <= 0.0f){
                    break;
                }
            }

            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
            // =   AIP                                                                                                 =
            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

            else if(mode == 3){
                if((recepteur > 0)){
                    avg += (recepteur * 3.0);
                    counter++;
                }
            }

            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =
            // =   Iso-surface                                                                                         =
            // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =

            else if(mode == 4){
                if(in_iso_surface(centre, range, recepteur)){
                    float value = (float)recepteur / 255.0f;
                    float v_left = (float)(centre - range) / 255.0f;
                    float v_right = (float)(centre + range) / 255.0f;
                    float shift = (float)(2 * range) / 255.0f;
                    value = glm::max(0.0f, value - v_left);
                    value /= shift;
                    buffer = glm::vec4(value);
                    break;
                }
            }

        }

        // Si on est tombé hors du volume
        else{
            if(reached_box){ // Si on a déjà été dans le volume et qu'on en sort, on peut arrêter de sample, on y retournera pas.
                break;
            }
        }
    }


    // = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - =


    // Si on était en AIP, on divise la valeur à la fin du rayon
    if(mode == 3){
        avg /= (double)counter;
        buffer = glm::vec4((float)avg/255.0f, (float)avg/255.0f, (float)avg/255.0f, 1.0f);
    }

    // On clamp le buffer au cas où il aurait des valeurs plus grandes que 255
    clamp(&buffer);

    // Alpha passée à 1.0 pour l'affichage
    buffer.w = 1.0f;

    camera->frame_buffer_device[idx_out + R] = 255 * buffer.x;
    camera->frame_buffer_device[idx_out + G] = 255 * buffer.y;
    camera->frame_buffer_device[idx_out + B] = 255 * buffer.z;
    camera->frame_buffer_device[idx_out + A] = 255 * buffer.w;
}



Voxels* create_cache_data(char* nom, Camera* cam, bool construct = true){
    Voxels* voxel_block = new Voxels;

    // Building Gaussian pyramid for the massive file
    std::string name_cache = " ";

    if(construct){
        GaussianPyramid g(nom, 1);
        g.build();
        name_cache = g.last_floor;
    }
    else{
        name_cache = nom;
    }

    // Opening the file containing the Gaussian pyramid
    ImageReader* imInCache = new ImageReader(name_cache);
    imInCache->set_size(Box(1212, 366, 1904));
    //imInCache->set_size(Box(1024));
    imInCache->set_overlap(0);
    imInCache->autoclean = false;

    // The last level of the pyramid can be loaded as one block in the memory
    if(imInCache->next()){
        std::cerr << "Data extracted" << std::endl;
    }
    else{
        std::cerr << "Error while extracting data" << std::endl;
    }

    // Creating matrices to apply and revert transformations
    voxel_block->shape = imInCache->get_shape();
    voxel_block->dims_voxels = imInCache->get_dims(1.0);
    voxel_block->data_device = imInCache->get_data();

    float axe_x = (voxel_block->dims_voxels[0] * (float)voxel_block->shape[0]) * -0.5;
    float axe_y = (voxel_block->dims_voxels[1] * (float)voxel_block->shape[1]) * -0.5;
    float axe_z = (voxel_block->dims_voxels[2] * (float)voxel_block->shape[2]) * -0.5;

    // Data au centre du monde (la boite englobante)
    float scale_factor = 0.75; // 0.75 pour struct normal
    cam->update_properties_model(scale_factor, glm::vec3(axe_x, axe_y, axe_z));

    std::cerr << "Axes: " << (axe_x) << ", " << (axe_y) << ", " << (axe_z) << std::endl;
    cam->update_properties_view(glm::vec3(0, 0, -axe_z * 0.5f), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

    return voxel_block;

}



void update(Camera* cam_host, Voxels* voxels_host, Camera* cam_device, Voxels* vox_device, u_int* pos_tale){
    //do{
    raycaster_cached<<<cam_host->sz_bloc, cam_host->sz_bloc>>>(cam_device, vox_device, pos_tale);  // Lance les calculs
    //}while(cam_host->next(pos_tale));
    cam_host->next(pos_tale);
    cudaDeviceSynchronize();

    cudaError_t cErr = cudaGetLastError();
    if(cErr){
        puts(cudaGetErrorString(cErr));
    }


    cam_host->swap_buffers();  // Rapatrie la data du buffer device vers le buffer host
    canvas.update(cam_host->frame_buffer_host);
    sprite.setTexture(canvas);
    //printf("Refresh %ld          \r", frames);
    //frames++;
}


void process_events(sf::RenderWindow* window, Camera* cam_host, Camera* cam_device){
    sf::Event event;

    while (window->pollEvent(event)) {
        switch(event.type){
            case(sf::Event::Closed):
                window->close();
                break;

            case(sf::Event::KeyPressed):
                break;

            case(sf::Event::MouseWheelMoved):
                cam_host->zoom_along(event.mouseWheel.delta);
                cam_host->update_cam_device(cam_device);
                break;
            case(sf::Event::MouseButtonPressed):
                last_clic.x = event.mouseButton.x;
                last_clic.y = event.mouseButton.y;
                break;

            case(sf::Event::MouseButtonReleased):
                glm::vec2 nouv_pos(event.mouseButton.x, event.mouseButton.y);
                glm::vec2 diff = nouv_pos - last_clic;
                float rate_x = 3.141592 * (diff.x / (float)cam_host->pixels_width);
                float rate_y = 3.141592 * (diff.y / (float)cam_host->pixels_height);
                cam_host->orbital(rate_y, rate_x);
                cam_host->update_cam_device(cam_device);
                break;
        }
    }

}

int main(int argc, char* argv[], char* env[]) {

    int coef = 140;
    Camera* camera = new Camera(coef*16, coef*9, 37.0f, 1.0f);
    Voxels* data = create_cache_data(argv[1], camera, false);
    u_int* pos_tale;
    Camera* cam_device = camera->to_device_cam(&pos_tale);
    camera->init_memory();
    Voxels* voxels_device = data->transfer();

    std::cerr << "Busy CUDA VRAM: " << (float)NB_LOADED / 1000000.0 << "MB" << std::endl;

    sf::RenderWindow App(sf::VideoMode(camera->pixels_width, camera->pixels_height), "Large Voxels Grids Viewer", sf::Style::Titlebar | sf::Style::Close);
    sprite.setPosition(sf::Vector2f(0.0f, 0.0f));
    canvas.create(camera->pixels_width, camera->pixels_height);

    //App.setFramerateLimit(1);

    while (App.isOpen()) {
        process_events(&App, camera, cam_device);
        App.clear();
        update(camera, data, cam_device, voxels_device, pos_tale);
        App.draw(sprite);
        App.display();
    }


    delete camera;
    delete data;
    cudaFree(cam_device);
    cudaFree(voxels_device);
    return 0;
}

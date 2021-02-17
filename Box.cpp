#include "Box.hpp"
#include <iostream>

Box::Box(){
    this->left = 0;
    this->right = 0;

    this->top = 0;
    this->bottom = 0;

    this->back = 0;
    this->front = 0;
}

Box::Box(uint64_t x) : Box(x, x, x){}

Box::Box(uint64_t x, uint64_t y, uint64_t z){
    this->left = 0;
    this->right = x;

    this->top = z;
    this->bottom = 0;

    this->back = 0;
    this->front = y;
}

// Coin de depart puis boite aux dimensions
Box::Box(uint64_t x, uint64_t y, uint64_t z, Box b){
    this->left = x;
    this->right = x + (b.right - b.left);

    this->top = z + (b.top - b.bottom);
    this->bottom = z;

    this->back = y;
    this->front = y + (b.front - b.back);
}

Box::Box(uint64_t x1, uint64_t x2, uint64_t y1, uint64_t y2, uint64_t z1, uint64_t z2){
    this->left = x1;
    this->right = x2;

    this->top = z2;
    this->bottom = z1;

    this->back = y1;
    this->front = y2;
}

void Box::reduced_area(){
    // Division by 2, ceiled:
    this->left   += 1;
    this->back   += 1;
    this->bottom += 1;

    this->right += 1;
    this->front += 1;
    this->top   += 1;

    this->left   /= 2;
    this->back   /= 2;
    this->bottom /= 2;

    this->right /= 2;
    this->front /= 2;
    this->top   /= 2;
}

Box Box::without_padding(uint8_t padding) const{
    Box without_padding = Box(*this);

    if(!(padding & Padding::Left)){
        without_padding.left++;
    }
    if(!(padding & Padding::Right)){
        without_padding.right--;
    }

    if(!(padding & Padding::Back)){
        without_padding.back++;
    }
    if(!(padding & Padding::Front)){
        without_padding.front--;
    }

    if(!(padding & Padding::Bottom)){
        without_padding.bottom++;
    }
    if(!(padding & Padding::Top)){
        without_padding.top--;
    }

    return without_padding;
}

uint64_t Box::size_x() const{
    return this->right - this->left;
}

uint64_t Box::size_y() const{
    return this->front - this->back;
}

uint64_t Box::size_z() const{
    return this->top - this->bottom;
}

uint64_t Box::size() const{
    return this->size_x() * this->size_y() * this->size_z();
}

void Box::reset(){
    this->right = this->size_x();
    this->top = this->size_z();
    this->front = this->size_y();
    this->left = this->back = this->bottom = 0;
}

void Box::nextX(){
    this->left += this->size_x();
    this->right += this->size_x();
}

void Box::previousX(){
    if((this->left < this->size_x()) || (this->right < this->size_x())){
        std::cerr << "Warning! Trying to go in negative with unsigned int. Clamping to 0" << std::endl;
        this->left = 0;
        this->right = 0;
    }
    else{
        this->left -= this->size_x();
        this->right -= this->size_x();
    }
}

void Box::nextY(){
    this->front += this->size_y();
    this->back += this->size_y();
}

void Box::previousY(){
    if((this->front < this->size_y()) || (this->back < this->size_y())){
        std::cerr << "Warning! Trying to go in negative with unsigned int. Clamping to 0" << std::endl;
        this->front = 0;
        this->back = 0;
    }
    else{
        this->front -= this->size_y();
        this->back -= this->size_y();
    }
}

void Box::nextZ(){
    this->bottom += this->size_z();
    this->top += this->size_z();
}

void Box::previousZ(){
    if((this->bottom < this->size_z()) || (this->top < this->size_z())){
        std::cerr << "Warning! Trying to go in negative with unsigned int. Clamping to 0" << std::endl;
        this->bottom = 0;
        this->top = 0;
    }
    else{
        this->bottom -= this->size_z();
        this->top -= this->size_z();
    }
}


std::string Box::to_string() const{
    std::string s = "";
    s += ("X: [" + std::to_string(this->left) + "  ->  " + std::to_string(this->right) + "]\n");
    s += ("Y: [" + std::to_string(this->back) + "  ->  " + std::to_string(this->front) + "]\n");
    s += ("Z: [" + std::to_string(this->bottom) + "  ->  " + std::to_string(this->top) + "]\n");

    return s;
}

Box& Box::operator+=(const Box& b){
    this->left += b.left;
    this->right += b.right;

    this->top += b.top;
    this->bottom += b.bottom;

    this->front += b.front;
    this->back += b.back;

    return (*this);
}

bool Box::inside(uint64_t l, uint64_t c, uint64_t d) const{
    if(l < this->back){
        return false;
    }
    if(l >= this->front){
        return false;
    }
    if(c < this->left){
        return false;
    }
    if(c >= this->right){
        return false;
    }
    if(d < this->bottom){
        return false;
    }
    if(d >= this->top){
        return false;
    }
    return true;
}

Box& Box::operator-=(const Box& b){
    this->left -= b.left;
    this->right -= b.right;

    this->top -= b.top;
    this->bottom -= b.bottom;

    this->front -= b.front;
    this->back -= b.back;
    return (*this);
}

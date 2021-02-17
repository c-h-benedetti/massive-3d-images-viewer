#ifndef BOX_HPP_INCLUDED
#define BOX_HPP_INCLUDED

#include <string>


enum Padding : uint8_t{
    Left = 1 << 5 ,
    Right = 1 << 4,
    Top = 1 << 3,
    Bottom = 1 << 2,
    Front = 1 << 1,
    Back = 1
};


class Box{
    public:
        Box();
        Box(uint64_t x);
        Box(uint64_t x, uint64_t y, uint64_t z);
        Box(uint64_t x1, uint64_t x2, uint64_t y1, uint64_t y2, uint64_t z1, uint64_t z2);
        Box(uint64_t x, uint64_t y, uint64_t z, Box b);

        uint64_t left;
        uint64_t right;
        uint64_t top;
        uint64_t bottom;
        uint64_t front;
        uint64_t back;


        uint64_t size_x() const;
        uint64_t size_y() const;
        uint64_t size_z() const;
        uint64_t size() const;

        void reset();
        void nextX();
        void previousX();
        void nextY();
        void previousY();
        void nextZ();
        void previousZ();

        Box without_padding(uint8_t padding) const;
        void reduced_area();

        bool inside(uint64_t l, uint64_t c, uint64_t d) const;

        Box& operator+=(const Box& b);
        Box& operator-=(const Box& b);

        std::string to_string() const;
};

#endif //BOX_HPP_INCLUDED

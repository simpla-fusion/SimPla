//
// Created by salmon on 16-6-30.
//

#include <iostream>
#include <stdint.h>

union id_type
{
    struct { int16_t x, y, z, w; };
    int64_t v;
};

id_type operator+(id_type const &x, id_type const &y)
{
    return id_type{.v=x.v + y.v};
}

id_type operator-(id_type const &first, id_type const &second)
{
    return id_type{
            static_cast<int16_t >(first.x - second.x),
            static_cast<int16_t >(first.y - second.y),
            static_cast<int16_t >(first.z - second.z),
            first.w};
}

int main(int argc, char **argv)
{
    id_type u{1, 2, 3, 4};
    id_type u2[2] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8}
    };
    id_type a, b;
    a = u + u2[0];
    b = u - u2[1];
    std::cout << std::hex << u.v << std::endl;

    std::cout << std::hex << u2[0].v << std::endl;
    std::cout << std::hex << u2[1].v << std::endl;
    std::cout << std::hex << a.v << std::endl;
    std::cout << std::hex << b.v << std::endl;
}
//
// Created by salmon on 16-12-28.
//

#include <simpla/utilities/Array.h>
#include <simpla/utilities/nTuple.ext.h>
#include <simpla/utilities/nTuple.h>

#include <iostream>

using namespace simpla;

int main(int argc, char **argv) {
    typedef Array<double, 1> a1_t;
    typedef Array<double, 3> a3_slow_t;
    typedef Array<double, 3> a3_fast_t;

    index_box_type inner_box{{0, 0, 1}, {4, 5, 5}};
    Array<double, 3> a(inner_box);
    Array<double, 3> b(inner_box);
    Array<double, 3> c(inner_box);
    Array<double, 3> d(inner_box);
    Array<double, 3> e(inner_box);

    a.Clear();
    b.Clear();
    c.Clear();
    for (index_type i = -2; i < 6; ++i)
        for (index_type j = -2; j < 7; ++j) {
            a(i, j, 0) = i + j;
            b(i, j, 0) = i * j;
        }

    c = a + b * 3;

    //    e = a(I + 1, J) - a(I - 1, J) + a(I, J + 1) - a(I - 1, J - 1);
    //    nTuple<double, 3> v = {1, 2, 3};
    //    Array<nTuple<double, 3>, 3> d(4, 5, 2);
    //    d = c * v;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    std::cout << e << std::endl;

    std::cout << "DONE" << std::endl;
}
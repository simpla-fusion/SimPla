//
// Created by salmon on 16-12-28.
//

#include <simpla/utilities/Array.h>
#include <simpla/utilities/ArrayNTuple.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/nTuple.ext.h>
#include <simpla/utilities/nTuple.h>
#include <iostream>

using namespace simpla;

int main(int argc, char **argv) {
    typedef Array<double, 1> a1_t;
    typedef Array<double, 3> a3_slow_t;
    typedef Array<double, 3> a3_fast_t;

    index_box_type inner_box{{0, 0, 0}, {4, 5, 5}};
    Array<double, 3> a(inner_box);
    Array<double, 3> b(inner_box);
    Array<double, 3> c(inner_box);
    Array<double, 3> e(inner_box);

    a.Clear();
    b.Fill(1);
    c.Fill(2);

    //    for (index_type i = 0; i < 4; ++i)
    //        for (index_type j = 0; j < 5; ++j)
    //            for (index_type k = 0; k < 5; ++k) { b(i, j, k) = i + j + k; }
    //    std::cout << b << std::endl;
    c = a + b * 2;
    try {
        c = a + b * 3;
    } catch (std::runtime_error const &error) { VERBOSE << error.what() << std::endl; }

    //    d.Clear();
    //    e.Clear();
    //    c = a(IdxShift{1, 0, 0}) - a(IdxShift{-1, 0, 0}) + a(IdxShift{0, 1, 0}) - a(IdxShift{0, -1, 0});
    //    nTuple<double, 3> v = {1, 2, 3};
    //    Array<nTuple<double, 3>, 3> d(inner_box);
    //    d = b * v;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << e << std::endl;
    //    std::cout << d << std::endl;

    std::cout << "DONE" << std::endl;
}
//
// Created by salmon on 16-12-28.
//

#include <iostream>
#include "simpla/algebra/Array.h"
#include "simpla/algebra/ExpressionTemplate.h"
#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"

using namespace simpla;

int main(int argc, char **argv) {
    index_box_type inner_box{{0, 0, 0}, {1, 5, 4}};
    Array<double> a(inner_box);
    Array<double> b(inner_box);
    Array<double> c(inner_box);
    Array<double> d(inner_box);

    a = [](index_type i, index_type j, index_type k) { return i + j + k; };

    b = [](index_type i, index_type j, index_type k) { return j * k; };

    c = a + b * 2;

    std::cout << " a = " << a << std::endl;
    std::cout << " b = " << b << std::endl;
    std::cout << " c = " << c << std::endl;
    //
    //    std::cout << " d = " << c << std::endl;
    //    FE_CMD(c = a + sin(b) * 3 + d);

    //    FE_CMD(c = a + sin(b) * 3 + d);
    //    c = a(IdxShift{0, 1, 0}) - a(IdxShift{0, -1, 0}) + a(IdxShift{0, 0, 1}) - a(IdxShift{0, 0, -1});
    //    //    nTuple<double, 3> v = {1, 2, 3};
    //    //    Array<nTuple<double, 3>, 3> e(inner_box);
    //    //    e = b * v;
    std::cout << a << std::endl;
    std::cout << c << std::endl;

    std::cout << "DONE" << std::endl;
}
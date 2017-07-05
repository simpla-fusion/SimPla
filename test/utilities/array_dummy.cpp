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
    index_box_type inner_box{{0, 0, 0}, {4, 5, 5}};
    Array<double> a(inner_box);
    Array<double> b(inner_box);
    Array<double> c(inner_box);
    Array<double> d(inner_box);

    a = [](index_type i, index_type j, index_type k) { return i + j + k; };

    b = [](index_type i, index_type j, index_type k) { return i * j * k; };

    d.SetUndefined();

    c = a + b * 2;

    std::cout << " a = " << a << std::endl;
    std::cout << " b = " << b << std::endl;
    std::cout << " c = " << c << std::endl;

    std::cout << " d = " << c << std::endl;

    //    FE_CMD(c = a + sin(b) * 3 + d);

    c = calculus::getValue(a, IdxShift{1, 0, 0}) - calculus::getValue(a, IdxShift{-1, 0, 0}) +
        calculus::getValue(a, IdxShift{0, 1, 0}) - calculus::getValue(a, IdxShift{0, -1, 0});
    //    nTuple<double, 3> v = {1, 2, 3};
    //    Array<nTuple<double, 3>, 3> e(inner_box);
    //    e = b * v;

    std::cout << c << std::endl;

    std::cout << "DONE" << std::endl;
}
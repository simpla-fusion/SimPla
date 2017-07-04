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

    a.Fill(1);
    b.Fill(2);
//    d.SetUndefined();

    c = a + b * 2;

    std::cout << " a = " << a << std::endl;
    std::cout << " b = " << b << std::endl;
    std::cout << " c = " << c << std::endl;

    //    FE_CMD(c = a + sin(b) * 3 + d);

    std::cout << " d = " << d << std::endl;

    //    d.Clear();
    //    e.Clear();
    //    c = a(IdxShift{1, 0, 0}) - a(IdxShift{-1, 0, 0}) + a(IdxShift{0, 1, 0}) - a(IdxShift{0, -1, 0});
    //    nTuple<double, 3> v = {1, 2, 3};
    //    Array<nTuple<double, 3>, 3> e(inner_box);
    //    e = b * v;
    //
    //    std::cout << e << std::endl;

    std::cout << "DONE" << std::endl;
}
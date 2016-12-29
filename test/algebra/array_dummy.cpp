//
// Created by salmon on 16-12-28.
//

#include <iostream>
#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/Array.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/algebra/ArrayExt.h>

#include <simpla/algebra/Expression.h>
#include <simpla/algebra/Arithmetic.h>

using namespace simpla;
using namespace simpla::algebra;

int main(int argc, char **argv)
{
    typedef Array<double, 1> a1_t;
    typedef Array<double, 3, true> a3_slow_t;
    typedef Array<double, 3, false> a3_fast_t;


    Array<double, 3> a(4, 5, 2);
    Array<double, 3> b(4, 5, 2);
    Array<double, 3> c(4, 5, 2);

    a.clear();
    b.clear();
    c.clear();

    a = 1;
    b = 2;
    c = a;
//
    c += a + b * 3;

    c(1, 2, 3) = 100;
//    nTuple<double, 3> v = {1, 2, 3};
//    Array<nTuple<double, 3>, 3> d(4, 5, 2);
//    d = c * v;

    std::cout << a << std::endl;

    std::cout << b << std::endl;

    std::cout << c << std::endl;
}
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

    size_type dims[4] = {4, 5, 6, 7};
    size_type s[4] = {1, 2, 3, 4};
//    std::cout
//            << "slow first= " << calculus::calculator<Array<double, 3, true> >::hash(dims, 1, 2, 3) << "  == "
//            << calculus::calculator<Array<double, 3, true> >::hash(dims, s) << "  == "
//            << (1 * 5 + 2) * 6 + 3
//            << std::endl
//            << "fast first= " << calculus::calculator<Array<double, 3, false>>::hash(dims, 1, 2, 3) << "  == "
//            << calculus::calculator<Array<double, 3, false> >::hash(dims, s) << "  == "
//            << (3 * 5 + 2) * 4 + 1
//            << std::endl;
//
//    std::cout
//            << "slow first= " << calculus::calculator<Array<double, 3, true> >::hash(dims, 0, 0, 0) << std::endl
//            << "fast first= " << calculus::calculator<Array<double, 3, false>>::hash(dims, 0, 0, 0) << std::endl;
//
//    std::cout
//            << "slow first= " << calculus::calculator<Array<double, 3, true> >::hash(dims, 3, 4, 5) << std::endl
//            << "fast first= " << calculus::calculator<Array<double, 3, false>>::hash(dims, 3, 4, 5) << std::endl;
//
//
//    std::cout
//            << "slow first= " << calculus::calculator<Array<double, 2, true> >::hash(dims, 3, 4) << std::endl
//            << "fast first= " << calculus::calculator<Array<double, 2, false>>::hash(dims, 3, 4) << std::endl;
//
//    std::cout
//            << "slow first= " << calculus::calculator<Array<double, 1, true> >::hash(dims, 3) << std::endl
//            << "fast first= " << calculus::calculator<Array<double, 1, false>>::hash(dims, 3) << std::endl;

    Array<double, 3> a(4, 5, 2);
    Array<double, 3> b(4, 5, 2);
    Array<double, 3> c(4, 5, 2);
    a.clear();
    b.clear();
    c.clear();

    a = 1;
    b = 2;
    c = a;

    c += a + b * 3;

//    nTuple<double, 3> v = {1, 2, 3};
//    Array<nTuple<double, 3>, 3> d(4, 5, 2);
//    d = c * v;

    std::cout << a << std::endl;

    std::cout << b << std::endl;

    std::cout << c << std::endl;
}
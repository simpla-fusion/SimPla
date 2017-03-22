//
// Created by salmon on 16-12-28.
//

#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/Array.h>
#include <simpla/algebra/nTuple.h>
#include <iostream>

#include <simpla/algebra/Arithmetic.h>
#include <simpla/algebra/Expression.h>

using namespace simpla;
using namespace simpla::algebra;

int main(int argc, char **argv) {
    typedef Array<double, 1> a1_t;
    typedef Array<double, 3> a3_slow_t;
    typedef Array<double, 3> a3_fast_t;

    Array<double, 2> a{4, 5};
    Array<double, 2> b{4, 5};
    Array<double, 2> c{4, 5};

    a.Clear();
    b.Clear();
    c.Clear();
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 5; ++j) {
            a(i, j) = i;
            b(i, j) = j;
        }

    c = a + b * 3;

    auto d = c(I + 1, J);

    //    CHECK(c.GetInnerIndexBox());
    //    CHECK(c.GetOuterIndexBox());
    //    CHECK(d.GetInnerIndexBox());
    //    CHECK(d.GetOuterIndexBox());

    //    nTuple<double, 3> v = {1, 2, 3};
    //    Array<nTuple<double, 3>, 3> d(4, 5, 2);
    //    d = c * v;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
}
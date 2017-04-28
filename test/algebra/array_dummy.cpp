//
// Created by salmon on 16-12-28.
//

#include <simpla/algebra/Algebra.h>
#include <simpla/utilities/Array.h>
#include <simpla/utilities/nTuple.h>
#include <iostream>

#include <simpla/algebra/Arithmetic.h>
#include <simpla/algebra/Expression.h>

using namespace simpla;
using namespace simpla::algebra;

int main(int argc, char **argv) {
    typedef Array<double, 1> a1_t;
    typedef Array<double, 3> a3_slow_t;
    typedef Array<double, 3> a3_fast_t;

    auto inner_box = std::make_tuple(nTuple<index_type, 2>{0, 0}, nTuple<index_type, 2>{4, 5});
    auto outer_box = std::make_tuple(nTuple<index_type, 2>{-2, -2}, nTuple<index_type, 2>{6, 7});
    Array<double, 2> a(inner_box, outer_box);
    Array<double, 2> b(inner_box, outer_box);
    Array<double, 2> c(inner_box, outer_box);
    Array<double, 2> d(inner_box, outer_box);
    Array<double, 2> e(inner_box, outer_box);

    CHECK(a.GetInnerIndexBox());
    CHECK(a.GetOuterIndexBox());

    a.Clear();
    b.Clear();
    c.Clear();
    for (index_type i = -2; i < 6; ++i)
        for (index_type j = -2; j < 7; ++j) {
            a(i, j) = i + j;
            b(i, j) = i * j;
        }

    c = a + b * 3;

    e = a(I + 1, J) - a(I - 1, J) + a(I, J + 1) - a(I - 1, J - 1);
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
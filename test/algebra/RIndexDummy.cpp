#include <type_traits>

//
// Created by salmon on 17-1-9.
//
#include <simpla/mpl/integer_sequence.h>
#include <simpla/toolbox/Log.h>

#include <iostream>
using namespace simpla;
int main(int argc, char **argv) {
    CHECK((index_sequence<1, 2, 3>()));
    CHECK((_0));
    CHECK((_1));
    CHECK((-_1));
    CHECK((-_1 * _2));
    CHECK(((_1, _2, _3) + (_4, _5)));
    CHECK(((_0, _0, _0) + (_4, _5)));

    CHECK(((_4, _5) - (_1, _2, _3)));

    CHECK((_2 - _1));
    CHECK(((_1, _0, _3)));

    ;
}
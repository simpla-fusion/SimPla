//
// Created by salmon on 17-1-12.
//

#include <simpla/concept/CheckConcept.h>
#include <iostream>
using namespace simpla;

struct Foo {
    typedef double value_type;
};
struct Goo {
    typedef int value_type;
};
struct Koo {};

CHECK_MEMBER_TYPE(value_type, value_type)

int main(int argc, char** argv) {
    std::cout << " Foo::value_type " << (has_value_type<Foo>::value ? "is" : "is not") << " defined"
              << std::endl;
    std::cout << " Goo::value_type " << (has_value_type<Goo>::value ? "is" : "is not") << " defined"
              << std::endl;
    std::cout << " Koo::value_type " << (has_value_type<Koo>::value ? "is" : "is not") << " defined"
              << std::endl;

    std::cout << " Foo::value_type "
              << (std::is_same<value_type_t<Foo>, double>::value ? "is" : "is not") << " double"
              << std::endl;

    std::cout << " Goo::value_type "
              << (std::is_same<value_type_t<Goo>, double>::value ? "is" : "is not") << " double"
              << std::endl;

    std::cout << " Koo::value_type "
              << (std::is_same<value_type_t<Koo>, double>::value ? "is" : "is not") << " double"
              << std::endl;
}
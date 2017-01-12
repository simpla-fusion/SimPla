//
// Created by salmon on 17-1-12.
//

#include <simpla/concept/CheckConcept.h>
#include <iostream>
using namespace simpla;

struct Foo {
    typedef std::true_type is_foo;
};
struct Goo {
    typedef std::false_type is_foo;
};
struct Koo {};

CHECK_MEMBER_TYPE_BOOLEAN(is_foo, is_foo)

int main(int argc, char** argv) {
    std::cout << " Foo  " << (is_foo<Foo>::value ? "is" : "is not") << " foo." << std::endl;
    std::cout << " Goo  " << (is_foo<Goo>::value ? "is" : "is not") << " foo." << std::endl;
    std::cout << " Koo  " << (is_foo<Koo>::value ? "is" : "is not") << " foo." << std::endl;

}
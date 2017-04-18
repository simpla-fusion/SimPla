#include <type_traits>

//
// Created by salmon on 17-1-9.
//
#include <simpla/utilities/integer_sequence.h>
#include <simpla/utilities/Log.h>

#include <iostream>

struct Foo {
    static constexpr int iform = 1;
};

template <typename _T>
struct iform {
   private:
    template <typename U>
    static auto test(int) -> std::integral_constant<int ,U::iform>;
    template <typename>
    static  std::integral_constant<int ,0> test(...);

   public:
    static constexpr int value = decltype(test<_T>(0))::value;
};
using namespace simpla;
int main(int argc, char **argv) {
    std::cout << iform<Foo>::value << std::endl;

    ;
}
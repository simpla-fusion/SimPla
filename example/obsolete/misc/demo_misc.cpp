/**
 * @file demo_misc.cpp
 *
 * @date 2015-4-15
 * @author salmon
 */
#include <iostream>
#include <type_traits>


//namespace traits{
template<typename T> auto
check(T const &f, typename std::enable_if<std::is_arithmetic<T>::value>::type *p = nullptr)
{
    return "  is primary value";
}

template<typename T> auto
check(T const &f, typename std::enable_if<!std::is_arithmetic<T>::value>::type *p = nullptr)
{
    return "  is not primary value";
}


struct Foo { };


int main()
{
    double a;
    int b;

    Foo foo;

    std::cout << "a   " << check(a) << std::endl;
    std::cout << "b   " << check(b) << std::endl;
    std::cout << "foo   " << check(foo) << std::endl;

}

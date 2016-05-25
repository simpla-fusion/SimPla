/**
 * @file demo_misc.cpp
 *
 * @date 2015-4-15
 * @author salmon
 */
#include <iostream>
#include <type_traits>


template<typename T>
inline std::enable_if_t<std::is_same<std::remove_cv_t<T>, double>::value, auto> check(T const &f)
{
    return "this is double";
}
//
//template<typename T>
//inline auto check(T const &f,
//                  std::enable_if_t<std::is_same<std::remove_cv_t<T>, int>::value, void> _t = std::enable_if_t<std::is_same<std::remove_cv_t<T>, int>::value, void>())
//{
//    return "this is int";
//}


int main()
{
    double a;
    int b;

    std::cout << "a is " << check(a) << std::endl;
    std::cout << "b is " << check(b) << std::endl;

}

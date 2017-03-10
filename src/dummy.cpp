//
// Created by salmon on 15-7-1.
//


#include <iostream>

namespace foo
{
struct A
{
    template<typename TL, typename TR>
    A(TL const &l, TR const &r):a(l + r)
    {
        std::cout << " l= " << l << " r= " << r << std::endl;
    }

    template<typename TR>
    A(A const &l, TR const &r):a(l.a + r)
    {
        std::cout << " l.a= " << l.a << " r= " << r << std::endl;
    }

    A() {}

    A(int v) : a(v) {}

    ~A() {}

    A &operator=(A const &other)
    {
        a = other.a;
    }

    int a;
};

template<typename TL, typename TR>
A operator+(TL const &l, TR const &r)
{
    A t(l, r);
    return (t);
};

}

int main(int argc, char **argv)
{
    foo::A a(3);
    foo::A b;
    b = a + (4 + 5);
    std::cout << 5 + 6 << std::endl;
}
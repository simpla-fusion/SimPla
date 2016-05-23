/**
 * @file demo_misc.cpp
 *
 * @date 2015-4-15
 * @author salmon
 */
#include <iostream>

struct U;

struct Foo
{
    U const &m_u_;

    Foo(U const &u) : m_u_(u) { }

    ~Foo() { }
};

struct U
{
    int a{123345};
    Foo f1{*this};
    Foo f2{*this};


};


int main()
{

    U w;
    std::cout << w.f1.m_u_.a << std::endl;
}

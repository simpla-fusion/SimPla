//
// Created by salmon on 17-1-4.
//
#include <iostream>
template<typename T>
auto foo(T &d)
{
    d += 1;
    return d;
}

int main(int argc, char **argv)
{
    double a = 10;
    std::cout<<a<<std::endl;
    foo(a);
    std::cout<<a<<std::endl;
}
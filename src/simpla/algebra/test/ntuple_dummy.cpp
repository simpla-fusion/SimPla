//
// Created by salmon on 16-12-26.
//
#include <iostream>
#include <complex>
#include "../nTuple.h"

using namespace simpla;

int main(int argc, char **argv)
{

    double a[3] = {1, 2, 3};
    double b[3] = {1, 2, 3};
    double c[3] = {1, 2, 3};

    for (int i = 0; i < 3; ++i)
    {
        a[i] = b[i] + c[i];
    }
    nTuple<Real, 3> t0 = {0, 0, 0};
    nTuple<Real, 2, 3> t1 = {{0, 1, 2}, {3, 4, 5}};
    nTuple<std::complex<Real>, 2, 3> t2 = {{0, 1, 2}, {3, 4, 5}};

    std::cout << "{" << t1[0][0] << " , " << t1[0][1] << " , " << t1[0][2] << "} , {" << t1[1][0] << " , " << t1[1][1]
              << " , " << t1[1][2] << "} " << std::endl;
    t1 *= 2;
    std::cout << "{" << t1[0][0] << " , " << t1[0][1] << " , " << t1[0][2] << "} , {" << t1[1][0] << " , " << t1[1][1]
              << " , " << t1[1][2] << "} " << std::endl;
    nTuple<double, 3> A, B, C;

    A = B + C;

}
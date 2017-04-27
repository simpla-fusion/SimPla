//
// Created by salmon on 16-12-26.
//
#include <simpla/SIMPLA_config.h>

#include <simpla/utilities/nTuple.h>

#include <complex>
#include <iostream>
using namespace simpla;

int main(int argc, char** argv) {
    nTuple<Real, 3> a{1.0};
    nTuple<Real, 3> b{2.0};
    nTuple<Real, 3> c;
    c = (b + 1) ;
    std::cout << c << std::endl;
    //    double a[3] = {1, 2, 3};
    //    double b[3] = {1, 2, 3};
    //    double c[3] = {1, 2, 3};
    //
    //    for (int i = 0; i < 3; ++i) { a[i] = b[i] + c[i]; }
    //    nTuple<Real, 3> t0 = {0, 0, 0};
    //    nTuple<Real, 2, 3> t1 = {{0, 1, 2}, {3, 4, 5}};
    //    nTuple<std::complex<Real>, 2, 3> t2 = {{0, 1, 2}, {3, 4, 5}};
    //
    //    std::cout << "t1 = " << t1 << std::endl;
    //
    //    std::cout << "t2 = " << t2 << std::endl;
    //
    //    //    t1 *= 2;
    //
    //    std::cout << "t1 = " << t1 << std::endl;
    //
    //    double d[2][3] = {{10, 20, 30}, {40, 50, 60}};
    //
    //    nTuple<double, 2, 3> B = {{4, 5, 6}, {7, 8, 9}};
    //    nTuple<double, 2, 3> A, C;
    //    A = 0;
    //    C = d;
    //
    //    A =  (B + 1)/ (C -3) ;
    //
    //    std::cout << "A=" << A << std::endl;
    //    std::cout << "B=" << B << std::endl;

    //    Real res = dot(A, B);
    //    Real res2 = dot(A, B) / 2;
    //    std::cout << "dot(A , B  ) =" << res << std::endl;
    //    std::cout << "dot(A , B  ) / 2 =" << res2 << std::endl;
    //
    //    std::cout << "A=" << A << std::endl;
    //
    //    nTuple<Real, 3> v = {1, 2, 3};
    //    nTuple<Real, 3> u = {4, 5, 6};
    //    nTuple<Real, 3> w = {9, 8, 7};
    //    std::cout << static_cast<nTuple<Real, 3> >(cross(v, u)) << std::endl;
    //    std::cout << static_cast<Real>(dot(cross(v, u), cross(v, u))) << std::endl;
    //    std::cout << static_cast<Real>(dot(cross(v, u), w)) << std::endl;
    //
    //    nTuple<Real, 2> v2 = {1, 2};
    //    nTuple<Real, 2> u2 = {1, 2};
    //    std::cout << static_cast<Real>(dot(v2, u2)) << std::endl;
}
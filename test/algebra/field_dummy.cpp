//
// Created by salmon on 16-12-26.
//

#include "simpla/algebra/all.h"

#include "simpla/predefine/CartesianGeometry.h"
#include "simpla/predefine/CalculusPolicy.h"

using namespace simpla;
using namespace simpla::algebra;

int main(int argc, char **argv)
{
    index_type dims[3] = {1, 4, 3};
    Real xmin[3] = {0, 0, 0};
    Real xmax[3] = {1, 2, 3};
//        m->dimensions(dims);
//        m->box(xmin, xmax);

//    size_type gw[3] = {2, 2, 2};
//    index_type lo[3] = {0, 0, 0};
//    index_type hi[3];//= {dims[0], dims[1], dims[2]}
//

    typedef mesh::CartesianGeometry mesh_type;

    mesh_type m(nullptr, &dims[0]);

    Field<Real, mesh_type> f(&m);
    Field<Real, mesh_type> g(&m);


    f.clear();
    g.clear();
//    std::cout << f << std::endl;
//    std::cout << g << std::endl;

    f(0, 2, 3) = 1990;
    f = 1;
    g = 2;
    f.assign([&](point_type const &x)
             {
                 std::cout << x << std::endl;
                 return x[0];
             });
//    f = f + g;

    std::cout << f << std::endl;
    Field<Real, mesh_type, EDGE> E(&m);
    Field<Real, mesh_type, VERTEX> rho(&m);
    E.clear();
    rho.clear();
    rho += diverge(E);
}
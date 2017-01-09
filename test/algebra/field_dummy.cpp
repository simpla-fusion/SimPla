//
// Created by salmon on 16-12-26.
//

#include "simpla/algebra/all.h"

#include "simpla/predefine/CalculusPolicy.h"
#include "simpla/predefine/CartesianGeometry.h"

using namespace simpla;
using namespace simpla::algebra;

int main(int argc, char** argv) {
    index_type dims[3] = {2, 4, 3};
    Real xmin[3] = {0, 0, 0};
    Real xmax[3] = {1, 2, 3};
    //        m->dimensions(dims);
    //        m->box(xmin, xmax);

    size_type gw[3] = {2, 2, 2};
    index_type lo[3] = {0, 0, 0};
    index_type hi[3] = {dims[0], dims[1], dims[2]};

    typedef mesh::CartesianGeometry mesh_type;

    mesh_type m(nullptr, &dims[0]);
    m.deploy();
    Field<mesh_type, Real> f(&m);
    Field<mesh_type, Real> g(&m);

    f.clear();
    g.clear();

    f(0, 2, 3, 1) = 1990;
    f = 1;
    g = 2;

    f = f + g;
    f = -g * 0.2;

    CHECK(f);
    CHECK(g);

    f = [&](point_type const& x) { return x[1] + x[2]; };

    g = [&](mesh::MeshEntityId const& s) { return 1.0; };

    CHECK(f);
    Field<mesh_type, Real, EDGE> E(&m);
    Field<mesh_type, Real, VERTEX> rho(&m);

    E.clear();
    rho.clear();

    E = [&](point_type const& x) { return x; };
    CHECK(E);
    rho = diverge(E);

    diverge(E);
}
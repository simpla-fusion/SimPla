//
// Created by salmon on 16-12-26.
//

#include <simpla/mesh/Attribute.h>
#include "simpla/algebra/all.h"
#include "simpla/predefine/CalculusPolicy.h"
#include "simpla/predefine/CartesianGeometry.h"

using namespace simpla;
using namespace simpla::algebra;
using namespace simpla::mesh;

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
    m.Deploy();
    FieldAttribute<mesh_type, Real> f(&m);
    FieldAttribute<mesh_type, Real> g(&m);

    f.clear();
    g.clear();

    f(0, 2, 3) = 1990;
    f = 1;
    g = 2;

    f = f + g;
    f = -g * 0.2;

    CHECK(f);
    //    CHECK(g);

    f = [&](point_type const& x) { return x[1] + x[2]; };

    g = [&](mesh::MeshEntityId const& s) { return 1.0; };

    //    CHECK(f);
    FieldAttribute<mesh_type, Real, EDGE> E(&m);
    FieldAttribute<mesh_type, Real, FACE> B(&m);
    FieldAttribute<mesh_type, Real, VERTEX, 3> d(&m);
    FieldAttribute<mesh_type, Real, VERTEX> rho(&m);
    FieldAttribute<mesh_type, Real, VERTEX, 8> v(&m);
    E.clear();
    rho.clear();

    E = [&](point_type const& x) { return x; };
    //    CHECK(E);
    rho = diverge(E);

    //    E[I] = 2;

    //    B[2][I] = (E[0][-1_i, 1_j] * v[0][I] - E[0][J - 1] * v[0][J - 1] + E[1][I] * v[1][I] -
    //               E[1][I - 1] * v[1][I - 1]) /
    //              v[3][I];
    //    CHECK(E);
    diverge(E);
}